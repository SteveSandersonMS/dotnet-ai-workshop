using System.ComponentModel;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;
using Planner;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace CorrectiveRetrievalAugmentedGenerationApp;

public class ChatbotThread(
    IChatClient chatClient,
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    QdrantClient qdrantClient,
    Product currentProduct)
{
    private readonly List<ChatMessage> _messages =
    [
        new(ChatRole.System, $"""
                              You are a helpful assistant, here to help customer service staff answer questions they have received from customers.
                              The support staff member is currently answering a question about this product:
                              ProductId: ${currentProduct.ProductId}
                              Brand: ${currentProduct.Brand}
                              Model: ${currentProduct.Model}
                              """),
        /*
        Answer the user question using ONLY information found by searching product manuals.
            If the product manual doesn't contain the information, you should say so. Do not make up information beyond what is
            given in the product manual.
            
            If this is a question about the product, ALWAYS search the product manual before answering.
            Only search across all product manuals if the user explicitly asks for information about all products.
        */
    ];

    public async Task<(string Text, Citation? Citation, string[] AllContext)> AnswerAsync(string userMessage, CancellationToken cancellationToken = default)
    {
        // For a simple version of RAG, we'll embed the user's message directly and
        // add the closest few manual chunks to context.
        ReadOnlyMemory<float> userMessageEmbedding = await embeddingGenerator.GenerateEmbeddingVectorAsync(userMessage, cancellationToken: cancellationToken);
        IReadOnlyList<ScoredPoint> closestChunks = await qdrantClient.SearchAsync(
            collectionName: "manuals",
            vector: userMessageEmbedding.ToArray(),
            filter: Conditions.Match("productId", currentProduct.ProductId),
            limit: 3, cancellationToken: cancellationToken); // TODO: Evaluate with more or less
        
        var chunksById = closestChunks.ToDictionary(c => c.Id.Num, c => new
        {
            Id = c.Id.Num,
            Text = c.Payload["text"].StringValue,
            ProductId = (int)c.Payload["productId"].IntegerValue,
            PageNumber = (int)c.Payload["pageNumber"].IntegerValue
        });

        // calculate relevancy

        ContextRelevancyEvaluator contextRelevancyEvaluator = new(chatClient);

        List<string> allContext = [];

        foreach (var retrievedContext in chunksById.Values)
        {
            var score = await contextRelevancyEvaluator.EvaluateAsync(userMessage, retrievedContext.Text, cancellationToken);
            if (score.ContextRelevance!.ScoreNumber > 0.7)
            {
                allContext.Add(retrievedContext.Text);
            }
        }

        // perform corrective retrieval if needed

        if (allContext.Count < 2)
        {
            var planGenerator = new PlanGenerator(chatClient);

            var toolCallingClient = new FunctionInvokingChatClient(chatClient);
            var stepExecutor = new PlanExecutor(toolCallingClient);

            var evaluator = new PlanEvaluator(chatClient);

            string task = $"""
                           Given the <user_question>, search the product manuals for relevant information.
                           Look for information that may answer the question, and provide a response based on that information.
                           The <context> was not enough to answer the question. Find the information that can complement the context to address the user question

                           <user_question>
                           {userMessage}
                           </user_question>

                           <context>
                           {string.Join("\n", chunksById.Values.Select(c => $"<manual_extract id='{c.Id}'>{c.Text}</manual_extract>"))}
                           </context>
                           """;
            var plan = await planGenerator.GeneratePlanSync(
                task
                , cancellationToken);

            List<PanStepExecutionResult> pastSteps = [];

            var res = await  stepExecutor.ExecutePlanStep(plan, cancellationToken: cancellationToken);
            pastSteps.Add(res);

            var planOrResult = await evaluator.EvaluatePlanAsync(task, plan, pastSteps, cancellationToken);

            while (planOrResult.Plan is not null)
            {
                // pass bing search ai function so that the executor can search web for additional material
                res = await stepExecutor.ExecutePlanStep(plan, cancellationToken: cancellationToken);
                pastSteps.Add(res);

                planOrResult = await evaluator.EvaluatePlanAsync(task, plan, pastSteps, cancellationToken);
            }

            if (planOrResult.Result is not null)
            {
                allContext.Add(planOrResult.Result.Outcome);
            }

        }
        /*
        // Log the closest manual chunks for debugging (not using ILogger because we want color)
        foreach (var chunk in closestChunks)
        {
            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.WriteLine($"[Score: {chunk.Score:F2}, File: {chunk.Payload["productId"].IntegerValue}.pdf, Page: {chunk.Payload["pageNumber"].IntegerValue}");
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine(chunk.Payload["text"].StringValue);
        }
        */

        // Now ask the chatbot
        _messages.Add(new(ChatRole.User, $$"""
            Give an answer using ONLY information from the following product manual extracts.
            If the product manual doesn't contain the information, you should say so. Do not make up information beyond what is given.
            Whenever relevant, specify manualExtractId to cite the manual extract that your answer is based on.

            {{string.Join(Environment.NewLine, closestChunks.Select(c => $"<manual_extract id='{c.Id}'>{c.Payload["text"].StringValue}</manual_extract>"))}}

            User question: {{userMessage}}
            Respond as a JSON object in this format: {
                "ManualExtractId": numberOrNull,
                "ManualQuote": stringOrNull, // The relevant verbatim quote from the manual extract, up to 10 words
                "AnswerText": string
            }
            """));

        bool isOllama = chatClient.GetService<OllamaChatClient>() is not null;
        ChatCompletion<ChatBotAnswer> response = await chatClient.CompleteAsync<ChatBotAnswer>(_messages, cancellationToken: cancellationToken, useNativeJsonSchema: isOllama);
        _messages.Add(response.Message);

        if (response.TryGetResult(out ChatBotAnswer? answer))
        {
            // If the chatbot gave a citation, convert it to info to show in the UI
            Citation? citation = answer.ManualExtractId.HasValue && chunksById.TryGetValue((ulong)answer.ManualExtractId, out var chunk) 
                ? new Citation(chunk.ProductId,chunk.PageNumber, answer.ManualQuote ?? "")
                : null;

            return (answer.AnswerText, citation, allContext.ToArray());
        }
        else
        {
            return ("Sorry, there was a problem.", null, allContext.ToArray());
        }

        /*
        var chatOptions = new ChatOptions
        {
            Tools = [AIFunctionFactory.Create(ManualSearchAsync)]
        };

        _messages.Add(new(ChatRole.User, $$"""
            User question: {{userMessage}}
            Respond in plain text with your answer. Where possible, also add a citation to the product manual
            as an XML tag in the form <cite extractId='number' productId='number'>short verbatim quote</cite>.
            """));
        var response = await chatClient.CompleteAsync(_messages, chatOptions, cancellationToken: cancellationToken);
        _messages.Add(response.Message);
        var answer = ParseResponse(response.Message.Text!);

        // If the chatbot gave a citation, convert it to info to show in the UI
        var citation = answer.ManualExtractId.HasValue
            && (await qdrantClient.RetrieveAsync("manuals", (ulong)answer.ManualExtractId.Value)) is { } chunks
            && chunks.FirstOrDefault() is { } chunk
            ? new Citation((int)chunk.Payload["productId"].IntegerValue, (int)chunk.Payload["pageNumber"].IntegerValue, answer.ManualQuote ?? "")
            : default;

        return (answer.AnswerText, citation);
        */
    }

    [Description("Searches product manuals")]
    private async Task<SearchResult[]> ManualSearchAsync(
        [Description("The product ID, or null to search across all products")] int? productIdOrNull,
        [Description("The search phrase or keywords")] string searchPhrase)
    {
        Embedding<float> searchPhraseEmbedding = (await embeddingGenerator.GenerateAsync([searchPhrase]))[0];
        IReadOnlyList<ScoredPoint> closestChunks = await qdrantClient.SearchAsync(
            collectionName: "manuals",
            vector: searchPhraseEmbedding.Vector.ToArray(),
            filter: productIdOrNull is { } productId ? Qdrant.Client.Grpc.Conditions.Match("productId", productId) : (Filter?)default,
            limit: 5);
        return closestChunks.Select(c => new SearchResult((int)c.Id.Num, (int)c.Payload["productId"].IntegerValue, c.Payload["text"].StringValue)).ToArray();
    }

    public record Citation(int ProductId, int PageNumber, string Quote);
    private record SearchResult(int ManualExtractId, int ProductId, string ManualExtractText);
    private record ChatBotAnswer(int? ManualExtractId, string? ManualQuote, string AnswerText);

    private static ChatBotAnswer ParseResponse(string text)
    {
        Regex citationRegex = new(@"<cite extractId='(\d+)' productId='\d*'>(.+?)</cite>");
        if (citationRegex.Match(text) is { Success: true, Groups: var groups } match
            && int.TryParse(groups[1].ValueSpan, out int extractId))
        {
            return new(extractId, groups[2].Value, citationRegex.Replace(text, string.Empty));
        }

        return new(null, null, text);
    }
}
