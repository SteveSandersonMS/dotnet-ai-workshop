using System.ComponentModel;

using CorrectiveRetrievalAugmentedGenerationApp.Search;

using Microsoft.Extensions.AI;

using Planner;

using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace CorrectiveRetrievalAugmentedGenerationApp;

public class ChatbotThread(
    IChatClient chatClient,
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    QdrantClient qdrantClient,
    Product currentProduct,
    ISearchTool searchTool
    )
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

        Dictionary<ulong, Chunk> closestChunksById = closestChunks.ToDictionary(c => c.Id.Num, c => new Chunk
        (
            Id: c.Id.Num,
            Text: c.Payload["text"].StringValue,
            ProductId: (int)c.Payload["productId"].IntegerValue,
            PageNumber: (int)c.Payload["pageNumber"].IntegerValue)
        );

        Dictionary<ulong, Chunk> chunksForResponseGeneration = [];

        // calculate relevancy

        ContextRelevancyEvaluator contextRelevancyEvaluator = new(chatClient);
        double averageScore = 0;
        foreach (var retrievedContext in closestChunksById.Values)
        {
            var score = await contextRelevancyEvaluator.EvaluateAsync(userMessage, retrievedContext.Text, cancellationToken);
            if (score.ContextRelevance!.ScoreNumber >= 0.7)
            {
                averageScore += score.ContextRelevance!.ScoreNumber;
                chunksForResponseGeneration.Add(retrievedContext.Id, retrievedContext);
            }
        }

        averageScore /= chunksForResponseGeneration.Count;
        // perform corrective retrieval if needed

        if (chunksForResponseGeneration.Count == 0 || averageScore < 0.7)
        {
            var planGenerator = new PlanGenerator(chatClient);

            var toolCallingClient = new FunctionInvokingChatClient(chatClient);
            var stepExecutor = new PlanExecutor(toolCallingClient);

            var evaluator = new PlanEvaluator(chatClient);

            string task = $"""
                           Given the <user_question>, search the product manuals for relevant information.
                           Look for information that may answer the question, and provide a response based on that information.
                           The <context> was not enough to answer the question. Find the information that can complement the context to address the user question.
                           
                           Take into account the user is enquiring about 
                           ProductId: ${currentProduct.ProductId}
                           Brand: ${currentProduct.Brand}
                           Model: ${currentProduct.Model}
                           Description: ${currentProduct.Description}

                           <user_question>
                           {userMessage}
                           </user_question>

                           <context>
                           {string.Join("\n", closestChunksById.Values.Select(c => $"<manual_extract id='{c.Id}'>{c.Text}</manual_extract>"))}
                           </context>
                           """;

            var plan = await planGenerator.GeneratePlanAsync(
                task
                , cancellationToken);

            List<PlanStepExecutionResult> pastSteps = [];

            // pass bing search ai function so that the executor can search web for additional material


            async Task<string> SearchTool([Description("The questions we want to answer searching the web")] string userQuestion)
            {
                var results = await searchTool!.SearchWebAsync(userQuestion, 3, cancellationToken);
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($" Web Search: {userQuestion}");
                foreach (SearchResult searchResult in results)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"""
                                        Corrective step using data from :{searchResult.Url}
                                        Preview:
                                            {searchResult.Snippet.Substring(0, Math.Min(100, searchResult.Snippet.Length))}....
                                      
                                      """);
                }
                return string.Join("\n", results.Select(c => $"""
                                                              ## web page: {c.Url}
                                                              # Content
                                                              {c.Snippet}

                                                              """));
            }

            var options = new ChatOptions
            {
                Tools =
                [
                    AIFunctionFactory.Create(SearchTool, name: "web_search",
                        description: "This tools searches the web for answers")
                ],
                ToolMode = ChatToolMode.Auto
            };

            var maxSteps = plan.Steps.Length * 2;
            while (maxSteps > 0)
            {
                var res = await stepExecutor.ExecutePlanStep(plan, options: options, cancellationToken: cancellationToken);
                pastSteps.Add(res);
                maxSteps--;
                var planOrResult = await evaluator.EvaluatePlanAsync(task, plan, pastSteps, cancellationToken);
                if (planOrResult.Plan is not null)
                {
                    plan = planOrResult.Plan;
                }
                else
                {
                    // Add the result to context
                    if (planOrResult.Result is { } result)
                    {
                        var maxKey = chunksForResponseGeneration.Count == 0 ? 0 : chunksForResponseGeneration.Keys.Max() ;
                        var fakeId = maxKey + 1;
                        chunksForResponseGeneration[fakeId] = new Chunk(
                            Id: fakeId,
                            Text: result.Outcome,
                            ProductId: currentProduct.ProductId,
                            PageNumber: 1
                        );
                    }

                    break;
                }
            }
        }

        // Log the closest manual chunks for debugging (not using ILogger because we want color)
        //Console.WriteLine("Retrieved chunks via rag");
        //foreach (var chunk in closestChunks)
        //{
        //    Console.ForegroundColor = ConsoleColor.DarkYellow;
        //    Console.WriteLine($"[Score: {chunk.Score:F2}, File: {chunk.Payload["productId"].IntegerValue}.pdf, Page: {chunk.Payload["pageNumber"].IntegerValue}");
        //    Console.ForegroundColor = ConsoleColor.DarkGray;
        //    Console.WriteLine(chunk.Payload["text"].StringValue);
        //}

        //Console.WriteLine("Chunks relevant to the question");
        //foreach (var chunk in chunksForResponseGeneration.Values)
        //{
        //    Console.ForegroundColor = ConsoleColor.Green;
        //    Console.WriteLine($"[ File: {chunk.ProductId}.pdf, Page: {chunk.PageNumber}");
        //    Console.ForegroundColor = ConsoleColor.DarkGray;
        //    Console.WriteLine(chunk.Text);
        //}


        // Now ask the chatbot
        _messages.Add(new(ChatRole.User, $$"""
            Give an answer using ONLY information from the following product manual extracts.
            If the product manual doesn't contain the information, you should say so. Do not make up information beyond what is given.
            Whenever relevant, specify manualExtractId to cite the manual extract that your answer is based on.

            {{string.Join(Environment.NewLine, chunksForResponseGeneration.Select(c => $"<manual_extract id='{c.Value.Id}'>{c.Value.Text}</manual_extract>"))}}

            User question: {{userMessage}}
            Respond as a JSON object in this format: {
                "ManualExtractId": numberOrNull,
                "ManualQuote": stringOrNull, // The relevant verbatim quote from the manual extract, up to 10 words
                "AnswerText": string
            }
            """));

        ChatResponse<ChatBotAnswer> response = await chatClient.GetResponseAsync<ChatBotAnswer>(_messages, cancellationToken: cancellationToken);

        _messages.AddMessages(response);

        if (response.TryGetResult(out ChatBotAnswer? answer))
        {
            // If the chatbot gave a citation, convert it to info to show in the UI
            Citation? citation = answer.ManualExtractId.HasValue && chunksForResponseGeneration.TryGetValue((ulong)answer.ManualExtractId, out var chunk)
                ? new Citation(chunk.ProductId, chunk.PageNumber, answer.ManualQuote ?? "")
                : null;

            return (answer.AnswerText, citation, chunksForResponseGeneration.Values.Select(v => v.Text).ToArray());
        }

        return ("Sorry, there was a problem.", null, chunksForResponseGeneration.Values.Select(v => v.Text).ToArray());

    }

    public record Citation(int ProductId, int PageNumber, string Quote);
    private record ChatBotAnswer(int? ManualExtractId, string? ManualQuote, string AnswerText);
    private record Chunk(ulong Id, string Text, int ProductId, int PageNumber);
}
