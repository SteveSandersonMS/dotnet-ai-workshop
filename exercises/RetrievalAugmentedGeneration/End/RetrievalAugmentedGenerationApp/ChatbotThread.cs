using System.ComponentModel;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace RetrievalAugmentedGenerationApp;

public class ChatbotThread(
    IChatClient chatClient,
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    QdrantClient qdrantClient,
    Product currentProduct)
{
    private List<ChatMessage> _messages =
    [
        new ChatMessage(ChatRole.System, $"""
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
        var userMessageEmbedding = await embeddingGenerator.GenerateEmbeddingVectorAsync(userMessage, cancellationToken: cancellationToken);
        var closestChunks = await qdrantClient.SearchAsync(
            collectionName: "manuals",
            vector: userMessageEmbedding.ToArray(),
            filter: Qdrant.Client.Grpc.Conditions.Match("productId", currentProduct.ProductId),
            limit: 5, cancellationToken: cancellationToken); // TODO: Evaluate with more or less
        var allContext = closestChunks.Select(c => c.Payload["text"].StringValue).ToArray();

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

        var response = await chatClient.GetResponseAsync<ChatBotAnswer>(_messages, cancellationToken: cancellationToken);
        _messages.AddMessages(response);

        if (response.TryGetResult(out var answer))
        {
            // If the chatbot gave a citation, convert it to info to show in the UI
            var citation = answer.ManualExtractId.HasValue && closestChunks.FirstOrDefault(c => c.Id.Num == (ulong)answer.ManualExtractId) is { } chunk
                ? new Citation((int)chunk.Payload["productId"].IntegerValue, (int)chunk.Payload["pageNumber"].IntegerValue, answer.ManualQuote ?? "")
                : default;

            return (answer.AnswerText, citation, allContext);
        }
        else
        {
            return ("Sorry, there was a problem.", default, allContext);
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
        var response = await chatClient.GetResponseAsync(_messages, chatOptions, cancellationToken: cancellationToken);
        _messages.AddMessages(response);
        var answer = ParseResponse(response.Text);

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
        var searchPhraseEmbedding = (await embeddingGenerator.GenerateAsync([searchPhrase]))[0];
        var closestChunks = await qdrantClient.SearchAsync(
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
        var citationRegex = new Regex(@"<cite extractId='(\d+)' productId='\d*'>(.+?)</cite>");
        if (citationRegex.Match(text) is { Success: true, Groups: var groups } match
            && int.TryParse(groups[1].ValueSpan, out var extractId))
        {
            return new(extractId, groups[2].Value, citationRegex.Replace(text, string.Empty));
        }

        return new(default, default, text);
    }
}
