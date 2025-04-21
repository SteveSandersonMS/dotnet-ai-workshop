using Microsoft.Extensions.AI;
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
        new(ChatRole.System,
            $"""
            You are a helpful assistant, here to help customer service staff answer questions they have received from customers.
            The support staff member is currently answering a question about this product:
            ProductId: ${currentProduct.ProductId}
            Brand: ${currentProduct.Brand}
            Model: ${currentProduct.Model}
            """),
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

        var closestChunksById = closestChunks.ToDictionary(
            c => c.Id.Num,
            c => new Chunk(
                Id: c.Id.Num,
                Text: c.Payload["text"].StringValue,
                ProductId: (int)c.Payload["productId"].IntegerValue,
                PageNumber: (int)c.Payload["pageNumber"].IntegerValue));

        // For basic RAG, we just add *all* the chunks to context, ignoring relevancy
        var chunksForResponseGeneration = closestChunksById.Values.ToDictionary(c => c.Id, c => c);

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

        var response = await chatClient.GetResponseAsync<ChatBotAnswer>(_messages, cancellationToken: cancellationToken);
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
