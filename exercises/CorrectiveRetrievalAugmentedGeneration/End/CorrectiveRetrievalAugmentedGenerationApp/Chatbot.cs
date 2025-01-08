using Microsoft.Extensions.AI;
using Microsoft.Extensions.Hosting;
using Qdrant.Client;

namespace CorrectiveRetrievalAugmentedGenerationApp;

public class Chatbot(
    IChatClient chatClient,
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    QdrantClient qdrantClient)
    : IHostedService
{
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        Product currentProduct = Helpers.GetCurrentProduct();
        ChatbotThread thread = new(chatClient, embeddingGenerator, qdrantClient, currentProduct);

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"Assistant: Hi! You're looking at the {currentProduct.Model}. What do you want to know about it?");

        while (!cancellationToken.IsCancellationRequested)
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("\nYou: ");
            string? userMessage = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(userMessage))
            {
                continue;
            }

            (string Text, ChatbotThread.Citation Citation, string[] AllContext) answer = (await thread.AnswerAsync(userMessage, cancellationToken))!;

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"Assistant: {answer.Text}\n");

            // Show citation if given
            if (answer.Citation is { } citation)
            {
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"CITATION: {citation.ProductId}.pdf page {citation.PageNumber}: {citation.Quote}");
            }
        }
    }

    public Task StopAsync(CancellationToken cancellationToken)
        => Task.CompletedTask;
}
