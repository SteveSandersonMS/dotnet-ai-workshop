using Microsoft.Extensions.AI;

namespace Chat;

internal static class BasicCompletion
{
    public static async Task RunAsync(IChatClient chatClient)
    {
        // Basic completion
        var response = await chatClient.GetResponseAsync("Explain how real AI compares to sci-fi AI in max 20 words.");
        Console.WriteLine(response.Text);
        Console.WriteLine($"Tokens used: in={response.Usage?.InputTokenCount}, out={response.Usage?.OutputTokenCount}");

        // Streaming completion
        var responseStream = chatClient.GetStreamingResponseAsync("Explain how real AI compares to sci-fi AI in max 200 words.");
        await foreach (var message in responseStream)
        {
            Console.Write(message.Text);
        }
    }
}
