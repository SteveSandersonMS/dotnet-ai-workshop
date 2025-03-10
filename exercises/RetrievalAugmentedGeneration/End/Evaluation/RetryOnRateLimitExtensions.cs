using System.ClientModel;
using Microsoft.Extensions.AI;

namespace Evaluation;

/// <summary>
/// You don't need to do anything with this. It causes an IChatClient pipeline to auto-retry if you hit an HTTP 429 (rate limit) error.
/// This could happen if you're using too much parallelism or have set a low rate limit on an Azure OpenAI deployment.
/// If you want to use something like this for real, you should probably add a maximum number of retries to avoid infinite loops.
/// </summary>
public static class RetryOnRateLimitExtensions
{
    public static ChatClientBuilder UseRetryOnRateLimit(this ChatClientBuilder builder)
        => builder.Use(next => new RetryOnRateLimitChatClient(next));

    private class RetryOnRateLimitChatClient(IChatClient innerClient) : DelegatingChatClient(innerClient)
    {
        public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> chatMessages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            while (true)
            {
                try
                {
                    return await base.GetResponseAsync(chatMessages, options, cancellationToken);
                }
                catch (ClientResultException ex) when (ex.Message.Contains("HTTP 429"))
                {
                    Console.WriteLine("Rate limited exceeded. Retrying in 3 seconds");
                    await Task.Delay(TimeSpan.FromSeconds(3), cancellationToken);
                }
            }
        }
    }
}
