using Microsoft.Extensions.AI;

namespace Planner;

internal class StructuredChatClient : IStructuredPredictor
{
    private readonly IChatClient _client;
    private readonly Type[] _oneOf;

    public StructuredChatClient(IChatClient client, Type[] oneOf)
    {
        _client = client;
        _oneOf = oneOf;
        throw new NotImplementedException();
    }

    public Task<StructuredPrediction> PredictAsync(IEnumerable<ChatMessage> messages, ChatOptions options, CancellationToken cancellationToken)
    {
        throw new NotImplementedException();
    }
}
