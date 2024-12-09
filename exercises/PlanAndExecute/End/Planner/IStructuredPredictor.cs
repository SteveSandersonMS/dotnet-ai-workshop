using Microsoft.Extensions.AI;

namespace Planner;

public interface IStructuredPredictor
{
    Task<StructuredPrediction> PredictAsync(IEnumerable<ChatMessage> messages, ChatOptions options, CancellationToken cancellationToken);
}
