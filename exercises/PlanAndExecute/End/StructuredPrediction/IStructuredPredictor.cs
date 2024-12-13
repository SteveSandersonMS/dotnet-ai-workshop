using Microsoft.Extensions.AI;

namespace StructuredPrediction;

public interface IStructuredPredictor
{
    Task<StructuredPredictionResult> PredictAsync(IList<ChatMessage> messages, ChatOptions options, CancellationToken cancellationToken);
}
