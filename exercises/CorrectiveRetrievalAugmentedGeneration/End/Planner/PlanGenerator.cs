using Microsoft.Extensions.AI;
using StructuredPrediction;

namespace Planner;

public class PlanGenerator(IChatClient chatClient)
{
    private readonly IStructuredPredictor _structuredPredictor = chatClient.ToStructuredPredictor(typeof(Plan));

    public async Task<Plan> GeneratePlanAsync(string task, CancellationToken cancellationToken = default)
    {
        ChatMessage[] messages = [
            new(ChatRole.System,
                """
                For the given objective, come up with a simple step by step plan.
                This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
                The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
                """),
            new(ChatRole.User, task)];

        StructuredPredictionResult result = await _structuredPredictor.PredictAsync(messages, cancellationToken);

        if (result.Value is not Plan plan)
        {
            throw new InvalidOperationException("No plan generated");
        }

        return plan;
    }
}
