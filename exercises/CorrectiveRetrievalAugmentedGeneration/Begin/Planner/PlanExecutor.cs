using Microsoft.Extensions.AI;

namespace Planner;

public class PlanExecutor(IChatClient chatClient)
{
    public async Task<PlanStepExecutionResult> ExecutePlanStep(Plan plan, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        string planString = string.Join("\n", plan.Steps.Select((step,i) => $"{i+1}. {step.Action}"));
        PlanStep task = plan.Steps[0];
        string prompt = $"""
                         For the following plan:
                         {planString}

                         You are tasked with executing step 1, {task.Action}.
                         """;
        ChatCompletion response = await chatClient.CompleteAsync([new ChatMessage(ChatRole.User, prompt)], options, cancellationToken: cancellationToken);
        string? output = response.Message.Text;
        return new PlanStepExecutionResult(task.Action, Output:output??string.Empty);
    }
}
