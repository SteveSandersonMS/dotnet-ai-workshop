using Microsoft.Extensions.AI;

namespace Planner;

public class PlanExecutor(IChatClient chatClient)
{
    public async Task<PlanStepExecutionResult> ExecutePlanStep(Plan plan, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        string planString = string.Join("\n", plan.Steps.Select((step,i) => $"{i+1}. {step}"));
        var task = plan.Steps[0];
        string prompt = $"""
                         For the following plan:
                         {planString}

                         You are tasked with executing step 1, {task}.
                         """;
        ChatResponse response = await chatClient.GetResponseAsync([new ChatMessage(ChatRole.User, prompt)], options, cancellationToken: cancellationToken);
        return new PlanStepExecutionResult(task, Output: response.Text);
    }
}
