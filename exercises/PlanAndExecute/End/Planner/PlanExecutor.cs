using Microsoft.Extensions.AI;

namespace Planner;

public class PlanExecutor(IChatClient chatClient)
{
    public async Task<ExecutionResult> ExecutePlanStep(Plan plan)
    {
        var planString = string.Join("\n", plan.Steps.Select((step,i) => $"{i+1}. {step.Action}"));
        var task = plan.Steps[0];
        var prompt = $"""
                      For the following plan:
                      {planString}

                      You are tasked with executing step 1, {task.Action}.
                      """;
        var response = await chatClient.CompleteAsync([new ChatMessage(ChatRole.User, prompt)]);
        var ouput = response.Message.Text;
        return new ExecutionResult(task.Action, Output:ouput??string.Empty);

    }
}
