using Microsoft.Extensions.AI;

using StructuredPrediction;

namespace Planner;

public class PlanEvaluator(IChatClient chatClient)
{
    private readonly IStructuredPredictor _structuredPredictor = chatClient.ToStructuredPredictor(typeof(Plan), typeof(PlanResult));

    public async Task<PlanOrResult> EvaluatePlanAsync(string task, Plan currentPlan, List<PlanStepExecutionResult> previousStepExecutionResults,
        CancellationToken cancellationToken = default)
    {
        string plan = string.Join("\n", currentPlan.Steps.Select((step, i) => $"{i + 1}. {step.Action}"));

        string pastSteps = string.Join("\n", previousStepExecutionResults.Select((step, i) => $"{i + 1}.\tAction:{step.StepAction}\n\tResult: {step.Output}"));
        ChatMessage[] messages =
        [
            new(ChatRole.System,
                $"""
                 For the given objective, come up with a simple step by step plan.
                 This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
                 Do not keep any steps that have already been done.
                 The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

                 Your objective was this:
                 <objective>{task}</objective>

                 Your original plan was this:
                 <original_plan>
                 {plan}
                 </original_plan>

                 You have currently done the follow steps:
                 <past_steps>
                 {pastSteps}
                 </past_steps>

                 Update your <original_plan> accordingly. If no more steps are needed to satisfy <objective> with the <past_steps> and you can return to the user, then respond with that as PlanResult. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan. If the past steps contains the final answer, then return that as the result, adding steps like 'report the result' is not appropriate and will fail the plan.
                 """)
        ];

        StructuredPredictionResult result = await _structuredPredictor.PredictAsync(messages, cancellationToken);
        Plan? newPlan = result.Value as Plan;
        PlanResult? planResult = result.Value as PlanResult;

        return new PlanOrResult(Plan: newPlan, Result: planResult);
    }
}
