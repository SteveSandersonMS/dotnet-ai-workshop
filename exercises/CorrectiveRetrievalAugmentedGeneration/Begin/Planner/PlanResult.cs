using System.ComponentModel;

namespace Planner;

[Description("The result of executing all steps ofa plan")]
public record PlanResult([Description("The outcome of executing the plan steps")] string Outcome);
