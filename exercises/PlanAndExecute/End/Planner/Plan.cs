using System.ComponentModel;

namespace Planner;

[Description("The plan to execute")]
public record Plan([Description("The list of steps for the plan")] PlanStep[] Steps);
