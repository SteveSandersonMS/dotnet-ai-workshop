using System.ComponentModel;

namespace Planner;

[Description("The result of a plan")]
public record PlanResult([Description("The outcome of the plan")] string Result);
