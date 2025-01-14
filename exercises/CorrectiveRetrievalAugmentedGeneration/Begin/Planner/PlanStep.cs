using System.ComponentModel;

namespace Planner;

[Description("The single plan step")]
public record PlanStep([Description("The action to perform in this step")] string Action);
