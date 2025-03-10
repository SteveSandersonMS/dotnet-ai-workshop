using System.ComponentModel;

namespace Planner;

[Description("The plan to execute")]
public class Plan
{
    [Description("The list of steps for the plan")]
    public string[] Steps { get; set; } = default!;
}
