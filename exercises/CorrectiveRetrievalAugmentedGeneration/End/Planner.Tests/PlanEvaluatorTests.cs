using System.ClientModel;
using Azure.AI.OpenAI;
using FluentAssertions;
using FluentAssertions.Execution;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace Planner.Tests;

public class PlanEvaluatorTests
{
    private readonly IConfigurationRoot _configuration;

    public PlanEvaluatorTests()
    {
        IConfigurationBuilder builder = new ConfigurationBuilder()
            .AddUserSecrets<PlanEvaluatorTests>();
        _configuration = builder.Build();
    }

    [Fact]
    public async Task generates_result_if_all_steps_are_performed()
    {
        string endpoint = _configuration["AI:Endpoint"] ?? string.Empty;
        string key = _configuration["AI:Key"] ?? string.Empty;
        IChatClient chatClient = new AzureOpenAIClient(
                new Uri(endpoint),
                new ApiKeyCredential(key))
            
            .GetChatClient("gpt-4o-mini").AsIChatClient();
        PlanEvaluator planEvaluator = new (chatClient);

        Plan plan = new() { Steps = ["calculate necessary fuel for spaceship to cover distance between earth and the moon"] };

        string task = "find how much fuel a spaceship needs to reach the moon from earth";

        List<PlanStepExecutionResult> previousSteps = [
            new ("find distance from earth to the moon", "The distance from earth to the moon is 384,400 km"),
            new ("find out ship fuel consumptions", "The spaceship needs 1 gallons of fuel per 1000km"),
            new("calculate necessary fuel for spaceship to cover distance between earth and the moon", "The ship will consume 384.4 gallons to cover the distance between the earth and the moon")
        ];

        PlanOrResult planOrResult = await planEvaluator.EvaluatePlanAsync(task, plan, previousSteps);

        using AssertionScope scope = new();
        planOrResult.Result.Should().NotBeNull();
        planOrResult.Result!.Outcome.Should().NotBeNullOrWhiteSpace();
        planOrResult.Result.Outcome.Should().Contain("384.4");
    }

    [Fact]
    public async Task generates_updated_plan()
    {
        string endpoint = _configuration["AI:Endpoint"] ?? string.Empty;
        string key = _configuration["AI:Key"] ?? string.Empty;
        IChatClient chatClient = new AzureOpenAIClient(
                new Uri(endpoint),
                new ApiKeyCredential(key))

            .GetChatClient("gpt-4o-mini").AsIChatClient();
        PlanEvaluator planEvaluator = new(chatClient);

        Plan plan = new()
        {
            Steps = [
                "find distance from earth to the moon",
                "find out ship fuel consumptions",
                "calculate necessary fuel for the spaceship to cover the distance"
            ]
        };

        string task = "find how much fuel a spaceship needs to reach the moon from earth";

        List<PlanStepExecutionResult> previousSteps = [
            new ("find distance from earth to the moon", "The distance from earth to the moon is 384,400 km")
        ];

        PlanOrResult planOrResult = await planEvaluator.EvaluatePlanAsync(task, plan, previousSteps);

        using AssertionScope scope = new();

        planOrResult.Plan.Should().NotBeNull();
        planOrResult.Plan!.Steps.Should().HaveCountGreaterThan(1);
    }
}
