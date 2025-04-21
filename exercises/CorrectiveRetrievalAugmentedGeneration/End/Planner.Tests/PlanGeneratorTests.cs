using System.ClientModel;
using Azure.AI.OpenAI;
using FluentAssertions;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace Planner.Tests;

public class PlanGeneratorTests
{
    private readonly IConfiguration _configuration;

    public PlanGeneratorTests()
    {
        IConfigurationBuilder builder = new ConfigurationBuilder()
            .AddUserSecrets<PlanEvaluatorTests>();
        _configuration = builder.Build();
    }


    [Fact]
    public async Task generates_plan_to_accomplish_task()
    {
        string endpoint = _configuration["AI:Endpoint"] ?? string.Empty;
        string key = _configuration["AI:Key"] ?? string.Empty;
        IChatClient chatClient = new AzureOpenAIClient(
                new Uri(endpoint),
                new ApiKeyCredential(key))

            .GetChatClient("gpt-4o-mini").AsIChatClient();
        PlanGenerator planGenerator = new(chatClient);

        Plan plan = await planGenerator.GeneratePlanAsync("find how much fuel a spaceship needs to reach the moon from earth");

        plan.Steps.Length.Should().BeGreaterThanOrEqualTo(1);
    }
}
