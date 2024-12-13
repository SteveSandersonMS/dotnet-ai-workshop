using System.ClientModel;
using Microsoft.Extensions.AI;

using Moq;

using Xunit;

using FluentAssertions;

using Microsoft.Extensions.Configuration;

using Azure.AI.OpenAI;
using FluentAssertions.Execution;
using StructuredPrediction;

namespace Planner.Tests;

public class StructuredChatClientTests
{

    private readonly IConfiguration _configuration;

    public StructuredChatClientTests()
    {
        var builder = new ConfigurationBuilder()
            .AddUserSecrets<StructuredChatClientTests>();

        _configuration = builder.Build();
    }

    [Fact]
    public void StructuredChatClient_generates_tools_for_each_type()
    {
        var clientMock = new Mock<IChatClient>();
        var client = new StructuredChatClient(clientMock.Object, [typeof(Plan)]);

        client.GetSupportedTypes().Should().BeEquivalentTo([typeof(Plan)]);
    }

    [Fact]
    public void AIParserTool_generates_a_function_metadata()
    {
        var tool = new AIParserFunction(typeof(Plan));
        tool.Metadata.Parameters.Should().HaveCount(1);
        tool.Metadata.ReturnParameter.Should().NotBeNull();
    }

    [Fact]
    public async Task AIParserTool_parses_a_conversation()
    {

        var endpoint = _configuration["meai:endpoint"] ?? string.Empty;
        var key = _configuration["meai:apikey"] ?? string.Empty;
        var chatClient = new AzureOpenAIClient(
                new Uri(endpoint!),
                new ApiKeyCredential(key!))

            .AsChatClient("gpt-4o-mini");
        var client = new StructuredChatClient(chatClient, [typeof(Plan)]);

        var result = await client.PredictAsync([new ChatMessage(ChatRole.User, "create a plan to go to the moon")], new ChatOptions(), CancellationToken.None);

        using var _ = new AssertionScope();

        result.PredictionType.Should().Be(typeof(Plan));

        var plan = result.Value as Plan;

        plan.Should().NotBeNull();
        plan!.Steps.Should().HaveCountGreaterThan(0);

    }

    [Fact]
    public async Task AIParserTool_chooses_one_type_to_parse_a_conversation()
    {
        var endpoint = _configuration["meai:endpoint"] ?? string.Empty;
        var key = _configuration["meai:apikey"] ?? string.Empty;
        var chatClient = new AzureOpenAIClient(
            new Uri(endpoint),
            new ApiKeyCredential(key))

            .AsChatClient("gpt-4o-mini");
        var client = new StructuredChatClient(chatClient, [typeof(Plan), typeof(PlanResult)]);

        var result = await client.PredictAsync([
            new ChatMessage(ChatRole.System, "Create a plan if the user asks for help on how to achieve a goal, if is clear what to do then just present a result"),
            new ChatMessage(ChatRole.User, "We got on the moon.")], new ChatOptions(), CancellationToken.None);

        using var _ = new AssertionScope();

        result.PredictionType.Should().Be(typeof(PlanResult));

        var planResult = result.Value as PlanResult;

        planResult.Should().NotBeNull();
    }
}



