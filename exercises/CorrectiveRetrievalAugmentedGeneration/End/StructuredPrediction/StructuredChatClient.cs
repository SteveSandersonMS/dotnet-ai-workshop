using Microsoft.Extensions.AI;

namespace StructuredPrediction;

public class StructuredChatClient : IStructuredPredictor
{
    private readonly IChatClient _client;
    private readonly Dictionary<string, AIParserFunction> _nameToParserTool = [];
    private readonly Dictionary<string, Type> _nameToType = [];

    public StructuredChatClient(IChatClient client, Type[] oneOf)
    {
        if (client is FunctionInvokingChatClient)
        {
            throw new ArgumentException("FunctionInvokingChatClient is not supported", nameof(client));
        }

        _client = client;
        GenerateTools(oneOf.Distinct().ToArray());
    }

    private void GenerateTools(Type[] types)
    {
        foreach (Type type in types)
        {
            AIParserFunction aiParserFunction = new(type);
            _nameToParserTool[aiParserFunction.Metadata.Name] = aiParserFunction;
            _nameToType[aiParserFunction.Metadata.Name] = type;
        }
    }

    public IEnumerable<Type> GetSupportedTypes()
    {
        return _nameToType.Values;
    }

    public async Task<StructuredPredictionResult> PredictAsync(IList<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        ChatOptions localOptions = new()
        {
            Tools = _nameToParserTool.Values.Cast<AITool>().ToList(),
            ToolMode = ChatToolMode.RequireAny
        };


        IList<ChatMessage> chatMessages = [
            new (ChatRole.System, "Select only the most appropriate tools, only one tool call is allowed."),
            ..messages
        ];
        ChatCompletion response = await _client.CompleteAsync(
            chatMessages, localOptions, cancellationToken).ConfigureAwait(false);

        FunctionCallContent[] functionCallContents = response.Message.Contents.OfType<FunctionCallContent>().ToArray();
        if (functionCallContents.Length == 0)
        {
            throw new InvalidOperationException("No Parsing action performed");
        }

        if (functionCallContents.Length > 1)
        {
            throw new InvalidOperationException("Only one parsing action is supported");
        }

        FunctionCallContent functionCallContent = functionCallContents[0];

        if (!_nameToParserTool.TryGetValue(functionCallContent.Name, out AIParserFunction? aiParserTool))
        {
            throw new InvalidOperationException($"Unexpected function call: {functionCallContent.Name}");
        }

        object? result = await aiParserTool.InvokeAsync(functionCallContent.Arguments, cancellationToken);
        Type type = _nameToType[aiParserTool.Metadata.Name];

        return new StructuredPredictionResult(type, result);
    }
}
