using Microsoft.Extensions.AI;

namespace StructuredPrediction;

public class StructuredChatClient : IStructuredPredictor
{
    private readonly IChatClient _client;
    private readonly Dictionary<string, AIParserFunction> _nameToParserTool = [];
    private readonly Dictionary<string, Type> _nameToType = [];

    public StructuredChatClient(IChatClient client, Type[] oneOf)
    {
        _client = client;
        GenerateTools(oneOf.Distinct().ToArray());
    }

    private void GenerateTools(Type[] types)
    {
        foreach (var type in types)
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

    public async Task<StructuredPredictionResult> PredictAsync(IList<ChatMessage> messages, ChatOptions options, CancellationToken cancellationToken)
    {

        var localOptions = options.Clone();
        localOptions.Tools ??= new List<AITool>();

        if (localOptions.Tools?.Count == 0)
        {
            localOptions.Tools = _nameToParserTool.Values.Cast<AITool>().ToList();
        }
        else
        {

            foreach (var tool in _nameToParserTool.Values.Cast<AITool>().ToList())
            {
                localOptions.Tools!.Add(tool);
            }
        }
        localOptions.ToolMode = ChatToolMode.RequireAny;
        
        var response = await _client.CompleteAsync(messages, localOptions, cancellationToken).ConfigureAwait(false);

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

        if (!_nameToParserTool.TryGetValue(functionCallContent.Name, out var aiParserTool))
        {
            throw new InvalidOperationException($"Unexpected function call: {functionCallContent.Name}");
        }

        var result = await aiParserTool.InvokeAsync(functionCallContent.Arguments, cancellationToken);
        var type = _nameToType[aiParserTool.Metadata.Name];

        return new StructuredPredictionResult(type, result);
    }
}
