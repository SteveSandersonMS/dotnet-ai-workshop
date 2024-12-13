using System.Text.Json;
using Microsoft.Extensions.AI;

namespace StructuredPrediction;

public class AIParserFunction : AIFunction
{
    private readonly Type _type;

    private static readonly AIJsonSchemaCreateOptions _inferenceOptions = new()
    {
        IncludeSchemaKeyword = true,
        DisallowAdditionalProperties = true,
        IncludeTypeInEnumSchemas = true
    };
    private static readonly JsonSerializerOptions _serializerOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    public AIParserFunction(Type type)
    {
        _type = type;
        var schemaElement = AIJsonUtilities.CreateJsonSchema(
            type: type,
            serializerOptions: AIJsonUtilities.DefaultOptions,
            inferenceOptions: _inferenceOptions);

        var propertiesElement = schemaElement.GetProperty("properties");
        var parameters = new List<AIFunctionParameterMetadata>();
        foreach (var p in propertiesElement.EnumerateObject())
        {
            var parameterSchema = new AIFunctionParameterMetadata(p.Name) { Schema = p.Value, };
            parameters.Add(parameterSchema);
        }

        Metadata = new AIFunctionMetadata($"{type.Name}_generator")
        {
            Description = $"Generates a {type.Name} object from the chat context",
            Parameters = parameters,
            ReturnParameter = new()
            {
                ParameterType = type,
                Schema = schemaElement,
            },
        };
    }
    public override AIFunctionMetadata Metadata { get; }

    protected override Task<object?> InvokeCoreAsync(IEnumerable<KeyValuePair<string, object?>> arguments, CancellationToken cancellationToken)
    {

        var argumentDictionary = new Dictionary<string, object?>(arguments);
        object? result = JsonSerializer.Deserialize(JsonSerializer.Serialize(argumentDictionary), _type, _serializerOptions);
        return Task.FromResult(result);
    }
}
