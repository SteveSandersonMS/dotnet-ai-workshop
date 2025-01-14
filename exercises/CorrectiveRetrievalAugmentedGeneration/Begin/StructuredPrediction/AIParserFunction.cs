using System.Text.Json;
using Microsoft.Extensions.AI;

namespace StructuredPrediction;

public class AIParserFunction : AIFunction
{
    private readonly Type _type;

    private static readonly AIJsonSchemaCreateOptions s_inferenceOptions = new()
    {
        IncludeSchemaKeyword = true,
        DisallowAdditionalProperties = true,
        IncludeTypeInEnumSchemas = true
    };
    private static readonly JsonSerializerOptions s_serializerOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    public AIParserFunction(Type type)
    {
        _type = type;
        JsonElement schemaElement = AIJsonUtilities.CreateJsonSchema(
            type: type,
            serializerOptions: AIJsonUtilities.DefaultOptions,
            inferenceOptions: s_inferenceOptions);

        JsonElement propertiesElement = schemaElement.GetProperty("properties");
        List<AIFunctionParameterMetadata> parameters = [];
        foreach (JsonProperty p in propertiesElement.EnumerateObject())
        {
            AIFunctionParameterMetadata parameterSchema = new(p.Name) { Schema = p.Value, };
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

        Dictionary<string, object> argumentDictionary = new(arguments);
        object? result = JsonSerializer.Deserialize(JsonSerializer.Serialize(argumentDictionary), _type, s_serializerOptions);
        return Task.FromResult(result);
    }
}
