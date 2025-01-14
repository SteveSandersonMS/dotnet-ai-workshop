using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using SmartComponents.LocalEmbeddings;

namespace Embeddings;

public class LocalEmbeddingsGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly LocalEmbedder _embedder = new();

    public EmbeddingGeneratorMetadata Metadata { get; } = new("local");

    public void Dispose() => _embedder.Dispose();

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        var embeddings = _embedder.EmbedRange(values);
        var result = new GeneratedEmbeddings<Embedding<float>>(embeddings.Select(e =>
            new Embedding<float>(Normalize(e.Embedding.Values))));
        return Task.FromResult(result);
    }

    public object? GetService(Type serviceType, object? key = null)
        => key is null ? this : null;

    private static ReadOnlyMemory<float> Normalize(ReadOnlyMemory<float> vec)
    {
        var buffer = new float[vec.Length];
        TensorPrimitives.Divide(vec.Span, TensorPrimitives.Norm(vec.Span), buffer);
        return buffer;
    }
}
