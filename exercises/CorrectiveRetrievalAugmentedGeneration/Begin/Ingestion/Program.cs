using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel.Text;
using Qdrant.Client.Grpc;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.DocumentLayoutAnalysis.PageSegmenter;
using UglyToad.PdfPig.DocumentLayoutAnalysis.WordExtractor;

// This assumes you're already running:
//
// - Ollama with "all-minilm" available (e.g. `ollama pull all-minilm` then `ollama serve`)
// - Qdrant in Docker (e.g., `docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z -d qdrant/qdrant`)
//
// Of course, you could also use Aspire to orchestrate this.

IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
    new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), modelId: "all-minilm");

var qdrantClient = new Qdrant.Client.QdrantClient("127.0.0.1");
if (!await qdrantClient.CollectionExistsAsync("manuals"))
{
    await qdrantClient.CreateCollectionAsync("manuals", new VectorParams { Size = 384, Distance = Distance.Cosine });
}

var manualPdfDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../data/product-manuals"));
ulong pointId = 0;
foreach (var filePath in Directory.EnumerateFiles(manualPdfDir, "*.pdf"))
{
    var productId = int.Parse(Path.GetFileNameWithoutExtension(filePath));
    Console.WriteLine($"Ingesting manual for product {productId}...");

    using var pdf = PdfDocument.Open(filePath);
    foreach (var page in pdf.GetPages())
    {
        // [1] Parse (PDF page -> string)
        var pageText = GetPageText(page);

        // [2] Chunk (split into shorter strings on natural boundaries)
        var paragraphs = TextChunker.SplitPlainTextParagraphs([pageText], 200);

        // [3] Embed (map into semantic space)
        var embeddings = await embeddingGenerator.GenerateAsync(paragraphs);
        var paragraphsWithEmbeddings = paragraphs.Zip(embeddings);

        // [4] Save to vector database, also attaching enough info to link back to the original document
        await qdrantClient.UpsertAsync("manuals", paragraphsWithEmbeddings.Select(x => new PointStruct
        {
            Id = ++pointId,
            Vectors = x.Second.Vector.ToArray(),
            Payload =
            {
                ["text"] = x.First,
                ["productId"] = productId,
                ["pageNumber"] = page.Number,
            }
        }).ToList());
    }
}

static string GetPageText(Page pdfPage)
{
    var words = NearestNeighbourWordExtractor.Instance.GetWords(pdfPage.Letters);
    var textBlocks = DocstrumBoundingBoxes.Instance.GetBlocks(words);
    return string.Join("\n\n", textBlocks.Select(t => t.Text.ReplaceLineEndings(" ")));
}
