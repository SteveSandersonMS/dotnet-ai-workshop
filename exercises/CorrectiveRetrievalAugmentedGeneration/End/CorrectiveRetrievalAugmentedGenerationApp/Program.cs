using System.ClientModel;

using Azure.AI.OpenAI;
using CorrectiveRetrievalAugmentedGenerationApp;
using CorrectiveRetrievalAugmentedGenerationApp.Search;

using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

using Qdrant.Client;

// Set up app host
var builder = Host.CreateApplicationBuilder(args);
builder.Configuration.AddUserSecrets<Program>();
builder.Services.AddLogging(loggingBuilder => loggingBuilder.AddConsole().SetMinimumLevel(LogLevel.Warning));

IChatClient innerChatClient = new AzureOpenAIClient(new Uri(builder.Configuration["AI:Endpoint"]!), new ApiKeyCredential(builder.Configuration["AI:Key"]!))
    .GetChatClient("gpt-4o-mini").AsIChatClient();


// Register services
builder.Services.AddHostedService<Chatbot>();
builder.Services.AddEmbeddingGenerator(
    new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), modelId: "all-minilm"));
builder.Services.AddSingleton(new QdrantClient("127.0.0.1"));
builder.Services.AddChatClient(innerChatClient);

// bing
builder.Services.AddSingleton<ISearchTool>(b =>
{
    var httpClient = new HttpClient();
    return new BingSearchTool(
        builder.Configuration["BingSearch:Key"]!,
        httpClient);
});

//// DuckDuckGoSearchTool
//builder.Services.AddSingleton<ISearchTool>(b =>
//{
//    var httpClient = new HttpClient();
//    return new DuckDuckGoSearchTool(httpClient);
//});

// Go
await builder.Build().RunAsync();
