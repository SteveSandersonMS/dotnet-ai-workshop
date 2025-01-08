using System.ClientModel;

using Azure.AI.OpenAI;
using CorrectiveRetrievalAugmentedGenerationApp;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

using Qdrant.Client;


// Set up app host
HostApplicationBuilder builder = Host.CreateApplicationBuilder(args);
builder.Configuration.AddUserSecrets<Program>();
builder.Services.AddLogging(logBuilder => logBuilder.AddConsole().SetMinimumLevel(LogLevel.Warning));

IChatClient innerChatClient = new AzureOpenAIClient(new Uri(builder.Configuration["AzureOpenAI:Endpoint"]!), new ApiKeyCredential(builder.Configuration["AzureOpenAI:Key"]!))
    .AsChatClient("gpt-4o-mini");
// Or for Ollama:
//IChatClient innerChatClient = new OllamaChatClient(new Uri("http://127.0.0.1:11434"), "llama3.1");

// Register services
builder.Services.AddHostedService<Chatbot>();
builder.Services.AddEmbeddingGenerator<string, Embedding<float>>(pipeline => pipeline
    .Use(new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), modelId: "all-minilm")));
builder.Services.AddSingleton(new QdrantClient("127.0.0.1"));
builder.Services.AddChatClient(pipeline => pipeline
    .Use(innerChatClient));

// Go
await builder.Build().RunAsync();
