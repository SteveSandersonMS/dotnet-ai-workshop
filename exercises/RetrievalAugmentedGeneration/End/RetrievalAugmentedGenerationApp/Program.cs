using System.ClientModel;
using Azure.AI.OpenAI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Qdrant.Client;
using RetrievalAugmentedGenerationApp;

// Set up app host
var builder = Host.CreateApplicationBuilder(args);
builder.Configuration.AddUserSecrets<Program>();
builder.Services.AddLogging(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Warning));

// For GitHub Models or Azure OpenAI:
IChatClient innerChatClient = new AzureOpenAIClient(new Uri(builder.Configuration["AI:Endpoint"]!), new ApiKeyCredential(builder.Configuration["AI:Key"]!))
    .GetChatClient("gpt-4o-mini").AsIChatClient();

// Or for OpenAI Platform:
// var aiConfig = builder.Configuration.GetRequiredSection("AI");
// var innerChatClient = new OpenAI.Chat.ChatClient("gpt-4o-mini", aiConfig["Key"]!).AsIChatClient();

// Or for Ollama:
// IChatClient innerChatClient = new OllamaChatClient(new Uri("http://127.0.0.1:11434"), "llama3.1");

// Register services
builder.Services.AddHostedService<Chatbot>();
builder.Services.AddEmbeddingGenerator(
    new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), modelId: "all-minilm"));
builder.Services.AddSingleton(new QdrantClient("127.0.0.1"));
builder.Services.AddChatClient(innerChatClient)
    .UseFunctionInvocation();

// Go
await builder.Build().RunAsync();
