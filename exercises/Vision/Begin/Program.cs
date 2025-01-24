using Azure.AI.OpenAI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System.ClientModel;

// Set up DI etc
var hostBuilder = Host.CreateApplicationBuilder(args);
hostBuilder.Configuration.AddUserSecrets<Program>();

// Register an IChatClient

// For Azure OpenAI:
var azureOpenAiConfig = hostBuilder.Configuration.GetRequiredSection("AzureOpenAI");
var innerChatClient = new AzureOpenAIClient(new Uri(azureOpenAiConfig["Endpoint"]!), new ApiKeyCredential(azureOpenAiConfig["Key"]!))
    .AsChatClient("gpt-4o-mini");

// Or for OpenAI Platform:
// var openAiConfig = hostBuilder.Configuration.GetRequiredSection("OpenAI");
// var innerChatClient = new OpenAI.Chat.ChatClient("gpt-4o-mini", openAiConfig["Key"]!).AsChatClient();

// Or for Ollama:
// IChatClient innerChatClient = new OllamaChatClient(new Uri("http://localhost:11434"), "llava");

hostBuilder.Services.AddChatClient(innerChatClient);

// Run the app
var app = hostBuilder.Build();
var chatClient = app.Services.GetRequiredService<IChatClient>();
var trafficImages = Directory.GetFiles("../../../traffic-cam", "*.jpg");
var isOllama = chatClient.GetService<OllamaChatClient>() is not null;

// TODO: Add your code here
