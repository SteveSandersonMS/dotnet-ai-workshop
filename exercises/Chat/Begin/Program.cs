using Azure.AI.OpenAI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.ClientModel;

// Set up DI etc
var hostBuilder = Host.CreateApplicationBuilder(args);
hostBuilder.Configuration.AddUserSecrets<Program>();
hostBuilder.Services.AddLogging(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Trace));

// Register an IChatClient
// For GitHub Models or Azure OpenAI:
var aiConfig = hostBuilder.Configuration.GetRequiredSection("AI");
var innerChatClient = new AzureOpenAIClient(new Uri(aiConfig["Endpoint"]!), new ApiKeyCredential(aiConfig["Key"]!))
    .AsChatClient("gpt-4o-mini");

// Or for OpenAI Platform:
// var aiConfig = hostBuilder.Configuration.GetRequiredSection("AI");
// var innerChatClient = new OpenAI.Chat.ChatClient("gpt-4o-mini", aiConfig["Key"]!).AsChatClient();

// Or for Ollama:
// var innerChatClient = new OllamaChatClient(new Uri("http://localhost:11434"), "llama3.1");

hostBuilder.Services.AddChatClient(innerChatClient);

// Run the app
var app = hostBuilder.Build();
var chatClient = app.Services.GetRequiredService<IChatClient>();

// TODO: Add your code here
