using Azure.AI.OpenAI;
using Chat;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.ClientModel;

// Set up DI etc
var hostBuilder = Host.CreateApplicationBuilder(args);
hostBuilder.Configuration.AddUserSecrets<Program>();
hostBuilder.Services.AddLogging(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Information));

// Register an IChatClient

// For GitHub Models or Azure OpenAI:
var innerChatClient = new AzureOpenAIClient(
    new Uri(hostBuilder.Configuration["AI:Endpoint"]!),
    new ApiKeyCredential(hostBuilder.Configuration["AI:Key"]!))
    .GetChatClient("gpt-4o-mini").AsIChatClient();

// Or for OpenAI Platform:
// var aiConfig = hostBuilder.Configuration.GetRequiredSection("AI");
// var innerChatClient = new OpenAI.Chat.ChatClient("gpt-4o-mini", aiConfig["Key"]!).AsIChatClient();

// Or for Ollama:
// var innerChatClient = new OllamaChatClient(new Uri("http://localhost:11434"), "llama3.1");

hostBuilder.Services.AddChatClient(innerChatClient)
    .UseFunctionInvocation();
    // .UsePromptBasedFunctionCalling(); // Needed to make functions work at all on many Ollama models, and often improves those that support it anyway

// Run the app
var app = hostBuilder.Build();
var chatClient = app.Services.GetRequiredService<IChatClient>();

// Uncomment one of these to decide which to run
await BasicCompletion.RunAsync(chatClient);
//await StructuredOutput.RunAsync(chatClient);
//await ChatBot.RunAsync(chatClient);
