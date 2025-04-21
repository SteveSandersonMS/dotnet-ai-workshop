using QuizApp.Components;
using Microsoft.Extensions.AI;
using Azure.AI.OpenAI;
using System.ClientModel;

var builder = WebApplication.CreateBuilder(args);

// For GitHub Models or Azure OpenAI:
var innerChatClient = new AzureOpenAIClient(
    new Uri(builder.Configuration["AI:Endpoint"] ?? throw new InvalidOperationException("Missing AI:Endpoint")),
    new ApiKeyCredential(builder.Configuration["AI:Key"] ?? throw new InvalidOperationException("Missing AI:Key")))
    .GetChatClient("gpt-4o-mini").AsIChatClient();

// Or for OpenAI Platform:
// var aiConfig = builder.Configuration.GetRequiredSection("AI");
// var innerChatClient = new OpenAI.Chat.ChatClient("gpt-4o-mini", aiConfig["Key"]!).AsIChatClient();

// Or for Ollama:
// var innerChatClient = new OllamaChatClient(new Uri("http://localhost:11434"), "llama3.1");

// Add services to the container.
builder.Services.AddRazorComponents().AddInteractiveServerComponents();
builder.Services.AddChatClient(innerChatClient);

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();

app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
