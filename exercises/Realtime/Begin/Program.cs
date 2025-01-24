using Azure.AI.OpenAI;
using System.ClientModel;
using Realtime.Components;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

var openAiClient = new AzureOpenAIClient(
    new Uri(builder.Configuration["AzureOpenAI:Endpoint"]!),
    new ApiKeyCredential(builder.Configuration["AzureOpenAI:Key"]!));

// Or for OpenAI Platform:
// var openAiClient = new OpenAI.OpenAIClient(builder.Configuration["OpenAI:Key"]!);

// TODO: Register RealtimeConversationClient in DI

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
