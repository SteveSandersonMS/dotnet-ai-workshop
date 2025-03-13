using Azure.AI.OpenAI;
using System.ClientModel;
using Realtime.Components;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

var openAiClient = new AzureOpenAIClient(
    new Uri(builder.Configuration["AI:Endpoint"]!),
    new ApiKeyCredential(builder.Configuration["AI:Key"]!));
var realtimeClient = openAiClient.GetRealtimeConversationClient("gpt-4o-realtime-preview");
builder.Services.AddSingleton(realtimeClient);

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
