using Azure.AI.OpenAI;
using Microsoft.Extensions.AI;
using System.ClientModel;
using Microsoft.Extensions.Configuration;
using Qdrant.Client;
using RetrievalAugmentedGenerationApp;
using System.Text.Json;
using Evaluation;

// ------ GET SERVICES ------

var config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();

// For GitHub Models or Azure OpenAI:
IChatClient innerChatClient = new AzureOpenAIClient(new Uri(config["AI:Endpoint"]!), new ApiKeyCredential(config["AI:Key"]!))
    .GetChatClient("gpt-4o-mini").AsIChatClient();

// Or for OpenAI Platform:
// var aiConfig = config.GetRequiredSection("AI");
// var innerChatClient = new OpenAI.Chat.ChatClient("gpt-4o-mini", aiConfig["Key"]!).AsIChatClient();

// Or for Ollama:
// IChatClient innerChatClient = new OllamaChatClient(new Uri("http://127.0.0.1:11434"), "llama3.1");

var chatClient = new ChatClientBuilder(innerChatClient)
    .UseFunctionInvocation()
    .UseRetryOnRateLimit()
    .Build();

// There's nothing to stop you from using a different LLM for evaluation vs the one that actually powers the chatbot
// In fact, really you *should* use the best LLM you can for scoring, even when testing out a smaller model for the chatbot
// In this case we'll use the same for both, since you might only have access to one of them.
var evaluationChatClient = new ChatClientBuilder(innerChatClient)
    .UseRetryOnRateLimit()
    .Build();

var embeddingGenerator = new OllamaEmbeddingGenerator(new Uri("http://127.0.0.1:11434"), modelId: "all-minilm");

var qdrantClient = new QdrantClient("127.0.0.1");

// ------ LOAD TEST DATA ------

var products = Helpers.GetAllProducts().ToDictionary(p => p.ProductId, p => p);
var evalQuestions = JsonSerializer.Deserialize<EvalQuestion[]>(File.ReadAllText(Path.Combine(Helpers.DataDir, "evalquestions.json")))!;

// ------ RUN EVALUATION LOOP ------

// TODO: Implement evaluation here
