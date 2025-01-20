# Agentic RAG and Corrective RAG (CRAG)

Retrieving information to help generative AI is a delicate process. It is easy to get started with searches and vector searches, but it is hard to ensure the right data is loaded. **Agentic RAG** approaches bring in more power to load context to satisfy user requests and application goals.

In its most simple form, an agent performs the following loop:

```mermaid
graph TD;
    A[Perception] --> B[Planning];
    B --> C[Action];
    C --> D[Evaluation]
    D --> A
```

Using generative AI and providing tools to the `Action` step, the agent is able to consume additional logic.

A **corrective retrieval-augmented generation (CRAG)** system reasons about the user request and the current context. It evaluates how supportive and useful the context is. If the context is sufficient to accomplish the task, the response is generated. Otherwise, the system will plan actions to cover the gap in the context.

A classic approach here is to try to use in-house data, often a vector search as in basic RAG. After evaluating the context relevance, the system may choose to supplement the missing parts by searching the web using a search engine like [Bing](https://www.bing.com). Alternatively the system may generate a better query to probe again within the RAG element.

In this exercise you will build on top of the earlier [RAG chatbot and tool calling example](./6_RAGChatbot.md) to implement a CRAG system.

## Project setup

*Prerequisites: These instructions assume you've done earlier sessions, in particular session 1, which gives the basic environment setup steps.*

If you're not already running Qdrant, start it in Docker now:

```
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z -d qdrant/qdrant
```

### Populating Qdrant

If your Qdrant Docker volume (`qdrant_storage`) already contains the sample PDF chunks because you did the [RAG Chatbot](6_RAGChatbot.md) session, you can skip this part.

Otherwise, populate your Qdrant storage as follows:

 * Open the project `exercises/RetrievalAugmentedGeneration/End` (and notice this path contains `End`, not `Begin`)
 * If you're using VS, ensure that `Ingestion` is marked as the startup project. For non-VS users, `Ingestion` is the project you should be ready to `dotnet run`.
 * Open `Program.cs`. Follow the instructions at the top, which explain how to:
   * Make sure Ollama is running and has the `all-minilm` model available
   * Make sure Qdrant, a vector database, is running in Docker

If you run the project, you should see it claim to ingest many PDFs. This will populate a collection in Qdrant. This might take a minute or two.

To learn more about how PDFs are ingested (i.e., how they are parsed, chunked, and embedded), you can refer back to the [RAG Chatbot](6_RAGChatbot.md) session.

### Starting services

Make sure you're running Ollama and that you have the `all-minilm` model available. If you're not sure, run:

```
ollama pull all-minilm
ollama serve
```

If `ollama serve` returns the error *tcp 127.0.0.1:11434: bind: Only one usage of each socket address*, be sure to first close any existing instance of Ollama (e.g., from the system tray on Windows) before running `ollama serve` again.

## Implementing the CRAG chatbot

Switch over to work on the CRAG project:

 * Open the project `exercises/CorrectiveRetrievalAugmentedGeneration/Begin`
 * For VS users, set `CorrectiveRetrievalAugmentedGenerationApp` as the startup project
 * Everyone else, prepare to `dotnet run` in the `CorrectiveRetrievalAugmentedGenerationApp` directory

In `Program.cs`, you'll see there's quite a lot of setup code. But none of this is a chatbot at all. It's just setting up an `IChatClient`, and `IEmbeddingGenerator`, and a `QdrantClient`.

Find where `IChatClient innerChatClient` is declared and make sure it's using the LLM backend you want to use, likely either Azure OpenAI or Ollama.

### Ranking and filtering RAG results 

Inside `ChatbotThread.cs`, you'll see that `AnswerAsync` currently just performs a search operation over the manual chunks and then tries to generate an answer.

The approach here is *predetermined context* as seen in the [RAG Chatbot](6_RAGChatbot.md) sample. The code gets the 3 best matches it can find in the vector store.

We will take this further and use the LLM to evaluate how relevant are the chunks we have retrieved. 
The class `ContextRelevancyEvaluator` can help us ranking and filtering the chunks. 

When performing the semantic search we have found the closest matches using vector search. This is a similarity measure, not a measure of how relevant the chunks are to the user question.
Using an LLM to reason about *context relevancy* means to ask it to score how useful the context is to satisfying a user question.

Let's do it. Inside `ChatbotThread.cs`'s `AnswerAsync` method, replace these lines:

```cs
// For basic RAG, we just add *all* the chunks to context, ignoring relevancy
var chunksForResponseGeneration = closestChunksById.Values.ToDictionary(c => c.Id, c => c);
```

... with this improved alternative:

```cs
// For improved RAG, add only the truly relevant chunks to context
var chunksForResponseGeneration = new Dictionary<ulong, Chunk>();
var contextRelevancyEvaluator = new ContextRelevancyEvaluator(chatClient);
foreach (var retrievedContext in closestChunksById.Values)
{
    var score = await contextRelevancyEvaluator.EvaluateAsync(userMessage, retrievedContext.Text, cancellationToken);
    if (score.ContextRelevance?.ScoreNumber >= 0.7)
    {
        chunksForResponseGeneration.Add(retrievedContext.Id, retrievedContext);
    }
}
```

### How it works

Inside `ContextRelevancyEvaluator.cs` we can see the logic used to ask the LLM to perform ranking:

```cs
public async Task<EvaluationResponse> EvaluateAsync(string question, string context, CancellationToken cancellationToken)
{
    bool isOllama = chatClient.GetService<OllamaChatClient>() is not null;

    // Assess the quality of the answer
    // Note that in reality, "relevance" should be based on *all* the context we supply to the LLM, not just the citation it selects
    var response = await chatClient.CompleteAsync<EvaluationResponse>($$"""
    There is an AI assistant that helps customer support staff to answer questions about products.
    You are evaluating the quality of the answer given by the AI assistant for the following question.

    <question>{{question}}</question>
    <context>{{context}}</context>

    You are to provide two scores:

    1. Score the relevance of <context> to <question>.
       Does <context> contain information that may answer <question>?


    Each score comes with a short justification, and must be one of the following labels:
     * Awful: it's completely unrelated to the target or contradicts it
     * Poor: it misses essential information from the target
     * Good: it includes the main information from the target, but misses smaller details
     * Perfect: it includes all important information from the target and does not contradict it

    Respond as JSON object of the form {
        "ContextRelevance": { "Justification": string, "ScoreLabel": string },
    }
    """, useNativeJsonSchema: isOllama, cancellationToken: cancellationToken);

    if (response.TryGetResult(out var score) && score.Populated)
    {
        return score;
    }

    throw new InvalidOperationException("Invalid response from the AI assistant");
}
```
Returning a structured object instead of a string makes it easier to integrate an LLM in traditional code.

### Correcting the aim

Now that we have discarded irrelevant context we might need additional material. This is the corrective part of the algorthm. There are few approaches possible

1. **Query rewriting**
  
   [Query rewriting](https://medium.com/@florian_algo/advanced-rag-06-exploring-query-rewriting-23997297f2d1) takes the current available context and user question, and asks the LLM to generate new questions that could address the user's original goal. For example, we could generate five more questions (i.e., different phrasings of the user's goal) and use these to load more chunks from the vector store.

2. **HyDE (Hypothetical Document Embeddings)**

   Instead of generating hypothetical user inputs as above, [HyDE](https://medium.com/etoai/advanced-rag-precise-zero-shot-dense-retrieval-with-hyde-0946c54dfdcb) generates hypothetical documents to be indexed in your vector store.

   This is done using LLMs to generate hypothetical answers to queries. These answers are then turned into vector embeddings and placed in the same space as real documents. When a search is performed, the system may find these hypothetical documents, and if it does, the search results are amended to be the corresponding real documents from which the hypotheticals were derived.

*Query rewriting* and *HyDE* are closely related in that they both aim to improve retrieval by allowing for alternate ways to phrase things. A difference between the two is that query rewriting computes those alternatives at runtime during each query, whereas HyDE computes the alternatives just once up front.

### Reasoning in agentic worflow

    Agents implement a loop, when reasoning they use the cotnext and the objective to formualte a plan. In the loop they perform some action which can change the current context.
    Changes to the context could lead to plan changes or the final goal. We will be using a **Plan, Step, Eval** approach.

    First let's break down few thigns we need to consider.
#### Making plans

LLMs can generate plans to accomplish a goal giving us back a list of steps. If we use text in and text out the agentic loop becomes very weak as it will be as strong as the parsing logic will be. To improve our work we will be forcing the LLM itself to reason in terms of structured objects. In the project `StructuredPrediction` You will find some utilities to create a `IStructuredPredictor` from a `IChatClient`, have a look at the tests for the project.
Since we will be using the `Planner` project this is how to use teh structured parser.

In the following snippet we are using it to create a plan (see the implementation of the `PlanGenerator` class)
// create a structures predictro from an instance of IChatClient
```csharp

IStructuredPredictor structuredPredictor = chatClient.ToStructuredPredictor([typeof(Plan)]);
// user it to obtain a plan

StructuredPredictionResult result = await structuredPredictor.PredictAsync([new ChatMessage(ChatRole.User, "create a plan to go to the moon")]);
if (result.Value is not Plan plan)
{
    throw new InvalidOperationException("No plan generated");
}
...
```

We can provide a lsit of types when creating a `IStructuredPredictor`, this will accomplish the same as passing a discriminated union.
The objective is to force the choice of one of the tyep provided.

This makes it easier to write our **plan-execute-eval** loop as a plain and clear csharp algorythm.

In the 