# Chatbots and Retrieval-Augmented Generation (RAG)

One of the most well-known UX patterns in intelligent apps is *chatbots*. And if you're making a chatbot, you'll almost certainly want to connect it to your own data. In businesses, this is often called "Enterprise Q&A", i.e., software that can answer questions about the business's own data.

How do you get a chatbot to **generate** answers based on your own data? You **augment** it with a **retrieval** step. This is the **retrieval augmented generation** pattern, or RAG.

There are many moving pieces needed for a successful and reliable RAG system:

 * **Ingestion**: pulling in data - usually unstructured data from sources such as text files, Word/PowerPoint/SharePoint, PDFs, a custom CMS, and so on, and storing its embeddings in an index. Any large text documents typically need to be chunked, i.e., split into smaller pieces so that semantic search can retrieve just the relevant part.
 * **Assistant logic**: implementing the UI and logic around the chatbot itself, so that it retrieves the relevant data and produces well-grounded answers.
 * **Evaluation**: quantitatively measuring how well the whole system works. Are the answers correct? How often does it hallucinate? How fast/expensive is it?

In this exercise you'll build an end-to-end system handling all of this.

## Project setup

*Prerequisites: These instructions assume you've done earlier sessions, in particular session 1, which gives the basic environment setup steps.*

To start:

 * Open the project `exercises/RetrievalAugmentedGeneration/Begin`
 * If you're using VS, ensure that `Ingestion` is marked as the startup project. For non-VS users, `Ingestion` is the project you should be ready to `dotnet run`.
 * Open `Program.cs`. Follow the instructions at the top, which explain how to:
   * Make sure Ollama is running and has the `all-minilm` model available
   * Make sure Qdrant, a vector database, is running in Docker

If you run the project, you should see it claim to ingest many PDFs, but it's lying and doesn't ingest them at all yet.

## The scenario

We'll be using the sample data from [eShopSupport](https://github.com/dotnet/eShopSupport), a much bigger sample app that demonstrates many AI features with .NET, Blazor, and Aspire. You don't need to clone that repo or know anything about it. All the files you need are already here.

Our scenario is a customer-support system in which staff members receive product questions from the public. We'll build a chatbot that staff members can use to find answers to those questions. The source data will be PDFs - the product manuals.

## Ingesting PDFs

*Ingestion* for RAG means populating a vector database or other index with content you can later retrieve. But we don't want to put *entire* product manuals into the vector DB as single units, because:

 * If you compute the sentence-embedding of very long block of text, all the concepts in it will average out, producing an embedding that isn't strongly associated with any specific meaning.
 * If you retrieve a very long block of text, it might be too long to fit into the context window of an LLM, so it might be unusable. And even for LLMs with huge context windows, you're paying for all the tokens you feed in.

The solution is to split longer documents into many smaller chunks. Ideally each chunk will represent a useful, self-contained piece of information on a distinctive subject. Then a semantic search will likely find it, and it will fit easily into an LLM prompt.

So our first task is to load the product manual PDFs, parse out the text, and split it into small chunks.

### Parsing PDFs

In `Ingestion`'s `Program.cs`, find the `TODO` comment near the bottom, and add the following to load each PDF:

```cs
using var pdf = PdfDocument.Open(filePath);
foreach (var page in pdf.GetPages())
{
    // [1] Parse (PDF page -> string)
    var pageText = GetPageText(page);
}
```

Implement `GetPageText` by adding the following at the bottom of the file. Don't worry if you don't recognize these APIs. This uses a library called [PdfPig](https://github.com/UglyToad/PdfPig), which handles realistic documents very well:

```cs
static string GetPageText(Page pdfPage)
{
    var words = NearestNeighbourWordExtractor.Instance.GetWords(pdfPage.Letters);
    var textBlocks = DocstrumBoundingBoxes.Instance.GetBlocks(words);
    return string.Join("\n\n", textBlocks.Select(t => t.Text.ReplaceLineEndings(" ")));
}
```

Now if you set a breakpoint on the closing `}` after `var pageText = ...`, you should be able to hit it and see each page of text in turn. If you want, compare this to the actual PDF files in `data/product-manuals`.

### Chunking

Next let's chunk it, i.e., split it into small pieces. There are many possible strategies, e.g.,

 * Just split every N characters (simple, but loses meaning whenever it slices words in half)
 * Split on paragraph boundaries, aiming for around N characters per chunk
 * Split on sentence boundaries, aiming for around N characters per chunk
 * Accumulate text up to around N characters per chunk, but end the chunk early if it seems to change subject. This is called *[semantic chunking](https://towardsdatascience.com/a-visual-exploration-of-semantic-text-chunking-6bb46f728e30)*.

The goal is for each chunk to have a clear meaning of its own. We'll use a "hierarchical chunking" utility provided by Semantic Kernel. This tries to split on paragraph boundaries, and falls back on sentence-level splitting if necessary, and falls back further to word-level then character-level splitting if necessary.

Add this right after your `var pageText = ...` line:

```cs
// [2] Chunk (split into shorter strings on natural boundaries)
var paragraphs = TextChunker.SplitPlainTextParagraphs([pageText], 200);
```

Again, you can set a breakpoint on the `}` after this if you want to inspect what it produces. You'll have to hit F5 a few times to get past the title and contents pages and onto a page that contains more interesting text.

### Embedding

Now you've got chunks, what's next? It's time to embed them, just like you did in earlier exercises:

```cs
// [3] Embed (map into semantic space)
var embeddings = await embeddingGenerator.GenerateAsync(paragraphs);
var paragraphsWithEmbeddings = paragraphs.Zip(embeddings);
```

### Storing in a vector DB

Finally, store these embeddings in a vector database. This exercise is set up to use [Qdrant](https://qdrant.tech/), mainly because it's so easy to run under Docker without having to register a database or configure anything.

If you're not already running Qdrant, start it in Docker now:

```
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z -d qdrant/qdrant
```

Near the top of `Program.cs`, before the `foreach` loop that processes the PDFs, instantiate a Qdrant client:

```cs
var qdrantClient = new Qdrant.Client.QdrantClient("127.0.0.1");

// Make sure we have a collection called "manuals"
if (!await qdrantClient.CollectionExistsAsync("manuals"))
{
    await qdrantClient.CreateCollectionAsync("manuals", new VectorParams { Size = 384, Distance = Distance.Cosine });
}

ulong pointId = 0; // We'll use this in a moment
```

To fill up that collection, go back down below the `// [3] Embed` code and add:

```cs
// [4] Save to vector database, also attaching enough info to link back to the original document
await qdrantClient.UpsertAsync("manuals", paragraphsWithEmbeddings.Select(x => new PointStruct
{
    Id = ++pointId, // This was defined above
    Vectors = x.Second.Vector.ToArray(),
    Payload =
    {
        ["text"] = x.First,
        ["productId"] = productId,
        ["pageNumber"] = page.Number,
    }
}).ToList());
```

Run it. This might take a minute or so. While it's going, you might like to go into the Qdrant dashboard at http://localhost:6333/dashboard. You should see it has a single collection called `manuals`, and if you go into it and visit the *Info* tab, it should say it has `points_count` of some amount. It doesn't update continuously, but if you refresh every few seconds you'll see the count go up.

When your ingestion process completes, you should have thousands of "points" (i.e., chunks of text from product manuals) in your vector database.

> [!TIP]
> If you want to change how the chunking works, you might need to empty out your Qdrant database before re-running the ingestor. To do that, ensure the Qdrant container has stopped (e.g., in Docker Desktop UI), then run `docker volume rm qdrant_storage`.

What you've done here might seem routine and mechanical, but there are many choices that will impact the behavior and performance of your RAG chatbot:

 * The accuracy of PDF parsing (note: PDFs don't contain text in the form you want, they just contain many individual characters at X/Y coordinates. Deciding what constitutes a continuous string is subjective)
 * Your chunking strategy and - very critically - chunk length
 * Your choice of embedding model
 * Your choice of vector database or other index

## Implementing the RAG chatbot

Switch over to work on the `RetrievalAugmentedGenerationApp` project.

 * For VS users, set `RetrievalAugmentedGenerationApp` as the startup project
 * Everyone else, prepare to `dotnet run` in the `RetrievalAugmentedGenerationApp` directory

In `Program.cs`, you'll see there's quite a lot of setup code. But none of this is a chatbot at all. It's just setting up an `IChatClient`, and `IEmbeddingGenerator`, and a `QdrantClient`.

Find where `IChatClient innerChatClient` is declared and make sure it's using the LLM backend you want to use, likely one of Azure OpenAI, OpenAI Platform, or Ollama.

### Adding a chat loop

If you run the app now, it will ask you what product you're interested in, and then does nothing else. Go to `Chatbot.cs` and find this comment:

```cs
// TODO: Implement the chat loop here
```

Replace it with:

```cs
while (true)
{
    Console.ForegroundColor = ConsoleColor.White;
    Console.Write("\nYou: ");
    var userMessage = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(userMessage))
    {
        continue;
    }

    // TODO: Get and display answer
}
```

Before we actually call an `IChatService`, we're going to separate the "UI" part of this code (i.e., the console input/output) from the assistant logic. It may seem pointless right now, but will make things hugely easier when we get to the "evaluation" part of this session.

Add this *before* the `while(true)`:

```cs
var thread = new ChatbotThread(chatClient, embeddingGenerator, qdrantClient, currentProduct);
```

... and then replace `TODO: Get and display answer` with:

```cs
var answer = await thread.AnswerAsync(userMessage, cancellationToken);
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"Assistant: {answer.Text}\n");
```

### Getting an answer

Inside `ChatbotThread.cs`, you'll see that `AnswerAsync` currently just throws a `NotImplementedException`. But your goal is to return an answer, and more than that, also a citation that justifies your answer.

Let's start by defining instructions and context for the chatbot. Replace the `private List<ChatMessage> _messages = [];` declaration with:

```cs
private List<ChatMessage> _messages =
[
    new ChatMessage(ChatRole.System, $"""
        You are a helpful assistant, here to help customer service staff answer questions they have received from customers.
        The support staff member is currently answering a question about this product:
        ProductId: ${currentProduct.ProductId}
        Brand: ${currentProduct.Brand}
        Model: ${currentProduct.Model}
        """),
];
```

Fill out `AnswerAsync` with this logic:

```cs
_messages.Add(new(ChatRole.User, userMessage));
var response = await chatClient.CompleteAsync(_messages, cancellationToken: cancellationToken);
_messages.Add(response.Message);

return (response.Message.Text!, Citation: null);
```

*Note*: You'll also need to add the `async` keyword to the `AnswerAsync` signature, i.e.:

```cs
public async Task<(string Text, Citation? Citation)> AnswerAsync ... etc
```

If you run the app now, it will sort of work vaguely. For example, if you pick product 1, you could have a conversation like this:

```
You: Who makes this product?
Assistant: The product is made by Escapesafe.

You: Does it work underwater?
Assistant: The Adventure GPS Tracker is designed to be waterproof to a certain extent.
```

Which part of this conversation is true, and which is a hallucination? Well, if you check the `ChatRole.System` message, it *does* know the brand, and so it does produce true information about that. But it has no information about the suitability for underwater use, so whatever claim it makes about that is a hallucination. It might be right just by chance, or it might be wrong, but either way it's hallucinating (or you could say, bluffing). Let's fix that.

## The "Retrieval" part of RAG

You need to add further information to the prompt (or "chat history") that it can base a real answer on. There are two main ways to do this:

1. **Predetermined context**

   This is the simpler option. On every loop, simply do a semantic search for whatever text the user just typed (filtered to results for the current product ID), and inject the results into context. The LLM gets no control over what semantic search is performed, but as long as it gets the necessary information, the LLM can provide a grounded answer.

2. **Dynamic context**

   The more complex option. Don't hardcode the rule that a semantic search must happen, but instead, register a function (a.k.a. "tool") that the `IChatClient` can call if the LLM wants to perform a search. The LLM can then choose parameters like the search term, and could even specify a different product ID or perform multiple searches.

We'll go with the simpler one for now. The results will be almost as good, it's way more reliable on smaller models in Ollama, and is easier to evaluate and debug. It's a reasonable choice in real apps too, often being twice as fast (one LLM call per user input, instead of 2+).

Clear out the `AnswerAsync` method body and replace it with:

```cs
// For a simple version of RAG, we'll embed the user's message directly and
// add the closest few manual chunks to context.
var userMessageEmbedding = await embeddingGenerator.GenerateEmbeddingVectorAsync(userMessage);
var closestChunks = await qdrantClient.SearchAsync(
    collectionName: "manuals",
    vector: userMessageEmbedding.ToArray(),
    filter: Qdrant.Client.Grpc.Conditions.Match("productId", currentProduct.ProductId),
    limit: 5);
```

### The "Augmented Generation" part of RAG

Next let's call the `IChatClient` with this information in the context. We'll use structured output, so first define an output type. Put this line at the end of the `ChatbotThread` class, just above or below `Citation`:

```cs
private record ChatBotAnswer(int? ManualExtractId, string? ManualQuote, string AnswerText);
```

... then back at the end of `AnswerAsync`, call the LLM to get a `ChatBotAnswer`:

```cs
// Now ask the chatbot
_messages.Add(new(ChatRole.User, $$"""
    Give an answer using ONLY information from the following product manual extracts.
    If the product manual doesn't contain the information, you should say so. Do not make up information beyond what is given.
    Whenever relevant, specify manualExtractId to cite the manual extract that your answer is based on.

    {{string.Join(Environment.NewLine, closestChunks.Select(c => $"<manual_extract id='{c.Id}'>{c.Payload["text"].StringValue}</manual_extract>"))}}

    User question: {{userMessage}}
    Respond as a JSON object in this format: {
        "ManualExtractId": numberOrNull,
        "ManualQuote": stringOrNull, // The relevant verbatim quote from the manual extract, up to 10 words
        "AnswerText": string
    }
    """));

var isOllama = chatClient.GetService<OllamaChatClient>() is not null;
var response = await chatClient.CompleteAsync<ChatBotAnswer>(_messages, cancellationToken: cancellationToken, useNativeJsonSchema: isOllama);
_messages.Add(response.Message);

return response.TryGetResult(out var answer)
    ? (answer.AnswerText, Citation: null)
    : ("Sorry, there was a problem.", default);
```

As you can see, it's aggressively prompting the LLM to use *only* the information from `closestChunks`, and not make up any other claims. It's also asking for a citation, though we're not using that yet.

Try running it now. If you want, set a breakpoint on the line that calls `chatClient.CompleteAsync` and inspect the chat history being passed in. Hopefully you'll see context relevant to your query.

For example, if you pick product 130 (the Bluetooth speaker), you could have this conversation:

```
> 130
Assistant: Hi! You're looking at the WildBeat Bluetooth Speaker. What do you want to know about it?

You: What's the bluetooth range?
Assistant: The Bluetooth range is up to 50 feet.

You: What's that in meters?
Assistant: The Bluetooth range is up to 50 feet, which is approximately 15.24 meters.
```

### Getting citations

How do you know if it's correct? Well, you already asked it to return a citation of whichever chunk it considered most relevant.

Update the `return` line so that instead of setting `Citation: null`, it calls some new method `GetCitation`:

```cs
return response.TryGetResult(out var answer)
    ? (answer.AnswerText, Citation: GetCitation(answer, closestChunks))
    : ("Sorry, there was a problem.", default);
```

... and define it:

```cs
private static Citation? GetCitation(ChatBotAnswer answer, IReadOnlyList<ScoredPoint> chunks)
{
    return answer.ManualExtractId is int id && chunks.FirstOrDefault(c => c.Id.Num == (ulong)id) is { } chunk
        ? new Citation((int)chunk.Payload["productId"].IntegerValue, (int)chunk.Payload["pageNumber"].IntegerValue, answer.ManualQuote ?? "")
        : null;
}
```

Finally, to display this in the console UI, go back to `Chatbot.cs` and underneath this line:

```cs
Console.WriteLine($"Assistant: {answer.Text}\n");
```

... add this:

```cs
// Show citation if given
if (answer.Citation is { } citation)
{
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"CITATION: {citation.ProductId}.pdf page {citation.PageNumber}: {citation.Quote}");
}
```

Now if you pick product 1, you could have this conversation with real citations:

```
> 1
Assistant: Hi! You're looking at the Adventure GPS Tracker. What do you want to know about it?

You: How do I check battery level?

Assistant: To check the battery status, press the Battery Check button on the device and hold it for 3 seconds.
CITATION: 1.pdf page 9: press the Battery Check button

You: How do I know if the level is low?

Assistant: The LED indicator will show red if the battery level is low.
CITATION: 1.pdf page 9: Red: Low battery level
```

Check out the file `data/product-manuals/1.pdf`. On page 9, do the cited phrases appear? You could build a UI that displays the PDF with the cited phrases highlighted (as in [eShopSupport](https://github.com/dotnet/eShopSupport)).

Or ask it about bluetooth range for product 130 again, or just generally find out about other products.

## Evaluation

Now let's think about what it means to be a competent professional. Should you just check if it seems to kinda work in your cherry-picked example one time, and then deploy to prod? Hopefully we can do a bit better! Evaluation lets you *measure systematically* how well your AI feature works.

### What kind of evaluation?

When people talk about evaluation for AI apps, they usually mean one of two things:

* **Offline**, or **development-time** evaluation.
  * This uses a test dataset to score the behavior of the system against a baseline of desired behavior.
  * It runs on the developer's workstation or possibly in CI
  * It's a way of tracking the impact of each potential change you make, including changes to the ingestion process, prompts, assistant logic, the backend AI model, or anything else. If the score goes up, the change is good. If the score goes down, don't deploy it!
* **Online**, or **runtime (in production)** evaluation. This tracks what real users are actually doing/did and somehow quantifies how well your AI handled it. You can use this to choose how to roll out AI features or to detect particularly bad incidents that warrant debugging and blocking in future.
  * In some cases, this is just the AI equivalent of traditional A/B testing. Does a cohort who gets the AI feature reach desired goals (e.g., conversion) at a higher or lower rate than a control group that does not?
  * Or it can be a way of stopping unwanted behavior in realtime, such as by scoring the harmfulness of chatbot output and terminating the conversation instantly if it's getting off track.
  * Or it might simply be a way to let humans leave a rating about their experience with the AI feature.

In this exercise, you'll set up some *offline* (or *development time*) evaluation so that you can reason objectively about the effects of different design choices on your RAG system.

## A simple evalation pipeline

Switch over to work on the `Evaluation` project.

 * For VS users, set `Evaluation` as the startup project
 * Everyone else, prepare to `dotnet run` in the `Evaluation` directory

In `Evaluation`'s `Program.cs`, find where `IChatClient innerChatClient` is declared and make sure it's using the LLM backend you want to use, likely Azure OpenAI, OpenAI Platform, or Ollama.

You'll see that `Program.cs` sets up all the dependencies needed to run your RAG logic - an `IChatClient`, an `IEmbeddingGenerator`, and a `QdrantClient`. It also loads the contents of a test dataset called `evalquestions.json`. Take a look in that file and you'll see it's a list of 500 sample question/answer pairs. These were all generated using AI in the same way that the manual PDFs were.

We regard the `Answer` values in these question/answer pairs to be the ground truth. If you're wondering what makes them true, it's that they represent the underlying source data that was used to produce the manual PDFs (we didn't get those answers out of the PDFs; we got the PDFs out of the answers). So we'll *define* those facts to be true, or at least, they are the target we're evaluating against.

To get started, find the `TODO` comment at the bottom and begin looping over all the eval questions:

```cs
var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = 2 };
var outputLock = new object();

await Parallel.ForEachAsync(evalQuestions, parallelOptions, async (evalQuestion, cancellationToken) =>
{
    Console.ForegroundColor = ConsoleColor.White;
    Console.WriteLine($"Asking question {evalQuestion.QuestionId}...");
    var thread = new ChatbotThread(chatClient, embeddingGenerator, qdrantClient, products[evalQuestion.ProductId]);
    var answer = await thread.AnswerAsync(evalQuestion.Question, cancellationToken);

    lock(outputLock)
    {
        Console.WriteLine($"Got answer: {answer.Text}");
    }
});
```

Run it. You should see it start producing answers rapidly, in an undefined order because of the parallelism.

### Scoring the answers

The normal way to score RAG output is to ask another LLM to score its quality. At this point people normally object:

> If we can't trust the first LLM, why can we trust the second one? Who polices the police???

But it's not as tricky as it seems! When evaluating,

 * ... we are **not** asking the second LLM to understand the question and decide if the first LLM gave an ideal answer
 * ... we are **only** asking the second LLM to look at two statements (which happen to be "actual answer" and "desired answer") and decide if the two statements are factually equivalent (within the context of "question").

So, the evaluator doesn't decide if the answer is right or not. It just decides if the "actual" and "desired" answers represent the same factual claims. This is a basic language comprehension task that LLMs are really good at.

For now, we're just going to do a basic, one-dimensional evaluation in which the entire quality rating is reduced to a single number. We can go a bit further later.

After the `var answer = await thread.AnswerAsync(...)` call but before the `lock`, make this call to get a score:

```cs
// Assess the quality of the answer
var response = await evaluationChatClient.CompleteAsync<ScoreResponse>($$"""
    There is an AI assistant that helps customer support staff to answer questions about products.
    You are evaluating the quality of the answer given by the AI assistant for the following question.

    <question>{{evalQuestion.Question}}</question>
    <truth>{{evalQuestion.Answer}}</truth>
    <answer_given>{{answer.Text}}</answer_given>

    Score how well answer_given represents the truth. You must first give a short justification for your score.
    The score must be one of the following labels
     * Awful: The answer contains no relevant information, or information that contradicts the truth
     * Poor: The answer fails fails to include key information from the truth
     * Good: The answer includes the main points from the truth, but misses some facts
     * Perfect: The answer gives all relevant facts from the truth, without anything that contradicts it

    Respond as JSON object of the form {
      "Justification": string, // Up to 10 words
      "ScoreLabel": string // One of "Awful", "Poor", "Good", "Perfect"
    }
    """, useNativeJsonSchema: isOllama);
```

Note that asking for the score as a label, rather than as a word, tends to produce more consistent results. Language models understand language better than numbers.

You'll also need to define the `ScoreResponse` and `ScoreLabel` types at the bottom of the file:

```cs
record ScoreResponse(string? Justification, ScoreLabel ScoreLabel)
{
    public double ScoreNumber => ScoreLabel switch
    {
        ScoreLabel.Awful => 0,
        ScoreLabel.Poor => 0.3,
        ScoreLabel.Good => 0.7,
        ScoreLabel.Perfect => 1,
        _ => throw new InvalidOperationException("Invalid score label")
    };
}

enum ScoreLabel { Awful, Poor, Good, Perfect }
```

Now you can start keeping track of the scores. Before the `Parallel.ForEach` call, define two variables:

```cs
var runningAverageCount = 0;
var runningAverageTotal = 0.0;
```

Then replace the `lock` statement inside it with:

```cs
if (response.TryGetResult(out var score))
{
    lock (outputLock)
    {
        runningAverageCount++;
        runningAverageTotal += score.ScoreNumber;

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"Question {evalQuestion.QuestionId} scored {score.ScoreNumber} ({score.Justification})");

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"Average: {(runningAverageTotal / runningAverageCount):F2} after {runningAverageCount} questions");
    }
}
```

OK, you're ready to go! When you run it, you'll get a running average score that may be erratic at first but will soon stabilize.

You don't have to wait for it to process all 500 questions. After 50 or so, you'll see where the average is settling and can stop it.

### Let's get empirical

If you're using `gpt-4o-mini`, the score will likely average out at about 0.8. The absolute value of the number is meaningless, but what does matter is whether a change makes it go up or down.

We'll now try changing various parts of the system and determine what makes it better or worse. But first, let's try to decouple the "RAG LLM" from the "evaluation LLM". If you want to compare different language models, you only want to change the "RAG LLM" (the one used by the chatbot), while leaving the "evaluation LLM" (the one scoring answers) unchanged. That way there's a consistent judge applying the same interpretation of `how well answer_given represents the truth`.

Find where `evaluationChatClient` is defined and update it *not* to use `innerChatClient`. Instead, make it have its own independent underlying provider, e.g., for Azure OpenAI:

```cs
var evaluationChatClient = new ChatClientBuilder(
        new AzureOpenAIClient(
            new Uri(config["AzureOpenAI:Endpoint"]!),
            new ApiKeyCredential(config["AzureOpenAI:Key"]!)).AsChatClient("gpt-4o-mini"))
    .UseRetryOnRateLimit()
    .Build();
```

... or for OpenAI Platform:

```cs
var evaluationChatClient = new ChatClientBuilder(
        new OpenAI.Chat.ChatClient(
            "gpt-4o-mini",
            config["OpenAI:Key"]!).AsChatClient())
    .UseRetryOnRateLimit()
    .Build();
```

... or for Ollama:

```cs
var evaluationChatClient = new ChatClientBuilder(
        new OllamaChatClient(new Uri("http://127.0.0.1:11434"), "llama3.1"))
    .UseRetryOnRateLimit()
    .Build();
```

Then, even if you change `innerChatClient`, you can leave `evaluationChatClient` unchanged.

> [!TIP]
> If at all possible, use OpenAI or Azure OpenAI for `evaluationChatClient`, regardless of what you're using in `innerChatClient` for the chatbot. The whole thing will go so much faster.

### Can we make the prompt better or worse?

We'll start with an extreme example just to prove that evaluation measures a real signal and isn't just noise.

Back in `ChatbotThread.cs`, update the prompt inside `AnswerAsync`. Try deleting this line:

```cs
{{string.Join(Environment.NewLine, closestChunks.Select(...
```

... so that the chatbot no longer has any information from the manual. Run the evaluation project again. Does that change make the quality go down? Of course it does, massively!

**Be sure to put the line back before continuing.**

Can you find any changes to prompt phrasing that make it perform better? Or are there parts of the prompt that can be removed with no loss of quality? Don't spend too long on this - try the rest of this exercise first.

### How much context is best?

Go back to `ChatbotThread.cs` and find where `closestChunks` is computed. Currently it uses `limit: 5` (i.e., it includes the closest 5 chunks in context).

Could we get away with reducing this to 1? If the quality is still as good, that would make it cheaper, because there'd be fewer tokens going in.

Does the quality go up if you increase it to 10? What about 50? What's optimal?

### Comparing language models

You might want to compare the performance of different LLM backends, since they have very different cost characteristics. Try setting `innerChatClient` to use different models, both across OpenAI and Ollama. How do they compare? Here are some numbers I got (all evaluated by `gpt-4o-mini`):

| Model | Score |
|---|---|
| `gpt-4o-mini` | 0.82 |
| `llama3.1` | 0.76 |

Others you might want to try:

 * GPT 3.5 Turbo (be sure to deploy a version that supports JSON output)
 * `phi3:mini` and/or `phi3:medium`

Of course, these are not universal measures of the quality of the LLM overall. They are as much a measure of the specific prompts used within the context of the LLM.

### Advanced challenges

* As well as showing the answer quality score, also show the average time taken to answer. Maybe some models are way faster even if their accuracy is slightly lower. You want to be able to make tradeoffs between accuracy, speed, and maybe even cost.
* If you're really determined, you can go right back to the "ingestion" phase and try out other chunking strategies. Is it better if the chunks are much longer or shorter than the 200 chars we're currently using?
* Remember when we talked about "predetermined context" vs "dynamic context" earlier in this doc? Can you implement "dynamic context"? This means *not* performing the semantic search automatically for each user input, but instead, registering a `Tools` entry in `ChatOptions` that peforms the search, letting the LLM decide on the search phrase. Does that perform better or worse?
  * Tip: while this works out fairly straightforwardly on larger GPT models, it's actually really hard to make 7B/8B models on Ollama do it reliably. They tend not to call the search tool at all unless you know how to phrase the prompt just right. You will likely have to *not* use structured (JSON) output if you want the smaller Ollama models to use your tools properly.

## More advanced evaluation

So far, we've reduced everything to a single score. But there are more advanced techniques that try to tease apart the differing underlying reasons for how it behaves, producing multiple scores on different axes.

A well-known one is [RAG Triad](https://truera.com/ai-quality-education/generative-ai-rags/what-is-the-rag-triad/). This specifically tries to evaluate how relevant is the supplied context (e.g., semantic search output), as well as how logically grounded the answer is in that context, as well as the final answer quality.

Philosophically, this is along the same lines as the concept of [justified true belief](https://en.wikipedia.org/wiki/Definitions_of_knowledge#Justified_true_belief), a definition of "knowledge" that has been around since at least Socrates. We don't just want the assistant to make true statements; we also want its statements to have a valid justification. We want its claims to be *knowledge*.

We can measure something like this. It's not precisely conforming to the original definition of RAG triad - and we'll discuss that later - but is quite possibly more useful, since our eval dataset has the advantage of a known ground truth. Replace the one score:

```cs
var runningAverageTotal = 0.0;
```

... with three scores:

```cs
var runningAverageContextRelevance = 0.0; // If low, the context isn't helping (need to improve the "retrieval" phase)
var runningAverageAnswerGroundedness = 0.0; // If low, we're probably hallucinating (even if the answer is true)
var runningAverageAnswerCorrectness = 0.0; // If low, then it's wrong (even if grounded in some context)
```

Now update the prompt to ask for three scores:

```cs
// Assess the quality of the answer
// Note that ideally, "relevance" should be based on *all* the context we supply to the LLM, not just the citation it selects
var response = await evaluationChatClient.CompleteAsync<EvaluationResponse>($$"""
    There is an AI assistant that helps customer support staff to answer questions about products.
    You are evaluating the quality of the answer given by the AI assistant for the following question.

    <question>{{evalQuestion.Question}}</question>
    <truth>{{evalQuestion.Answer}}</truth>
    <context>{{answer.Citation?.Quote}}</context>
    <answer_given>{{answer.Text}}</answer_given>

    You are to provide three scores:

    1. Score the relevance of <context> to <question>.
       Ignore <truth> when scoring this. Does <context> contain information that may answer <question>?
    2. Score the groundedness of <answer_given> in <context>
       Ignore <truth> when scoring this. Does <answer_given> take its main claim from <context> alone?
    2. Score the correctness of <answer_given> based on <truth>.
       Does <answer_given> contain the facts from <truth>?

    Each score comes with a short justification, and must be one of the following labels:
     * Awful: it's completely unrelated to the target or contradicts it
     * Poor: it misses essential information from the target
     * Good: it includes the main information from the target, but misses smaller details
     * Perfect: it includes all important information from the target and does not contradict it

    Respond as JSON object of the form {
      "ContextRelevance": { "Justification": string, "ScoreLabel": string },
      "AnswerGroundedness": { "Justification": string, "ScoreLabel": string },
      "AnswerCorrectness": { "Justification": string, "ScoreLabel": string },
    }
    """, useNativeJsonSchema: isOllama);
```

Also define `EvaluationResponse`:

```cs
class EvaluationResponse
{
    public ScoreResponse? ContextRelevance { get; set; }
    public ScoreResponse? AnswerGroundedness { get; set; }
    public ScoreResponse? AnswerCorrectness { get; set; }

    public bool Populated => ContextRelevance is not null && AnswerGroundedness is not null && AnswerCorrectness is not null;
}
```

Finally, update the code that tracks and display the scores:

```cs
if (response.TryGetResult(out var score) && score.Populated)
{
    lock (outputLock)
    {
        runningAverageCount++;
        runningAverageContextRelevance += score.ContextRelevance!.ScoreNumber;
        runningAverageAnswerGroundedness += score.AnswerGroundedness!.ScoreNumber;
        runningAverageAnswerCorrectness += score.AnswerCorrectness!.ScoreNumber;

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine(JsonSerializer.Serialize(score, new JsonSerializerOptions { WriteIndented = true }));

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"Average: Context relevance {(runningAverageContextRelevance / runningAverageCount):F2}, Groundedness {(runningAverageAnswerGroundedness / runningAverageCount):F2}, Correctness {(runningAverageAnswerCorrectness / runningAverageCount):F2} after {runningAverageCount} questions");
    }
}
```

Now you'll be able to see all three scores as it evaluates.

**Caution**: Here the "groundedness" calculation is based only on the chosen citation. This is a rough approximation. For a better implementation you should include *all* the supplied manual chunks in the `<context></context>`. For example, change `AnswerAsync` to return `Task<(string Text, Citation? Citation, string AllContext)>`.

### The regular version of RAG triad

What we've done above is slightly different to the original and more common definition of RAG triad. In the [original definition](https://www.trulens.org/getting_started/core_concepts/rag_triad/), evaluation is performed without any "ground truth" to compare against.

How is that possible? Well, you're almost doing it already. The code above produces three measures:

1. Context relevance: given **question**, assesses **context**
   - i.e., does **context** contain information relevant to **question**
2. Answer groundedness: given **context**, assesses **answer**
   - i.e., are the factual claims in **answer** based only on the facts given in **context**?
3. Answer truthfulness: given **truth**, assesses **answer**
   - i.e., are the factual claims in **answer** equivalent to the facts given in **truth**?

So, only one of the three even uses "truth"! You could redefine the third measure without "truth" as:

3. Answer helpfulness: given **question**, assess **answer**
   - i.e., does **answer** appear superficially to be helpful for **question** (ignoring whether or not it's factually correct)

Now if all three scores are good, it's probably a good answer, because we know it appears to address the question (score 3), and it's based entirely on some facts (score 2) that are relevant to that question (score 1).

<details>
<summary>SIDENOTE: Why score 3 is even needed</summary>

As for why score 3 is even needed here, consider this example:

* **Question**: Why is the sky blue?
* **Context**: "Chapter 3: Rayleigh Scattering. When sunlight passes through the Earth's atmosphere, it is scattered by air molecules, and blue wavelengths are more scattered than other colors".
* **Answer**: Sunlight passes through the Earth's atmosphere.

Score 1 is good (the context is relevant). Score 2 is good (the answer is based completely on context). But score 3 is bad, because it's not actually saying anything about colors.
</details>

You can use this kind of evaluation in production based on real questions that users actually ask, not just based on a test dataset. You can do so either in realtime, or afterwards as an occasional batch process.

It likely isn't quite as precise as if you do have ground truth to compare against, but in the real world, ground truth is a luxury you often don't have.

**Optional exercise**: Update your evaluation code to work this way. To what extent do you find its scores are still helpful for making decisions, e.g., for trying out different prompts or retrieval logic?
