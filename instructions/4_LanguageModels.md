# Language models

In this session we'll go through many of the core features of language models and the APIs we use with them.

## Project setup

*Prerequisites: These instructions assume you've done earlier sessions, in particular session 1, which gives the basic environment setup steps.*

Start by opening the project `exercises/Chat/Begin`. Near the top, find the variable `innerChatClient` and update its value according to the LLM service you wish to use.

For Azure OpenAI, you should have code like this:

```cs
var azureOpenAiConfig = hostBuilder.Configuration.GetRequiredSection("AzureOpenAI");
var innerChatClient = new AzureOpenAIClient(new Uri(azureOpenAiConfig["Endpoint"]!), new ApiKeyCredential(azureOpenAiConfig["Key"]!))
    .AsChatClient("gpt-4o-mini");
```

If you're using a model other than `gpt-4o-mini`, update this code.

For Ollama, you should assign a value like this:

```cs
var innerChatClient = new OllamaChatClient(new Uri("http://localhost:11434"), "llama3.1");
```

**If possible, get access to *both* an OpenAI-based service and Ollama. It's instructive to try them both and compare the performance of different models.**

## Basic completions

The simplest LLM API you'll use is `CompleteAsync`, which has an overload that takes just a plain prompt string and returns the entire completion.

At the bottom of `Program.cs`, make a simple completion call:

```cs
var response = await chatClient.CompleteAsync(
    "Explain how real AI compares to sci-fi AI in max 20 words.");
Console.WriteLine(response.Message.Text);
Console.WriteLine($"Tokens used: in={response.Usage?.InputTokenCount}, out={response.Usage?.OutputTokenCount}");
```

Try it and verify your `IChatClient` works.

### Accessing provider-specific data

Set a breakpoint on `Console.WriteLine(response.Message.Text);` and run again. In the debugger, explore the values returned on `response`. You should be able to find:

 * `AdditionalProperties`, a string-object dictionary in which `IChatClient` implementations can place loosely-typed data that will show up in logging and telemetry.
 * `RawRepresentation`, the actual object from the underlying provider client library. You could use code like the following to access OpenAI-specific data, for example:

```cs
if (response.RawRepresentation is OpenAI.Chat.ChatCompletion openAiCompletion)
{
    Console.WriteLine(openAiCompletion.SystemFingerprint);
}
```

`IChatClient` is designed so that in most cases you can use the same programming model across all providers. But if you need to break out of the abstraction, it allows you to do so.

## Streaming

If the LLM is returning a long response, you might want to start displaying it while it's still being generated. This is particularly relevant in a chat UI.

Try the following code:

```cs
var responseStream = chatClient.CompleteStreamingAsync("Explain how real AI compares to sci-fi AI in max 200 words.");
await foreach (var message in responseStream)
{
    Console.Write(message.Text);
}
```

You should see the response appear incrementally.

## It's really not all about chat

Despite the name, `IChatClient` isn't only for building chat-based UIs! That's actually just one special case within a much broader range of usages. Chatting with humans isn't even particularly important - **LLMs can offer a lot more business value by automating business processes.** This could include:

 * Classification
 * Summarization
 * Data extraction
 * Anomaly detection
 * Translation
 * Sentiment detection

... and presumably much more we haven't yet discovered. "Chat" is just the name for the calling protocol we're using, i.e., a stateful sequence of request-responses from your code to an AI service. This is a very general protocol that can model a wide range of tasks.

## Structured output

So far we've only used `CompleteAsync` to return arbitrary, unstructured text. But when automating business processes, we normally don't want the result to be arbitrary text written in a human language like English. We want structured data: objects, enums, arrays, flags, and so on.

Instead of returning arbitrary strings, let's get `IChatClient` to return .NET objects matching a business domain.

Our example here will be *data extraction*. There's a real estate agent or property website, and it needs to process human-written descriptions of properties, and convert it into structured data so that we can do programmatic things with it (e.g., detect undervalued properties or send notifications to people looking to buy properties matching criteria, etc.).

First delete all the code from `Program.cs` below this line:

```cs
var chatClient = app.Services.GetRequiredService<IChatClient>();
```

... so you're back to a fresh start with an `IChatClient` ready to go.

Paste in this sample data:

```cs
var propertyListings = new[]
{
    "Experience the epitome of elegance in this 4-bedroom, 3-bathroom home located in the upscale neighborhood of Homelands. With its spacious garden, modern kitchen, and proximity to top schools, this property is perfect for families seeking a serene environment. The home features a large living room with a fireplace, a formal dining area, and a master suite with a walk-in closet and en-suite bathroom. The neighborhood is known for its well-maintained parks, excellent public transport links, and a strong sense of community. Minimum offer price: $850,000. Contact Dream Homes Realty at (555) 123-4567 to schedule a viewing or visit our website for more information.",
    "A cozy 2-bedroom apartment for rent in the heart of Starside. This trendy neighborhood is known for its vibrant art scene, eclectic cafes, and lively nightlife. The apartment features an open-plan living area, a modern kitchen with stainless steel appliances, and a balcony with city views. The building offers amenities such as a rooftop terrace, a fitness center, and secure parking. Ideal for young professionals and creatives, this property is within walking distance to public transport and popular local attractions. Rent: $1,200 per month, excluding utilities. Contact Urban Nest Rentals at (555) 987-6543 to arrange a viewing.",
    "Wake up to the sound of waves in this stunning 3-bedroom, 2-bathroom beach house. Located in Deep Blue, this property offers breathtaking ocean views, a private deck, and direct beach access. The house features a spacious living area with floor-to-ceiling windows, a fully equipped kitchen, and a master bedroom with an en-suite bathroom. The neighborhood is known for its pristine beaches, seafood restaurants, and water sports activities. Perfect for those who love the sea, this home is a rare find. Minimum offer price: $1,200,000. Contact Coastal Living Realty at (555) 321-4321 for more details.",
    "For Sale: A spacious 3-bedroom house in Neumann. Despite its troubled reputation, this area is undergoing revitalization. The property features a large backyard, modern amenities, and is close to public transport. The house includes a bright living room, a renovated kitchen, and a master bedroom with ample closet space. The neighborhood offers a mix of cultural attractions, community centers, and new development projects aimed at improving the area. Minimum offer price: $350,000. Contact New Beginnings Realty at (555) 654-3210 to learn more.",
    "Escape the hustle and bustle in this charming 2-bedroom cottage in Maeverton. With its quiet streets, friendly neighbors, and beautiful parks, this home is ideal for retirees or anyone seeking tranquility. The cottage features a cozy living room with a fireplace, a modern kitchen, and a lovely garden perfect for relaxing. The neighborhood is known for its excellent schools, local shops, and community events. This property offers a peaceful lifestyle while still being close to the city. Minimum offer price: $450,000. Contact Serenity Homes at (555) 789-0123 for more information.",
    "Rent: A unique 5-bedroom mansion in the enigmatic neighborhood of Blacknoir. This property boasts gothic architecture, a sprawling garden, and a rich history. The mansion features a grand entrance hall, a library, and a master suite with a private balcony. The neighborhood is known for its mysterious charm, with hidden alleyways, historic buildings, and a vibrant arts scene. Perfect for those who appreciate the unusual, this home offers a one-of-a-kind living experience. Rent: $3,500 per month, excluding utilities. Contact Enigma Estates at (555) 456-7890 to schedule a viewing.",
    "Stylish 1-bedroom condo available in Starside. This property features an open-plan living area, high-end finishes, and is within walking distance to the best bars and restaurants. The condo includes a modern kitchen with granite countertops, a spacious bedroom with a walk-in closet, and a private balcony. The building offers amenities such as a fitness center, a rooftop terrace, and secure parking. Ideal for singles or couples, this condo provides a chic urban lifestyle in one of the city's most vibrant neighborhoods. Minimum offer price: $300,000. Contact Cityscape Realty at (555) 234-5678 for more details.",
    "A beautiful 4-bedroom house with a large garden, modern kitchen, and spacious living areas, is now available for sale. Located in the prestigious Homelands neighborhood, this property is perfect for families looking for a safe and pleasant environment. The home features a formal dining room, a family room with a fireplace, and a master suite with a walk-in closet and en-suite bathroom. The neighborhood is known for its excellent schools, parks, and community events. Minimum offer price: $900,000. Contact Family Nest Realty at (555) 123-4567 to schedule a viewing or visit our website for more information.",
    "Charming 2-bedroom bungalow for rent in Deep Blue. Enjoy the ocean breeze from your private patio, and take advantage of the nearby shops and restaurants. The bungalow features a cozy living room, a fully equipped kitchen, and a master bedroom with an en-suite bathroom. The neighborhood is known for its beautiful beaches, seafood restaurants, and outdoor activities. Ideal for beach lovers, this property offers a relaxed coastal lifestyle. Rent: $1,500 per month, excluding utilities. Contact Seaside Rentals at (555) 321-4321 to arrange a viewing.",
    "For Sale: A 3-bedroom fixer-upper in Neumann. This property has great potential with a little TLC. The house features a spacious living room, a kitchen with ample storage, and a large backyard. The neighborhood is undergoing revitalization, with new development projects and community initiatives aimed at improving the area. Close to schools and public transport, this is a great opportunity for investors or first-time buyers. Minimum offer price: $250,000. Contact Future Homes Realty at (555) 654-3210 for more information.",
};
```

... and define an object model we want to work with:

```cs
class PropertyDetails
{
    public ListingType ListingType { get; set; }
    public required string Neighbourhood { get; set; }
    public int NumBedrooms { get; set; }
    public decimal Price { get; set; }
    public required string[] Amenities { get; set; }
    public required string TenWordSummary { get; set; }
}

[JsonConverter(typeof(JsonStringEnumConverter))]
enum ListingType { Sale, Rental }
```

We'll loop through the source data and use the generic-typed overload `CompleteAsync<T>`.

*Note: You'll need to insert the following code above the `PropertyDetails` class definition. This is a C# compiler limitation related to top-level statements.*

```cs
foreach (var listingText in propertyListings)
{
    var response = await chatClient.CompleteAsync<PropertyDetails>(
        $"Extract information from the following property listing: {listingText}");

    if (response.TryGetResult(out var info))
    {
        Console.WriteLine(JsonSerializer.Serialize(info, 
            new JsonSerializerOptions { WriteIndented = true }));
    }
    else
    {
        Console.WriteLine("Response was not in the expected format.");
    }
}
```

Does it work? In my experience:

 * `gpt-4o-mini` will produce near-perfect results every time without any further prompting
 * Small models on Ollama (e.g., 7 or 8-billion parameter ones like the default `llama3.1`) are much more hit-and-miss. It will probably produce good output on about 50% of the inputs. For others it might miss properties or even return invalid data.

Regardless of whether this works for you, let's understand what's happening behind the scenes. LLMs can't return a `PropertyDetails` directly, so `CompleteAsync<T>` augments your prompt by describing the JSON schema of the data to return. Then, `response.TryGetResult` does JSON-parsing on the result.

To see this for yourself, find the call to `AddChatClient` near the top of `Program.cs`, and insert `UseLogging` as follows:

```cs
hostBuilder.Services.AddChatClient(innerChatClient)
    .UseLogging();
```

Next time you run, it will produce a detailed log of the calls to the underlying LLM provider. It will be very hard to read, but if you look closely, you may notice that your prompt has been augmented with an instruction like the following:

```
Respond with a JSON value conforming to the following schema:
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "ListingType": { "type": "string", "enum": ["Sale", "Rental" ] },
    "Neighbourhood": { "type": "string" },
    "NumBedrooms": { "type": "integer" },
    "Price": { "type": "number" },
    "Amenities": { "type": "array", "items": { "type": "string" } },
    "TenWordSummary": { "type": "string" }
  },
  "required": [ "Neighbourhood", "Amenities", "TenWordSummary" ],
  "additionalProperties": false
}
```

A sufficiently smart model like `gpt-4o-mini` will comfortably understand the JSON schema definition and produce a conformant output.

**Now remove the `UseLogging` call otherwise the console output will be impossible to read.**

### Improving the behavior on small models

Even if you weren't using Ollama before, try to do so now. This means changing your `innerChatClient` to something like:

```cs
IChatClient innerChatClient = new OllamaChatClient(new Uri("http://localhost:11434"), "llama3.1")
```

On `llama3.1`, it will produce good `PropertyDetails` data about 50% of the time. The problem is that smaller models aren't very good at understanding JSON schema - the concept is too formal and abstract. They are much better at understanding an *example* of the JSON you want them to return.

So, replace this code:

```cs
var response = await chatClient.CompleteAsync<PropertyDetails>(
    $"Extract information from the following property listing: {listingText}");
```

... with the following:

```cs
var messages = new List<ChatMessage>
{
    new(ChatRole.System, """
        Extract information from the following property listing.

        Respond in JSON in this format: {
            "TenWordSummary": string,
            "ListingType": string, // "Sale" or "Rental"
            "Neighbourhood": string,
            "NumBedrooms": number,
            "Price": number,
            "Amenities": [string, ...],
        }
        """),
    new(ChatRole.User, listingText),
};
var response = await chatClient.CompleteAsync<PropertyDetails>(messages, useNativeJsonSchema: true);
```

Any better? It will probably work perfectly on `llama3.1` every time now. You can even use a very small model like `phi3:mini` and get great results.

**Warning:** If you're using OpenAI, you can only set `useNativeJsonSchema: true` for models that support [native structured output](https://platform.openai.com/docs/guides/structured-outputs). On Ollama it's just ignored (except that it also causes `CompleteAsync<T>` *not* to attach the JSON schema to the prompt, reducing confusion). This is a temporary inconvenience - we expect a later version of Microsoft.Extensions.AI to toggle `useNativeJsonSchema` automatically based on the model's capabilities.

### Use cases for structured output

Most business-process automation will use structured output. Examples:

 * `"Based on this audio transcription, identify why the customer called and rate their satisfaction with the outcome"` -> returning `{ enum CallReason, int Satisfaction }`, possibly generating statistics and alerts for particularly bad cases
 * `"Check this forum post and determine if it is harmful or offensive (true if offensive, false otherwise)"` -> returning a bool, possibly blocking or flagging the post
 * `"Take this long text and return a suggested title and list of tags"` -> returning `{ string Title, string[] Tags }`, possibly prepopulating some UI elements

## A chat loop

Chat-with-humans is a classic use case for LLMs, so let's implement it.

Delete whatever code you have below this line:

```cs
var chatClient = app.Services.GetRequiredService<IChatClient>();
```

... to reset back to a simple state. And if you temporarily switched from OpenAI to Ollama for the preceding exercise, feel free to switch back to OpenAI now.

Start with this simple chat loop:

```cs
List<ChatMessage> messages = [new(ChatRole.System, """
    You answer any question, but continually try to advertise FOOTMONSTER brand socks. They're on sale!
    """)];

while (true)
{
    // Get input
    Console.ForegroundColor = ConsoleColor.White;
    Console.Write("\nYou: ");
    var input = Console.ReadLine()!;
    messages.Add(new(ChatRole.User, input));

    // Get reply
    var response = await chatClient.CompleteAsync(messages);
    messages.Add(response.Message);
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"Bot: {response.Message.Text}");
}
```

You can now have a stateful but stupid conversation with the bot:

```
You: What's 1+1?

Bot: 1 + 1 equals 2! Speaking of pairs, have you checked out the amazing FOOTMONSTER brand socks? They're currently on sale! Perfect for keeping your feet cozy while you crunch numbers!

You: What's the result if we add another 1?

Bot: If you add another 1 to 2, you get 3. Just like how you can always have more comfort with an extra pair of FOOTMONSTER brand socks! Grab yours while they're on sale - your feet will thank you!
```

Remember that sending `CompleteAsync` calls to the LLM doesn't cause it to learn or update its weights in any way. It's stateless. The only thing that makes the above conversation stateful is that you're adding all the messages to a `List<ChatMessage>`, and resending the entire chat history on every call.

**Optional exercise:** If you want, try changing from `CompleteAsync` to `CompleteStreamingAsync`, so that it displays the bot's replies in realtime while they are being generated. You'll also need to accumulate all the chunks in a `StringBuilder` so you can add a corresponding message to `messages` when it's finished replying. *Note: if you're using Ollama, you'll need to change it back to non-streaming before the next exercise, because Ollama doesn't yet support function calling and streaming at the same time.*

## Function calling (a.k.a. tool calling)

*Note for Ollama users: you need to use a tool-calling-enabled model, such as `llama3.1` or `qwen2.5`, or you'll get an error.*

OK, we're now getting to the bit where the LLM or chat system can actually do something useful. We need to give it the power to interact with the external world by invoking your code. This can be for:

 * Retrieving information relevant to the conversation
 * Performing operations as instructed by the user or your prompt

Right now, if you ask the bot how much the socks cost, it will hallucinate an answer (try it). You've given it no information about that, so it makes something up. Even if you said the price in your prompt, it couldn't reliably do arithmetic to multiply this by a desired quantity.

Define the following C# method:

```cs
[Description("Computes the price of socks, returning a value in dollars.")]
float GetPrice(
    [Description("The number of pairs of socks to calculate price for")] int count)
    => count * 15.99f;
```

Then, just above `while (true)`, define an `AIFunction` wrapping that method:

```cs
AIFunction getPriceTool = AIFunctionFactory.Create(GetPrice);
var chatOptions = new ChatOptions { Tools = [getPriceTool] };
```

... and finally update the existing `CompleteAsync` call to use it:

```cs
var response = await chatClient.CompleteAsync(messages, chatOptions);
```

Now if you run the app, you can try asking about the price, but you won't yet get an answer:

```
You: How much per pair?
Bot:
```

What's going on here? The LLM can't directly call your code. All it can do is return a response saying it *wants* you to call one of the functions you offered to it. If you want to see this for yourself, set a breakpoint right after `var response = ...` and ask about the price. In the debugger you'll see that `response.Message.Contents` contains an instance of `FunctionCallContent`, specifying the function name and arguments to use.

With `Microsoft.Extensions.AI`, the business of invoking functions is handled through the middleware pipeline. This decouples it from any particular `IChatClient` and lets all providers share a common implementation, keeping the programming model the same across all of them.

To enable automatic function invocation, go back to your `hostBuilder.Services.AddChatClient` call and insert the `UseFunctionInvocation` middleware:

```cs
hostBuilder.Services.AddChatClient(innerChatClient)
    .UseFunctionInvocation();
```

Now if you ask again:

```
You: Hey

Bot: Hey there! How's it going? Speaking of good vibes, have you checked out the latest deals on FOOTMONSTER socks?

You: OK, how much for 1000 pairs?

Bot: The price for 1000 pairs of FOOTMONSTER socks is $15,990! That's a great investment for some seriously cozy and stylish socks.
```

If you really want to see that it's invoking `GetPrice`, you can put a breakpoint on it.

If you want, check the options you can set when calling `UseFunctionInvocation`. You can control policies such as the maximum number of function calls allowed, whether or not exception information will be disclosed to the LLM, and so on.

## Adding more state

What if we want to manage per-conversation state, make it available to the bot, and update that state over time? Let's add a shopping cart:

```cs
class Cart
{
    public int NumPairsOfSocks { get; set; }

    [Description("Adds the specified number of pairs of socks to the cart")]
    public void AddSocksToCart(int numPairs)
    {
        NumPairsOfSocks += numPairs;
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("*****");
        Console.WriteLine($"Added {numPairs} pairs to your cart. Total: {NumPairsOfSocks} pairs.");
        Console.WriteLine("*****");
        Console.ForegroundColor = ConsoleColor.White;
    }

    [Description("Computes the price of socks, returning a value in dollars.")]
    public float GetPrice(
        [Description("The number of pairs of socks to calculate price for")] int count)
        => count * 15.99f;
}
```

As you see, we've moved `GetPrice` in here (and you can delete the old version of that method), plus defined an instance method that mutates state. The `[Description]` attribute may be placed on methods or parameters to give additional usage hints to the LLM.

Now to use this, replace this code:

```cs
AIFunction getPriceTool = AIFunctionFactory.Create(GetPrice);
var chatOptions = new ChatOptions { Tools = [getPriceTool] };
```

with this:

```cs
var cart = new Cart();
var getPriceTool = AIFunctionFactory.Create(cart.GetPrice);
var addToCartTool = AIFunctionFactory.Create(cart.AddSocksToCart);
var chatOptions = new ChatOptions { Tools = [addToCartTool, getPriceTool] };
```

The bot will now work with your cart data:

```
You: Hey

Bot: Hey there! How's it going? If you're on the lookout for some cozy socks, have you checked out the amazing deals on FOOTMONSTER socks? They're on sale right now!

You: How much for 150000 pairs?

Bot: The price for 150,000 pairs of socks comes to $2,398,500. That's a bulk purchase you'll be set for a while! If you're interested in adding some FOOTMONSTER socks to your cart, let me know! They're currently on sale!

You: Yeah add them

*****
Added 150000 pairs to your cart. Total: 150000 pairs.
*****

Bot: All set! You've added 150,000 pairs of FOOTMONSTER socks to your cart. Get ready for ultimate comfort and style! If you need anything else, just let me know!

You: Actually I need one more pair

*****
Added 1 pairs to your cart. Total: 150001 pairs.
*****

Bot: You've successfully added one more pair of FOOTMONSTER socks to your cart! That's a total of 150,001 pairs now. If you need anything else, I'm here to help!
```

**Experiment:** What if you want to let the user *remove* socks from their cart as well, or empty it? What's the minimum possible amount of extra code you need to write?

### Troubles with small models

If you're using GPT 3.5 or later, this code probably works great for you, and feels totally reliable. But on small 7-or-8-billion parameter models on Ollama, it may often:

 * Call methods unexpectedly, for example invoking `AddToCart(1)` even though you never asked
 * Fail to call methods when it should (e.g., hallucinating a price instead of calling `GetPrice`)
 * Produce invalid function call messages, causing XML and JSON to appear as messages to the user

Worse still, some small models don't even support functions/tools at all, and Ollama will just give an error message.

An interesting way you can try to mitigate this is with more prompt engineering. In fact, people have worked out that you can put a description of the tools into your prompt, and many small models will return a well-structured statement of how to call them, even if they don't officially support tool calls. Even small models that *do* officially support tool calls may become more reliable if you describe the available tools well in the prompt.

To explore this - and only if you're using Ollama - go back to your `IChatClient` middleware and add `UsePromptBasedFunctionCalling` right after `UseFunctionInvocation`:

```cs
hostBuilder.Services.AddChatClient(innerChatClient)
    .UseFunctionInvocation()
    .UsePromptBasedFunctionCalling();
```

`UsePromptBasedFunctionCalling` will automatically augment your prompt with a description of the available tools, and converts responses that look like tool call instructions into real `FunctionCallContent` instances that work with `UseFunctionInvocation`.

With this, `llama3.1` and `qwen2.5` will likely both do a decent job with this scenario. It's not a guarantee - they are still far from the solidity of `gpt-4o-mini`. But as long as you don't describe too many different tools and the scenario is kept simple, they tend to work. Bigger models on Ollama will of course work better, but you'll need a beast of a GPU to run them.

Note that `UsePromptBasedFunctionCalling` is an example in this repo. It's not a shipping part of `Microsoft.Extensions.AI`, because it's not reliable enough.

## Optional exercises

### Structured output

Go back to your `QuizApp` and change the logic in `SubmitAnswerAsync` so that it uses structured output.

## Middleware pipelines

One of the main design goals for `IChatClient` is to reuse standard implementations of cross-cutting concerns across all AI service provider implementations.

This is achieved by implementing those cross-cutting concerns as *middleware*. Built-in middleware currently includes:

 * Function invocation
 * Logging
 * Open Telemetry
 * Caching

Any middleware can freely be combined with other middleware and with any underlying AI service provider implementation. So, anyone building an `IChatClient` for a particular LLM backend doesn't need to implement their own version of function invocation, telemetry, etc.

You've already used two types of middleware earlier in this session: `UseLogging` and `UseFunctionInvocation`. Now let's take a look at how the middleware pipeline works and how you can implement custom pipeline steps.

## How the pipeline is built

When you register an `IChatClient` using code like this:

```cs
hostBuilder.Services.AddChatClient(innerChatClient)
    .UseLogging()
    .UseFunctionInvocation()
    .UseOpenTelemetry();
```

... that's actually shorthand for something like:

```cs
hostBuilder.Services.AddSingleton(services =>
{
    // Starting with the inner chat client, wrap in a nested sequence of steps
    var client0 = innerChatClient;
    var client1 = new OpenTelemetryChatClient(client0);
    var client2 = new FunctionInvokingChatClient(client1);
    var client3 = new LoggingChatClient(client2, someILoggerInstanceFromDI);

    // Return the outer chat client
    return client3;
});
```

So as you can see, the pipeline is a sequence of `IChatClient` instances, each of which holds a reference to the next one in the chain, until the final "inner" chat client (which is usually one that calls an external AI service over the network).

When there's a call, e.g., to `CompleteAsync`, this starts with the outer `IChatClient` which typically does something and passes the call through to the next in the chain, and this repeats all the way through to the end.

### What's the point of all this?

Middleware pipelines are an extremely flexible way to reuse logic. Each step in the chain can do any of the following:

 * Just pass the call through to the next `IChatClient` (default behavior)
 * Modify any of the parameters, such as adding extra prompts to the chat history (a.k.a. "prompt augmentation"), or mutating or replacing the `ChatOptions`
 * Return a result directly without calling the next in the chain (e.g., if resolving from a cache)
 * Delay before calling the next in the chain (e.g., for rate limiting)
 * And either **before** or **after** the next entry in the chain:
   * Trigger some side-effect, e.g., logging or emitting telemetry data about the input or output
   * Throw to reject the input/output

## Build custom middleware

We'll create some simple middleware that causes the LLM's response to come back in a different language than usual. This will be an example of *prompt augmentation*.

Start by defining a class like this:

```cs
public static class UseLanguageStep
{
    // This is an extension method that lets you add UseLanguageChatClient into a pipeline
    public static ChatClientBuilder UseLanguage(this ChatClientBuilder builder, string language)
    {
        return builder.Use(inner => new UseLanguageChatClient(inner, language));
    }

    // This is the actual middleware implementation
    private class UseLanguageChatClient(IChatClient next, string language) : DelegatingChatClient(next)
    {
        // TODO: Override CompleteAsync
    }
}
```

As you can see, it comes in two parts:

 * The actual implementation, which typically is a class derived from `DelegatingChatClient`. Use of that base class is optional (you can implement `IChatClient` directly if you prefer) but simplifies things by automatically passing through any calls to the next item in the pipeline.
 * An extension method on `ChatClientBuilder` that makes it easy to register into a pipeline.

Now to implement the logic, replace the `TODO: Override CompleteAsync` comment with an implementation, e.g.:

```cs
public override async Task<ChatCompletion> CompleteAsync(IList<ChatMessage> chatMessages, ChatOptions? options = null, CancellationToken cancellationToken = default)
{
    // Add an extra prompt
    var promptAugmentation = new ChatMessage(ChatRole.User, $"Always reply in the language {language}");
    chatMessages.Add(promptAugmentation);

    try
    {
        // Pass through to rest of pipeline
        return await base.CompleteAsync(chatMessages, options, cancellationToken);
    }
    finally
    {
        // Clean up
        chatMessages.Remove(promptAugmentation);
    }
}
```

The "clean up" phase here is optional. Doing this means the caller's chat history (i.e., their `List<ChatMessage>`) won't include the *Always reply in language...* message after the call completes. Normally it's good for prompt augmentation *not* to leave behind modifications to the chat history, but there may be cases where you do want to.

Now to use this, update your `AddChatClient` near the top of `Program.cs`:

```cs
hostBuilder.Services.AddChatClient(innerChatClient)
    .UseLanguage("Welsh")
    .UseFunctionInvocation();
```

Now even if you talk to it in English, you should get back a reply in Welsh:

```
You: Hello there!
Bot: Helo! Sut gallaf eich helpu heddiw? Peidiwch â cholli'r cyfle i brynu sgarffiau FOOTMONSTER sydd ar gael ar gynnig!
```

Things like function calling should continue to work the same.

Note that your `UseLanguage` middleware does **not** currently take effect for `CompleteStreamingAsync` calls, because you didn't override that method. It's not very hard to do this if you want.

## Optional: Build a rate-limiting middleware step

You're not limited to prompt augmentation. You can use arbitrary logic to decide if, when, and how to call through to the next step in the pipeline.

Can you build a middleware step that is used as follows?

```cs
hostBuilder.Services.AddChatClient(innerChatClient)
    .UseLanguage("Welsh")
    .UseRateLimit(TimeSpan.FromSeconds(5))
    .UseFunctionInvocation();
```

... and delays any incoming call so the user can't make more than one request every 5 seconds?

> [!TIP]
> Start by adding a package reference to `System.Threading.RateLimiting`.

Expand the section below for a possible solution.

<details>
<summary>SOLUTION</summary>

```cs
public static class UseRateLimitStep
{
    public static ChatClientBuilder UseRateLimit(this ChatClientBuilder builder, TimeSpan window)
        => builder.Use(inner => new RateLimitedChatClient(inner, window));

    private class RateLimitedChatClient(IChatClient inner, TimeSpan window) : DelegatingChatClient(inner)
    {
        RateLimiter rateLimit = new FixedWindowRateLimiter(new() { Window = window, QueueLimit = 1, PermitLimit = 1 });

        public override async Task<ChatCompletion> CompleteAsync(IList<ChatMessage> chatMessages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            using var lease = await rateLimit.AcquireAsync(cancellationToken: cancellationToken);
            return await base.CompleteAsync(chatMessages, options, cancellationToken);
        }
    }
}
```
</details>

And what if, instead of making the user wait until the 5 seconds has elapsed, you wanted to bail out and return a message like `"Sorry, I'm too busy - please ask again later"`?