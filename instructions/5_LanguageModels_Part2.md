# Language models (part 2)

This session continues on from [Language Models Part 1](./4_LanguageModels_Part1.md) by adding more advanced functionality. 

The main capabilities you'll learn about are *function calling* and *middleware*. Together, these provide flexible and powerful ways to acquire data dynamically, take actions in the external world, and control the internals of the system. Once you've learned this, you can implement most AI-based features!

## Project setup

Continue from the project `exercises/Chat/Begin` that you were using in [the previous session](./4_LanguageModels_Part1.md).

If at all possible, **it's really a lot better to use OpenAI/Azure OpenAI for the following exercises (rather than Ollama)**. If you temporarily switched to use Ollama for the structured output exercise, switch back now.

If you do want to use Ollama for the following exercises that's fine but you'll likely hit reliabiity issues. There's a whole section about this, and suggested mitigations, entitled *Troubles with small models* below.

## A chat loop

Chat-with-humans is a classic use case for LLMs, so let's implement it.

In `Program.cs`, delete whatever code you have below this line:

```cs
var chatClient = app.Services.GetRequiredService<IChatClient>();
```

... to reset back to a simple state. And once again, if you temporarily switched from OpenAI to Ollama for the preceding exercise, feel free to switch back to OpenAI now.

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

### Hiding the log spam

In this sample project, the log threshold is initially set to `LogLevel.Trace`. So by default, you'll see `trce` console lines like the following for each function call:

```
You: How much for 250
trce: Microsoft.Extensions.AI.FunctionInvokingChatClient[1953841678]
      Invoking __Main___g__GetPrice_0_1({
        "count": 250
      }).
trce: Microsoft.Extensions.AI.FunctionInvokingChatClient[700532736]
      __Main___g__GetPrice_0_1 invocation completed. Duration: 00:00:00.0002936. Result: 3997.5
Bot: The price for 250 pairs of FOOTMONSTER brand socks would be $3,997.50!
```

This is nice for debugging or just seeing what's going on, but will be distracting. Avoid this by finding the following line near the top:

```cs
hostBuilder.Services.AddLogging(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Trace));
```

... and change the `SetMinimumLevel` parameter value from `LogLevel.Trace` to `LogLevel.Information`.

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

... and delays any incoming call so that users can't make more than one request every 5 seconds?

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
        // Note that this rate limit is enforced globally across all users on your site.
        // It's not a separate rate limit for each user. You could do that but the implementation would be a bit different.
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

And what if, instead of making users wait until the 5 seconds has elapsed, you wanted to bail out and return a message like `"Sorry, I'm too busy - please ask again later"`?
