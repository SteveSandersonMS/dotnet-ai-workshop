# Realtime

> [!TIP]
> **This chapter uses an OpenAI-specific feature and cannot currently be done using Ollama**. If you don't have access to OpenAI or Azure OpenAI, you'll need either to pair-program with someone who does, or skip this chapter and try out some of the more advanced exercises from other chapters instead.

**Realtime** is a relatively recent step forwards in capabilities, currently limited to OpenAI and Azure OpenAI. It's all about producing **low-latency multi-modal interaction**, so that users can speak instead of type, and get reactions at near-human speeds.

### How it works, and why

You might think that speech input/output would look like this:

 1. User speaks
 2. Run speech-to-text on the audio to get an input string
 3. Pass input string to LLM; get output string back
 4. Run text-to-speech on the output string to get audio
 5. Play the output audio

... and in fact that is one way to do it. Traditional (non-realtime) chat completion APIs do handle audio input/output that way. But it has big drawbacks:

 * It can be slow, because the LLM doesn't see the user input until they stop talking and the speech-to-text phase completes
 * It can sound unnatural, because the text-to-speech part doesn't understand the subject of the conversation and where to put emphases or emotion

These drawbacks are overcome by the realtime APIs, since:

 * They process input while the user is talking
 * It's a true audio-to-audio model, without needing any intermediate text representation. Since the output audio is produced directly by the AI model that decides what to say, it can place emphasis or emotion based on the real subject of the conversation.

### Use cases

 * **Human-initiated conversation** (e.g., receiving phone calls from humans, and acting as a virtual customer support agent)
 * **Computer-initiated conversation** (e.g., providing verbal notifications about something that happened, or calling businesses to collect information or carry out tasks on behalf of a human)
 * **Software automation** (augmenting applications so that users can verbally instruct them what to do, usually much more flexibly and intelligently than with a GUI alone)

## Getting access to a suitable model

If you're using OpenAI with a paid account, you already have access to the `gpt-4o-realtime-preview-2024-10-01` model (or possibly a newer version). You can move on to the next section.

If you're using Azure OpenAI, then you need to deploy an instance of the `gpt-4o-realtime-preview` model. Use the Azure Portal to create an Azure OpenAI resource in [one of the regions that supports realtime](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?tabs=python-secure%2Cglobal-standard%2Cstandard-chat-completions#gpt-4o-audio), which at the time of writing is `eastus2` and `swedencentral`. Deploy the `gpt-4o-realtime-preview` model to that resource. If you're switching to a new resource, be sure to update the endpoint and key that you're using locally:

```
cd exercises/Realtime/Begin

dotnet user-secrets set "AzureOpenAI:Endpoint" https://HOSTNAME.openai.azure.com/
dotnet user-secrets set "AzureOpenAI:Key" abcdabcdabcdabcd
```

## Open and run the project

Now open and run the project `exercises/Realtime/Begin`.

 * If you're using Visual Studio, open the `.sln` file in that directory
 * If you're using VS Code, run `code exercises/Realtime/Begin`

Run it via Ctrl+F5 to launch without debugging. It should launch a browser window. Initially you'll see this error:

```
There is no registered service of type 'OpenAI.RealtimeConversation.RealtimeConversationClient'.
```

### Registering the service in DI

Open `Program.cs` and make sure that `openAiClient`, defined near the top, is set up to use whichever one of Azure OpenAI or OpenAI Platform you're using. In other words, toggle which one of the two is commented out if you need.

Next find this comment:

```cs
// TODO: Register RealtimeConversationClient in DI
```

Replace it with the following registration:

```cs
var realtimeClient = openAiClient.GetRealtimeConversationClient("gpt-4o-realtime-preview"); // Update the model name if your deployment is different
builder.Services.AddSingleton(realtimeClient);
```

Now if you run again, there should be no error. You'll be prompted by the browser to grant access to your microphone - do so. And then you'll be treated to the delightful message *TODO: Implement the whole thing*.

**You don't need to care about how any of the web UI works here**, and I won't even explain it (but feel free to inspect the code). We're only going to focus on the realtime AI APIs.

## Starting a session

Open `ConversationManager.cs`, and update `RunAsync` to start up a conversation session:

```cs
public async Task RunAsync(Stream audioInput, Speaker audioOutput, Func<string, Task> addMessageAsync, CancellationToken cancellationToken)
{
    var prompt = "You are very grumpy physics professor. A student is here to see you. Respond very tersely and minimally.";
    await addMessageAsync("Connecting...");

    var sessionOptions = new ConversationSessionOptions()
    {
        Instructions = prompt,
        Voice = ConversationVoice.Shimmer,
    };

    session = await client.StartConversationSessionAsync();
    await session.ConfigureSessionAsync(sessionOptions);
    await addMessageAsync("Connected");
}
```

> [!TIP]
> If you see `Connecting...` but not `Connected`, and no error, then it's likely that you are making too many connections too quickly. At present, Azure OpenAI throttles the realtime API to a maximum of 10 sessions per minute (or 1 every 6 seconds). Wait a few seconds and reload. Obviously this can't be used in production yet.

## Audio input

At the end of `RunAsync`, add this code:

```cs
await foreach (var update in session.ReceiveUpdatesAsync(cancellationToken))
{
    switch (update)
    {
        case ConversationSessionStartedUpdate:
            await addMessageAsync("Conversation started");
            _ = Task.Run(async () => await session.SendInputAudioAsync(audioInput, cancellationToken));
            break;

        case ConversationInputSpeechStartedUpdate:
            await addMessageAsync("Speech started");
            await audioOutput.ClearPlaybackAsync(); // If the user interrupts, stop talking
            break;

        case ConversationInputSpeechFinishedUpdate:
            await addMessageAsync("Speech finished");
            break;
    }
}
```

As you can see, this waits for a `ConversationSessionStartedUpdate` message, and then begins piping the input audio data (which in this case comes from the user's microphone via their browser) into the session.

If you run now, you should see it detects when you start and stop talking into your microphone, e.g.:

```
Connecting...
Connected
Conversation started
Speech started
Speech finished
Speech started
Speech finished
Speech started
Speech finished
```

By default, it's using server-side voice activity detection (VAD). The client sends a continuous audio feed, and the server determines when the user starts and stops talking. Alternatively you can disable "turn detection" and use your own logic for this (e.g., for a "push to talk" UI).

## Peek behind the scenes

Before moving on, add the following right above your `switch` statement:

```cs
await addMessageAsync(update.GetType().Name);
```

This logs the type name of every single message in the session. Now if you run, you'll see a huge list of messages, including a lot of `ConversationItemStreamingPartDeltaUpdate` instances which indicate the model is sending audio or text output.

Comment out the line you just added, otherwise it will be hard to read what's going on. But you can re-enable it if you need to debug anything.

## Audio output

Inside your `switch` statement, add:

```cs
case ConversationItemStreamingPartDeltaUpdate outputDelta:
    await audioOutput.EnqueueAsync(outputDelta.AudioBytes?.ToArray());
    break;
```

Give it a go! You should now be able to have a conversation with the world's least helpful physics professor.

Feel free to try out other prompts.

## Capturing transcriptions

There may be many reasons to keep a log of the text of these conversations. You can capture a transcription of both the input and output.

Let's start by transcribing the user input. Add this to your `switch` statement:

```cs
case ConversationInputTranscriptionFinishedUpdate inputTranscription:
    await addMessageAsync($"User: {inputTranscription.Transcript}");
    break;
```

But on its own, this won't do anything. As mentioned before, you're using a real audio-to-audio model, so by default there isn't any transcription for it to output. The user input never becomes text. If you want to enable transcription of user input, you must also update your `sessionOptions`:

```cs
var sessionOptions = new ConversationSessionOptions()
{
    // ... leave other options unchanged ...
    InputTranscriptionOptions = new() { Model = "whisper-1" },
};
```

Now you should see proper transcriptions of the user input. Strangely you might observe it produce incorrect transcriptions even though the audio response correctly matches what you said. Again, this is because the transcription is a separate process, and the model's output doesn't depend on it.

Next let's produce transcriptions of the output. These arrive in successive chunks so you can stream them into the UI if you want. But we'll just capture them in a `StringBuilder`. 

Just before your `await foreach`, add:

```cs
var outputTranscription = new StringBuilder();
```

... and then right after the call to `audioOutput.EnqueueAsync`, add:

```cs
outputTranscription.Append(outputDelta.Text ?? outputDelta.AudioTranscript);
```

The `Text` property is used if you request text output. We'll get to that in a bit. For now it only actually produces `AudioTranscript`.

Finally, to display it, add this to your `switch` statement:

```cs
case ConversationItemStreamingAudioTranscriptionFinishedUpdate:
case ConversationItemStreamingTextFinishedUpdate:
    await addMessageAsync($"Assistant: {outputTranscription}");
    outputTranscription.Clear();
    break;
```

Strangely you may notice the input and output transcriptions may appear in the wrong order (i.e., output before input). It's so fast to produce output that it sometimes does it before it even finishes transcribing the input. Once again, this is because the output doesn't depend on transcribing the input - it's audio-to-audio.

## Function calling (a.k.a., doing something real)

There's not much point chatting with an AI if it can't actually do anything or find any real information for you. Fortunately, the realtime API also supports function calling.

For this example we'll make a restaurant booking assistant. The premise is that customers may phone in to get information about availability and to place bookings. Of course, many other business scenarios follow similar patterns.

Change the prompt to this:

```cs
var prompt = $"""
    You are a virtual assistant for a cowboy-themed restaurant called The Wild Brunch.
    You are receiving a phone call from a possible customer.
    Respond as succinctly as possible, ideally in just a few words, but occasionally use cowboy phrases.
    The current date is {DateTime.Now.ToLongDateString()}
    """;
```

Also add the following class (either in a separate file, or nested inside `ConversationManager`), which could be used to perform operations against external APIs or track information on a per-user basis:

```cs
public class BookingContext(Func<string, Task> addMessage)
{
    [Description("Determines whether a table is available on a given date for a given number of people")]
    public bool CheckTableAvailability(DateOnly date, int numPeople)
    {
        await addMessage($"Checking table availability for {date}");
        return Random.Shared.NextDouble() < (1.0 / numPeople);
    }

    public async void BookTableAsync(DateOnly date, int numPeople, string customerName)
    {
        await addMessage($"***** Booked table on {date} for {numPeople} people (name: {customerName}).");
    }
}
```

Next let's instantiate `BookingContext` and make its functions available to the session. Find this code:

```cs
var sessionOptions = new ConversationSessionOptions() { ... };
```

... and right after that line, add:

```cs
var bookingContext = new BookingContext(addMessageAsync);
var checkTableAvailabilityTool = AIFunctionFactory.Create(bookingContext.CheckTableAvailability);
var bookTableTool = AIFunctionFactory.Create(bookingContext.BookTableAsync);
List<AIFunction> tools = [checkTableAvailabilityTool, bookTableTool];

foreach (var tool in tools)
{
    sessionOptions.Tools.Add(tool.ToConversationFunctionTool());
}
```

As you can see, this attaches two tools to the session, corresponding to the two methods on `BookingContext`.

However, if you run it now, it won't actually call the functions. Try getting it to book a table for you. It might hallucinate and behave as if it's placed a booking, but in fact you won't see `***** Booked table...` in the output.

This is similar to how `IChatClient` doesn't call functions automatically unless you add the `UseFunctionInvocation` middleware. But for `RealtimeConversationSession` there isn't yet any middleware so we'll have to do this manually.

Add this *after* your `switch` statement:

```cs
// This is actually an extension method provided by Microsoft.Extensions.AI.OpenAI
// It knows how to invoke an AIFunction and continue the realtime conversation
await session.HandleToolCallsAsync(update, tools);
```

If you run it now, it should work. Try booking a table - it will call `CheckTableAvailability` to determine availability, then will reply to the user.

Remember that realtime is currently in beta/preview and can't yet be used in production. It has plenty of quirks, and the code you're using in this example might itself be imperfect (e.g., the audio sometimes gets cut off at the end). Hopefully this will all be streamlined by the time it's ready for production usage.

## Optional exercise

Try to modify your code to represent an *outbound* call example. Your goal is to collect specific information from the human. For example, this may be an outbound call to a vehicle mechanic, asking for the price and availability for servicing a particular car model.

Have the AI introduce itself and try to get all the necessary information. Once the human has supplied all the requested information, call a function to simulate storing the results, then end the conversation.

Tip: you can call `session.StartResponseAsync()` as soon as you receive a `ConversationSessionStartedUpdate`. That will make the AI begin talking without waiting for the user to speak.

### Thought exercise

You might imagine that, in the near future, your phone's voice assistant might be able to find and call all the top-rated local vehicle mechanics to get the best quote for you, and you won't have to speak with any of them! 

But wait a minute... what if the vehicle mechanics are using AI to answer their phones as well? Now it's just computers talking to computers, in the most unbelievably inefficient way!

How could we make this efficient?
