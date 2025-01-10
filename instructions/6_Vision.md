# Vision

Many up-to-date LLMs support *multi-modal* input and output. That is, besides text, they can also receive or produce images or audio. In this session we'll experiment with image handling capabilities. We'll build a video monitoring system with very little code, which until just a couple of years ago would have been incredibly challenging.

## Project setup

*Prerequisites: These instructions assume you've done earlier sessions, in particular session 1, which gives the basic environment setup steps.*

Start by opening the project `exercises/Vision/Begin`. Near the top of `Program.cs`, find the variable `innerChatClient` and update its value according to the LLM service you wish to use.

For Azure OpenAI, you should have code like this:

```cs
var azureOpenAiConfig = hostBuilder.Configuration.GetRequiredSection("AzureOpenAI");
var innerChatClient = new AzureOpenAIClient(new Uri(azureOpenAiConfig["Endpoint"]!), new ApiKeyCredential(azureOpenAiConfig["Key"]!))
    .AsChatClient("gpt-4o-mini");
```

If you're using a model other than `gpt-4o-mini`, update this code. But do note that you must use a multi-modal model for this exercise.

For Ollama, you should assign a value like this:

```cs
IChatClient innerChatClient = new OllamaChatClient(
    new Uri("http://localhost:11434"), "llava");
```

`llava` is one of the most common small image-capable models. While it lacks certain features such as native function calling, it is quite fast and behaves well. If you don't already have it, run `ollama pull llava` to get it. Alternatively you could try [`x/llama3.2-vision`](https://ollama.com/x/llama3.2-vision) which is more capable but will be more demanding on your hardware.

## Getting image descriptions

In the `traffic-cam` directory, you'll find a series of images from traffic cameras. To get started, let's ask the LLM to describe one of these images in free-form text. Add the following code to the bottom of `Program.cs`:

```cs
var message = new ChatMessage(ChatRole.User, "What's in this image?");
message.Contents.Add(new ImageContent(File.ReadAllBytes(trafficImages[0]), "image/jpg"));

var response = await chatClient.CompleteAsync([message]);
Console.WriteLine(response.Message.Text);
```

If you run this, hopefully you'll get back a sensible description of the first image, perhaps along these lines:

```
The image appears to be a black and white aerial view of a highway. It shows multiple lanes of traffic, including several vehicles such as cars and a truck. The road seems to be wet, possibly indicating recent rain.
```

Clearly the LLM is able to understand the image. Next let's do this in a loop over all the images. Replace the code you just added with:

```cs
foreach (var imagePath in trafficImages)
{
    var name = Path.GetFileNameWithoutExtension(imagePath);

    var message = new ChatMessage(ChatRole.User, $$"""
        Extract information from this image from camera {{name}}.
        """);
    message.Contents.Add(new ImageContent(File.ReadAllBytes(imagePath), "image/jpg"));
    var response = await chatClient.CompleteAsync([message]);
    Console.WriteLine(response.Message.Text);
}
```

Again, it should work, with the following caveats:

 * If you're using `gpt-4o-mini`, it may occasionally make the strange claim that it can't view images, even though we know it can. No problem - we'll fix that in a moment.
 * These plain-text descriptions just aren't very useful. What we need is structured output so that we could take actions programmatically, such as updating traffic status on a map, or notifying road crews to attend to problems.
 * It's kind of slow, taking maybe 3-5 seconds per image.

### Optimizing for speed

If you just need speed and aren't hugely focused on quality or capabilities like tool calling, you could try the `moondream` model on Ollama. Even on a laptop GPU this can handle these images in under a second each. However it won't behave very well on the structured output task we'll get to next, so go back to `llava` or `gpt-4o-mini`.

## Structured output

Let's define a C# class to represent the structured data we want:

```cs
class TrafficCamResult
{
    public TrafficStatus Status { get; set; }
    public int NumCars { get; set; }
    public int NumTrucks { get; set; }

    public enum TrafficStatus { Clear, Flowing, Congested, Blocked };
}
```

Now update your code to use it. Replace this:

```cs
var response = await chatClient.CompleteAsync([message]);
Console.WriteLine(response.Message.Text);
```

... with this:

```cs
var response = await chatClient.CompleteAsync<TrafficCamResult>([message]);
if (response.TryGetResult(out var result))
{
    Console.WriteLine($"{name} status: {result.Status} (cars: {result.NumCars}, trucks: {result.NumTrucks})");
}
```

Now you should receive structured output that you could take programmatic action on. This also fixes any issues with `gpt-4o-mini` claiming that it "can't see images directly", since now it's constrained to giving a `TrafficCamResult`.

### Improving behavior on small models

> [!TIP]
> You can skip this subsection if you're using `gpt-4o-mini`.

If you're using `llava` on Ollama, you'll find it occasionally produces this output:

```
status: Clear (cars: 0, trucks: 0)
```

... even if there are cars and trucks in the image. This is because it's not great at understanding the JSON schema definition of the output format, so it sometimes returns some other JSON object that doesn't match your property names.

To improve this, we can give it an example of the desired JSON output shape. Change your prompt to:

```cs
var message = new ChatMessage(ChatRole.User, $$"""
    Extract information from this image from camera {{name}}.

    Respond with a JSON object in this form: {
        "Status": string // One of these values: "Clear", "Flowing", "Congested", "Blocked",
        "NumCars": number,
        "NumTrucks": number
    }
    """);
```

... and modify the `CompleteAsync<T>` call to:

```cs
var response = await chatClient.CompleteAsync<TrafficCamResult>([message], useNativeJsonSchema: isOllama);
```

Setting `useNativeJsonSchema` causes Microsoft.Extensions.AI *not* to augment the prompt with JSON schema (since it assumes the model accepts JSON schema natively, and doesn't need prompt augmentation). This reduces the complexity of the prompt, making smaller models more reliable.

It should be really quite reliable now. Note: You don't need to do this if you're using `gpt-4o-mini`.

## Raising alerts via native function calling

One of the drawbacks of structured output is that the model is forced to respond in your chosen format even if it's not actually appropriate for the input.

For example, consider image `3199.jpg` - there's no sensible `TrafficCamResult` for that input, but the model is forced to make one up, and your code never finds out there was a problem. Or what if you feed in an image of a sandwich and ask for a `TrafficCamResult` - what will it do?

To address this, let's raise alerts if there's something wrong with the input image. This could also be used to raise alerts if there's something dangerous or strange on the road.

> [!TIP]
> The following instructions require a model that supports native function calling, such as `gpt-4o-mini` or `x/llama3.2-vision`. If you're using `llava`, skip ahead to the next section.

Define a C# function to call if there's an alert. Add this right above your `foreach` loop:

```cs
var raiseAlert = AIFunctionFactory.Create((string cameraName, string alertReason) =>
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine("*** CAMERA ALERT ***");
    Console.WriteLine($"Camera {cameraName}: {alertReason}");
    Console.ForegroundColor = ConsoleColor.White;
}, "RaiseAlert");
var chatOptions = new ChatOptions { Tools = [raiseAlert] };
```

Now update your `CompleteAsync` call to use it:

```cs
var response = await chatClient.CompleteAsync<TrafficCamResult>([message], chatOptions, useNativeJsonSchema: isOllama);
```

And don't forget to actually enable function invocation in your pipeline! Add `UseFunctionInvocation` to your `hostBuilder.Services.AddChatClient` call as follows:

```cs
hostBuilder.Services.AddChatClient(innerChatClient)
    .UseFunctionInvocation();
```

If you run now, you may find it either never raises alerts, or does so even for situations that don't warrant one (e.g., traffic congestion). It's up to you to specify in your prompt what conditions should lead to an alert. For example:

```cs
var message = new ChatMessage(ChatRole.User, $$"""
    Extract information from this image from camera {{name}}.
    Raise an alert only if the camera is broken or if there's something highly unusual or dangerous,
    not just because of traffic volume.
    """);
```

Now the output for `3199.jpg` should be something like:

```
*** CAMERA ALERT ***
Camera 3199: Camera appears to be malfunctioning
3199 status: Blocked (cars: 0, trucks: 0)
```

... and there may be alerts for other cameras too.

## Raising alerts on non-function-calling models

If you're using `llava`, you can't use the above code because it will give an error saying `llava does not support tools`.

One way to fix this would be to use the `UsePromptBasedFunctionCalling` middleware we added in an earlier exercise. However that is not totally realistic for production use as it won't be reliable - the instructions describing how to call functions are too complicated for a model like `llava` to understand consistently.

A more reliable solution is not to use native function calling, and instead make alerting part of the structured output. For example, update `TrafficCamResult` to add an extra property:

```cs
public string? AlertText { get; set; }
```

Then describe how to use it in your prompt:

```cs
var message = new ChatMessage(ChatRole.User, $$"""
    Extract information from this image from camera {{name}}.
    Raise an alert only if the camera is broken or if there's something highly unusual or dangerous,
    not just because of traffic volume.

    Respond with a JSON object in this form: {
        "Status": string // One of these values: "Clear", "Flowing", "Congested", "Blocked",
        "NumCars": number,
        "NumTrucks": number,
        "AlertText": string // Null unless there's an alert
    }
    """);
```

... and of course update the output to display it:

```cs
Console.WriteLine($"{name} status: {result.Status} (cars: {result.NumCars}, trucks: {result.NumTrucks}, alert: {result.AlertText})");
```

You should now see alert text for the problematic images and not for the OK ones.

Bear in mind that a small model like `llava` is not going to compete on accuracy with bigger ones. If your hardware is capable enough, consider trying out a bigger llama-based model such as `x/llama3.2-vision` which supports both vision and native function calling.

## Optional: Add distributed caching

Processing images is fairly time-consuming. In systems that handle a large volume of data that might often repeat (for example, when multiple servers are simultaneously taking work from the same input feeds), you could avoid repeated work using caching.

In this case you should be able to do it using an approach like the following:

 * Start a Redis instance in Docker (e.g., `docker run --name my-redis -p 6379:6379 -d redis`)
 * Reference the `Microsoft.Extensions.Caching.StackExchangeRedis` package
 * Add an `IDistributedCache` to DI (`hostBuilder.Services.AddStackExchangeRedisCache(o => o.Configuration = "127.0.0.1:6379");`)
 * Add `.UseDistributedCache()` to your `IChatClient` pipeline, after `UseFunctionInvocation`

The first time you run it like this, it should take the same amount of time as usual, as it's populating the cache. But on subsequent runs, it should complete immediately.

Exercise: how does it differ if you move `UseDistributedCache` to be *after* `UseFunctionInvocation`? You'll have to shut down and restart your Redis instance to find out, to avoid reusing cache entries from before that change. Why does it differ? What use cases are there for each of these two options?
