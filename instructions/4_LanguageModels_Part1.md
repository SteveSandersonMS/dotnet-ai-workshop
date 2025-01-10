# Language models (part 1)

In this session we'll go through some of the basic features of language models and the APIs we use with them.

This session is **part one of two** about language models. In the [second part](5_LanguageModels_Part2.md), we'll build on this by making a chatbot that can take real-world actions on the user's behalf, and show how you can plug in middleware to control it in more detail.

## Project setup

*Prerequisites: These instructions assume you've done earlier sessions, in particular [session 1](1_BuildAQuizApp.md), which gives the basic environment setup steps.*

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
    public int Price { get; set; }
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

## Optional exercises

### Structured output

Go back to your `QuizApp` from [session 1](./1_BuildAQuizApp.md) and change the logic in `SubmitAnswerAsync` so that it uses structured output.
