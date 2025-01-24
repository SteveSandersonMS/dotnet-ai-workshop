using System.ComponentModel;
using System.Text;
using Microsoft.Extensions.AI;
using OpenAI.RealtimeConversation;
using Realtime.Components;

namespace Realtime;

public class ConversationManager(RealtimeConversationClient client) : IDisposable
{
    private RealtimeConversationSession? session;

    public async Task RunAsync(Stream audioInput, Speaker audioOutput, Func<string, Task> addMessageAsync, CancellationToken cancellationToken)
    {
        var prompt = $"""
            You are a virtual assistant for a cowboy-themed restaurant called The Wild Brunch.
            You are receiving a phone call from a possible customer.
            Respond as succinctly as possible, ideally in just a few words, but occasionally use cowboy phrases.
            The current date is {DateTime.Now.ToLongDateString()}
            """;

        await addMessageAsync("Connecting...");

        var sessionOptions = new ConversationSessionOptions()
        {
            Instructions = prompt,
            Voice = ConversationVoice.Shimmer,
            InputTranscriptionOptions = new() { Model = "whisper-1" },
        };

        var bookingContext = new BookingContext(addMessageAsync);
        var checkTableAvailabilityTool = AIFunctionFactory.Create(bookingContext.CheckTableAvailability);
        var bookTableTool = AIFunctionFactory.Create(bookingContext.BookTableAsync);
        List<AIFunction> tools = [checkTableAvailabilityTool, bookTableTool];

        foreach (var tool in tools)
        {
            sessionOptions.Tools.Add(tool.ToConversationFunctionTool());
        }

        session = await client.StartConversationSessionAsync();
        await session.ConfigureSessionAsync(sessionOptions);
        await addMessageAsync("Connected");
        var outputTranscription = new StringBuilder();

        await foreach (var update in session.ReceiveUpdatesAsync(cancellationToken))
        {
            await addMessageAsync(update.GetType().Name);

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

                case ConversationItemStreamingPartDeltaUpdate outputDelta:
                    await audioOutput.EnqueueAsync(outputDelta.AudioBytes?.ToArray());
                    outputTranscription.Append(outputDelta.Text ?? outputDelta.AudioTranscript);
                    break;

                case ConversationInputTranscriptionFinishedUpdate inputTranscription:
                    await addMessageAsync($"User: {inputTranscription.Transcript}");
                    break;

                case ConversationItemStreamingAudioTranscriptionFinishedUpdate:
                case ConversationItemStreamingTextFinishedUpdate:
                    await addMessageAsync($"Assistant: {outputTranscription}");
                    outputTranscription.Clear();
                    break;
            }

            // This is actually an extension method provided by Microsoft.Extensions.AI.OpenAI
            // It knows how to invoke an AIFunction and continue the realtime conversation
            await session.HandleToolCallsAsync(update, tools);
        }
    }

    public void Dispose()
    {
        session?.Dispose();
    }

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
}
