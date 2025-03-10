using Microsoft.Extensions.AI;

namespace CorrectiveRetrievalAugmentedGenerationApp;

public class ContextRelevancyEvaluator(IChatClient chatClient)
{
    public async Task<EvaluationResponse> EvaluateAsync(string question, string context, CancellationToken cancellationToken)
    {
        // Assess the quality of the answer
        // Note that in reality, "relevance" should be based on *all* the context we supply to the LLM, not just the citation it selects
        var response = await chatClient.GetResponseAsync<EvaluationResponse>($$"""
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
        """, cancellationToken: cancellationToken);

        if (response.TryGetResult(out var score) && score.Populated)
        {
            return score;
        }

        throw new InvalidOperationException("Invalid response from the AI assistant");
    }
}

public class EvaluationResponse
{
    public ScoreResponse? ContextRelevance { get; set; }

    public bool Populated => ContextRelevance is not null;
}

public class ScoreResponse
{
    public string? Justification { get; set; }
    public ScoreLabel ScoreLabel { get; set; }

    public double ScoreNumber => ScoreLabel switch
    {
        ScoreLabel.Awful => 0,
        ScoreLabel.Poor => 0.3,
        ScoreLabel.Good => 0.7,
        ScoreLabel.Perfect => 1,
        _ => throw new InvalidOperationException("Invalid score label")
    };
}

public enum ScoreLabel { Awful, Poor, Good, Perfect }
