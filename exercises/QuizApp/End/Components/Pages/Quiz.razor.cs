using Microsoft.AspNetCore.Components;
using Microsoft.Extensions.AI;
using System.ComponentModel.DataAnnotations;

namespace QuizApp.Components.Pages;

public partial class Quiz(IChatClient chatClient) : ComponentBase
{
    private const string QuizSubject = "Modern .NET programming";

    private ElementReference answerInput;
    private int numQuestions = 5;
    private int pointsScored = 0;
    private string previousQuestions = "";

    private int currentQuestionNumber = 0;
    private string? currentQuestionText;
    private string? currentQuestionOutcome;
    private bool answerSubmitted;
    private bool DisableForm => currentQuestionText is null || answerSubmitted;

    [Required]
    public string? UserAnswer { get; set; }

    protected override Task OnInitializedAsync()
        => MoveToNextQuestionAsync();

    private async Task MoveToNextQuestionAsync()
    {
        // Can't move on until you answer the question and we mark it
        if (currentQuestionNumber > 0 && string.IsNullOrEmpty(currentQuestionOutcome))
        {
            return;
        }

        // Reset state for the next question
        currentQuestionNumber++;
        currentQuestionText = null;
        currentQuestionOutcome = null;
        answerSubmitted = false;
        UserAnswer = null;

        // Get the next question text
        var prompt = $"""
            Provide a quiz question about the following subject: {QuizSubject}
            Reply only with the question and no other text. Ask factual questions for which
            the answer only needs to be a single word or phrase.
            Don't repeat any of the previous questions: {previousQuestions}
            """;
        var response = await chatClient.GetResponseAsync(prompt);
        currentQuestionText = response.Text;
        previousQuestions += currentQuestionText + "\n";
    }

    private async Task SubmitAnswerAsync()
    {
        // Prevent double-submission
        if (answerSubmitted)
        {
            return;
        }

        // Mark the answer
        answerSubmitted = true;
        var prompt = $"""
            You are marking quiz answers as correct or incorrect.
            The quiz subject is {QuizSubject}.
            The question is: {currentQuestionText}

            The student's answer is as follows, enclosed in valid XML tags:
            <student_answer>
            {UserAnswer!.Replace("<", "")}
            </student_answer>

            If the literal text within <student_answer></student_answer> above was written on an exam
            paper, would a human examiner accept it as correct for the question {currentQuestionText}?

            Your response must start with CORRECT: or INCORRECT:
            followed by an explanation or another remark about the question.
            Examples: CORRECT: And did you know, Jupiter is made of gas?
                      INCORRECT: The Riemann hypothesis is still unsolved.
            """;
        var response = await chatClient.GetResponseAsync(prompt);
        currentQuestionOutcome = response.Text;

        // There's a better way to do this using structured output. We'll get to that later.
        if (currentQuestionOutcome.StartsWith("CORRECT"))
        {
            pointsScored++;
        }
    }

    protected override async Task OnAfterRenderAsync(bool firstRender)
        => await answerInput.FocusAsync();
}
