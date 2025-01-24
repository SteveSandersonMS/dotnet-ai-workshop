using Microsoft.AspNetCore.Components;
using Microsoft.Extensions.AI;
using System.ComponentModel.DataAnnotations;

namespace QuizApp.Components.Pages;

// TODO: Get an IChatClient from DI
public partial class Quiz : ComponentBase
{
    // TODO: Decide on a quiz subject
    private const string QuizSubject = "Your choice of subject goes here. Be descriptive.";

    private ElementReference answerInput;
    private int numQuestions = 5;
    private int pointsScored = 0;

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

        // TODO:
        //  - Ask the LLM for a question on the subject 'QuizSubject'
        //  - Assign the question text to 'currentQuestionText'
        //  - Make sure it doesn't repeat the previous questions
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

        // TODO:
        //  - Ask the LLM whether the answer 'UserAnswer' is correct for the question 'currentQuestionText'
        //  - If it's correct, increment 'pointsScored'
        //  - Set 'currentQuestionOutcome' to a string explaining why the answer is correct or incorrect
        currentQuestionOutcome = "TODO: Determine whether that's correct";
    }

    protected override async Task OnAfterRenderAsync(bool firstRender)
        => await answerInput.FocusAsync();
}
