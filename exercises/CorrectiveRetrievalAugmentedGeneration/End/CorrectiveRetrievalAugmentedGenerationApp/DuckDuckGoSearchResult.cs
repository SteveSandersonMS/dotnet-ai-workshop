namespace CorrectiveRetrievalAugmentedGenerationApp;

/// <summary>
/// Represents the result of a DuckDuckGo search.
/// </summary>
/// <param name="Abstract">The abstract of the search result.</param>
/// <param name="Url">The URL of the search result.</param>
public record DuckDuckGoSearchResult(string Abstract, string Url);
