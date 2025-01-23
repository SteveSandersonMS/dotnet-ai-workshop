namespace CorrectiveRetrievalAugmentedGenerationApp.Search;

/// <summary>
/// Represents a search result.
/// </summary>
/// <param name="Name">The name or title of the search result.</param>
/// <param name="Url">The URL of the search result.</param>
/// <param name="Snippet">A brief snippet or description of the search result.</param>
public record SearchResult(string Name, string Url, string Snippet);
