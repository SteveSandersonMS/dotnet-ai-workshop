namespace CorrectiveRetrievalAugmentedGenerationApp.Search;

public interface ISearchTool
{
    Task<IEnumerable<SearchResult>> SearchWebAsync(string query, int resultLimit = 3, CancellationToken cancellationToken = default);
}
