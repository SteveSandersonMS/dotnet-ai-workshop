using System.Collections.Specialized;
using System.Text.Json;

namespace CorrectiveRetrievalAugmentedGenerationApp.Search;

/// <summary>
/// Represents a tool for performing web searches using the DuckDuckGo API.
/// </summary>
public class DuckDuckGoSearchTool(HttpClient client) : ISearchTool
{
    /// <summary>
    /// Searches the web using the DuckDuckGo API.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="cancellationToken">A token to cancel the operation.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the abstract of the search result.</returns>
    public async Task<IEnumerable<SearchResult>> SearchWebAsync(string query, int resultLimit = 3, CancellationToken cancellationToken = default)
    {
        NameValueCollection queryString = System.Web.HttpUtility.ParseQueryString(string.Empty);

        queryString.Add("q", query);
        queryString.Add("format", "json");
        var uri = new UriBuilder("https://api.duckduckgo.com")
        {
            Query = queryString.ToString()
        };
        var response = await client.GetAsync(uri.ToString(), cancellationToken);

        // read all json from response
        string jsonString = await response.Content.ReadAsStringAsync(cancellationToken);

        var json = JsonDocument.Parse(jsonString).RootElement;
        string? answerAbstract = json.GetProperty("Abstract").GetString();
        string? url = json.GetProperty("AbstractURL").GetString();

        return [new SearchResult(answerAbstract ?? string.Empty, url ?? string.Empty, answerAbstract ?? string.Empty)];
    }
}
