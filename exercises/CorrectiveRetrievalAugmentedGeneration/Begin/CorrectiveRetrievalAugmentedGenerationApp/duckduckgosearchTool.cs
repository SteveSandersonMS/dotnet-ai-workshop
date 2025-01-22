using System.Collections.Specialized;
using System.Text.Json;

namespace CorrectiveRetrievalAugmentedGenerationApp;

/// <summary>
/// Represents a tool for performing web searches using the DuckDuckGo API.
/// </summary>
public class DuckDuckGoSearchTool(HttpClient client)
{
    /// <summary>
    /// Searches the web using the DuckDuckGo API.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="cancellationToken">A token to cancel the operation.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains the abstract of the search result.</returns>
    public async Task<DuckDuckGoSearchResult> SearchWebAsync(string query, CancellationToken cancellationToken = default)
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

        return new DuckDuckGoSearchResult(answerAbstract ?? string.Empty, url ?? string.Empty);
    }
}

/// <summary>
/// Represents the result of a DuckDuckGo search.
/// </summary>
/// <param name="Abstract">The abstract of the search result.</param>
/// <param name="Url">The URL of the search result.</param>
public record DuckDuckGoSearchResult(string Abstract, string Url);
