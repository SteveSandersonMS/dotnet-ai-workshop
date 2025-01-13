using System.Collections.Specialized;
using System.Text.Json;
using System.Web;

namespace CorrectiveRetrievalAugmentedGenerationApp;

/// <summary>
/// Represents a tool for performing web searches using the Bing Search API. Follow instructions here to get an API key: https://docs.microsoft.com/en-us/azure/cognitive-services/bing-web-search/quickstarts/csharp#prerequisites
/// </summary>
/// <param name="bingSearchApiKey">The API key for accessing the Bing Search API.</param>
/// <param name="client">The HTTP client used to send requests.</param>
/// <param name="allowedSiteList">An optional list of allowed sites to restrict the search results.</param>
internal class BingSearchTool(string bingSearchApiKey, HttpClient client, IReadOnlyList<string>? allowedSiteList = null)
{
    private const string BingSearchApiUrl = "https://api.bing.microsoft.com/v7.0/search";

    internal static readonly JsonSerializerOptions s_jsonSerializerOptions = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true };

    /// <summary>
    /// Searches the web using the Bing Search API.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="resultLimit">The maximum number of results to return.</param>
    /// <param name="cancellationToken">A token to cancel the operation.</param>
    /// <returns>A collection of search results.</returns>
    public async Task<IEnumerable<BingSearchResult>> SearchWebAsync(string query, int resultLimit = 3, CancellationToken cancellationToken = default)
    {
        NameValueCollection queryString = HttpUtility.ParseQueryString(string.Empty);
        string scopedQuery = BuildQuery(query);
        queryString["q"] = scopedQuery;

        using HttpRequestMessage request = new(HttpMethod.Get, $"{BingSearchApiUrl}?{queryString}");
        request.Headers.Add("Ocp-Apim-Subscription-Key", bingSearchApiKey);

        using HttpResponseMessage response = await client.SendAsync(request, cancellationToken);
        response.EnsureSuccessStatusCode();

        Stream stream = await response.Content.ReadAsStreamAsync(cancellationToken);

        BingSearchResponse result =
            await JsonSerializer.DeserializeAsync<BingSearchResponse>(stream, s_jsonSerializerOptions,
                cancellationToken) ??
            BingSearchResponse.Empty;
        IEnumerable<BingSearchResult> selectedResults = result.WebPages.Value.Take(resultLimit);

        return selectedResults;

        string BuildQuery(string userQuery)
        {
            if (allowedSiteList?.Count > 0)
            {
                List<string> requiredSitesSyntax = allowedSiteList.Select(site => $"site:{site}").ToList();
                string rightHandSide = requiredSitesSyntax.Count == 1
                    ? $" {requiredSitesSyntax[0]}"
                    : $" ({string.Join(" OR ", requiredSitesSyntax)})";
                string final = $"{userQuery}{rightHandSide}";
                return final;
            }

            return userQuery;
        }
    }

    private record WebPages(IReadOnlyList<BingSearchResult> Value);

    private record BingSearchResponse(WebPages WebPages)
    {
        public static readonly BingSearchResponse Empty = new(new WebPages([]));
    }
}
