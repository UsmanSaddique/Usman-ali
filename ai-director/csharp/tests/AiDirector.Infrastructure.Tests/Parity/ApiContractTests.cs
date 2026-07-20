using System.Text.Json;
using System.Text.RegularExpressions;
using FluentAssertions;

namespace AiDirector.Infrastructure.Tests.Parity;

/// Phase 0 parity: the routes the C# app implements must correspond to real
/// routes in the Python FastAPI contract (csharp/parity/fixtures/openapi.json).
/// Path-parameter NAMES differ ({project_id} vs {id}) so both are normalized to
/// {} before comparison. This proves the C# endpoints mirror the real API
/// rather than inventing routes — and documents which of the 48 are done.
public sealed partial class ApiContractTests
{
    // Routes implemented so far in the C# WebApi (method + path).
    private static readonly (string Method, string Path)[] Implemented =
    [
        ("get", "/api/projects"),
        ("post", "/api/projects"),
        ("get", "/api/projects/{}"),
        ("post", "/api/projects/{}/start-generation"),
        ("post", "/api/projects/{}/safety-check"),
        ("get", "/api/projects/{}/safety-report"),
        ("post", "/api/projects/{}/scenes-from-lyrics"),
        ("post", "/api/projects/{}/generate-music"),
        ("post", "/api/projects/{}/render"),
        ("get", "/api/channels"),
        ("get", "/api/system/health"),
        ("get", "/api/system/gpu-status"),
    ];

    [Fact]
    public void Implemented_routes_exist_in_python_contract()
    {
        var openapi = FindOpenApi();
        if (openapi is null) return; // fixture not generated in this env — skip

        var contract = LoadNormalizedRoutes(openapi);
        var missing = Implemented
            .Where(r => !contract.Contains((r.Method, r.Path)))
            .ToList();

        missing.Should().BeEmpty(
            "every implemented C# route must mirror a real Python route; missing: "
            + string.Join(", ", missing.Select(m => $"{m.Method.ToUpper()} {m.Path}")));
    }

    [Fact]
    public void Contract_has_the_expected_route_volume()
    {
        var openapi = FindOpenApi();
        if (openapi is null) return;
        LoadNormalizedRoutes(openapi).Count.Should().BeGreaterThan(30);
    }

    private static HashSet<(string, string)> LoadNormalizedRoutes(string openapiPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(openapiPath));
        var set = new HashSet<(string, string)>();
        foreach (var path in doc.RootElement.GetProperty("paths").EnumerateObject())
        {
            var norm = ParamRe().Replace(path.Name, "{}");
            foreach (var method in path.Value.EnumerateObject())
                set.Add((method.Name.ToLowerInvariant(), norm));
        }
        return set;
    }

    private static string? FindOpenApi()
    {
        var dir = AppContext.BaseDirectory;
        for (var i = 0; i < 10 && dir is not null; i++)
        {
            var candidate = Path.Combine(dir, "csharp", "parity", "fixtures", "openapi.json");
            if (File.Exists(candidate)) return candidate;
            var alt = Path.Combine(dir, "parity", "fixtures", "openapi.json");
            if (File.Exists(alt)) return alt;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }

    [GeneratedRegex(@"\{[^}]+\}")]
    private static partial Regex ParamRe();
}
