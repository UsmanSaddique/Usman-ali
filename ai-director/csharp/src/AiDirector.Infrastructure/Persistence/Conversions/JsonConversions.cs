using System.Text.Json;
using Microsoft.EntityFrameworkCore.ChangeTracking;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

namespace AiDirector.Infrastructure.Persistence.Conversions;

/// Converters for the SQLAlchemy JSON columns (stored as plain JSON text:
/// "[]", "{...}"). System.Text.Json round-trips them to typed CLR collections.
public static class JsonConversions
{
    private static readonly JsonSerializerOptions Opts = new(JsonSerializerDefaults.Web)
    {
        // Match Python's json.dumps default: no pretty-print, keep property names
        // as declared (converters below are used only for reference-type payloads
        // whose exact key casing is not schema-critical).
        WriteIndented = false,
    };

    public static ValueConverter<T, string> For<T>() => new(
        v => JsonSerializer.Serialize(v, Opts),
        s => Deserialize<T>(s));

    /// Deep-ish comparer so EF detects in-place mutation of collection columns.
    public static ValueComparer<T> Comparer<T>() => new(
        (a, b) => JsonSerializer.Serialize(a, Opts) == JsonSerializer.Serialize(b, Opts),
        v => v == null ? 0 : JsonSerializer.Serialize(v, Opts).GetHashCode(),
        v => Deserialize<T>(JsonSerializer.Serialize(v, Opts)));

    private static T Deserialize<T>(string? s)
    {
        if (string.IsNullOrWhiteSpace(s))
            return Activator.CreateInstance<T>()!;   // NULL column -> empty collection
        return JsonSerializer.Deserialize<T>(s, Opts) ?? Activator.CreateInstance<T>()!;
    }
}
