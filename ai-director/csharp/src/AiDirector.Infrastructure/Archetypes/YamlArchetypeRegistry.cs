using AiDirector.Application.Archetypes;
using AiDirector.Application.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace AiDirector.Infrastructure.Archetypes;

/// Loads archetypes/*.yaml with YamlDotNet (underscored keys -> PascalCase
/// properties). Parity with load_archetypes() in app/services/archetypes.py.
/// A broken YAML is logged and skipped, never fatal.
public sealed class YamlArchetypeRegistry : IArchetypeRegistry
{
    private readonly ILogger<YamlArchetypeRegistry> _log;
    private readonly string _dir;
    private readonly object _gate = new();
    private Dictionary<string, ContentArchetype>? _cache;

    private static readonly IDeserializer Yaml = new DeserializerBuilder()
        .WithNamingConvention(UnderscoredNamingConvention.Instance)
        .IgnoreUnmatchedProperties()
        .Build();

    public YamlArchetypeRegistry(IOptions<AiDirectorOptions> options,
        ILogger<YamlArchetypeRegistry> log)
    {
        _log = log;
        _dir = options.Value.Paths.ArchetypesDir;
    }

    public IReadOnlyDictionary<string, ContentArchetype> All()
    {
        if (_cache is not null) return _cache;
        lock (_gate)
        {
            if (_cache is not null) return _cache;
            _cache = Load();
            return _cache;
        }
    }

    public ContentArchetype? Get(string? id) =>
        string.IsNullOrEmpty(id) ? null : (All().TryGetValue(id, out var a) ? a : null);

    private Dictionary<string, ContentArchetype> Load()
    {
        var map = new Dictionary<string, ContentArchetype>(StringComparer.Ordinal);
        if (!Directory.Exists(_dir))
        {
            _log.LogWarning("[archetypes] dir not found: {Dir}", _dir);
            return map;
        }

        foreach (var path in Directory.EnumerateFiles(_dir, "*.yaml").OrderBy(p => p))
        {
            try
            {
                var raw = Yaml.Deserialize<ContentArchetype>(File.ReadAllText(path));
                if (raw is null) continue;
                raw.Id ??= Path.GetFileNameWithoutExtension(path);
                map[raw.Id] = raw;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[archetypes] failed to load {File}", Path.GetFileName(path));
            }
        }

        _log.LogInformation("[archetypes] loaded {Count}: {Ids}",
            map.Count, string.Join(", ", map.Keys.OrderBy(k => k)));
        return map;
    }
}
