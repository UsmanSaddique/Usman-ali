using System.Text.Json.Nodes;

namespace AiDirector.Infrastructure.ComfyUi;

/// Port of app/services/ltx_director.py GraphConverter: converts a ComfyUI
/// UI-format graph (with subgraphs, Set/Get nodes, "Anything Everywhere"
/// broadcasts, reroutes and bypassed nodes) into the flat API prompt format
/// {id: {class_type, inputs}}.
public sealed class LtxGraphConverter
{
    // widget names that are UI-side companions of a seed widget, never API inputs
    private static readonly HashSet<string> SeedControlValues =
        ["fixed", "increment", "decrement", "randomize"];

    // virtual/UI-only node types that never reach the API prompt
    private static readonly HashSet<string> VirtualTypes =
        ["SetNode", "GetNode", "Anything Everywhere", "Reroute",
         "MarkdownNote", "Note", "PrimitiveNode"];

    private readonly JsonObject _graph;
    private readonly JsonObject _objectInfo;
    private readonly Dictionary<string, JsonObject> _subgraphDefs = [];
    private readonly Dictionary<string, JsonObject> _nodes = [];       // flat id -> expanded node
    private readonly Dictionary<string, (string Fid, int Slot)> _links = [];
    private Dictionary<string, (string Fid, int Slot)> _registry = [];
    private int _uid;

    public LtxGraphConverter(JsonObject graph, JsonObject objectInfo)
    {
        _graph = graph;
        _objectInfo = objectInfo;
        if (graph["definitions"]?["subgraphs"] is JsonArray defs)
            foreach (var d in defs.OfType<JsonObject>())
                _subgraphDefs[d["id"]!.GetValue<string>()] = d;
    }

    private string NextId() => $"n{++_uid}";

    public JsonObject Convert()
    {
        Expand((JsonArray)_graph["nodes"]!, _graph["links"] as JsonArray ?? [],
            new Dictionary<int, (string, int)>(), "top");
        ResolveSetGet();
        ResolveEverywhere();
        return Emit();
    }

    // ── expansion ──────────────────────────────────────────────────────

    private Dictionary<int, (string Fid, int Slot)> Expand(
        JsonArray nodes, JsonArray links,
        Dictionary<int, (string Fid, int Slot)> outerInputMap, string linkNs)
    {
        var idMap = new Dictionary<long, string>();     // local node id -> flat id
        var subInstances = new List<JsonObject>();
        foreach (var n in nodes.OfType<JsonObject>())
        {
            var type = n["type"]!.GetValue<string>();
            if (_subgraphDefs.ContainsKey(type)) { subInstances.Add(n); continue; }
            var fid = NextId();
            idMap[n["id"]!.GetValue<long>()] = fid;
            _nodes[fid] = (JsonObject)n.DeepClone();
        }

        // normalize link records (top-level uses arrays, subgraphs use dicts)
        static (long Id, long OId, int OSlot, long TId, int TSlot) Rec(JsonNode l) =>
            l is JsonObject o
                ? (o["id"]!.GetValue<long>(), o["origin_id"]!.GetValue<long>(),
                   o["origin_slot"]!.GetValue<int>(), o["target_id"]!.GetValue<long>(),
                   o["target_slot"]!.GetValue<int>())
                : (((JsonArray)l)[0]!.GetValue<long>(), ((JsonArray)l)[1]!.GetValue<long>(),
                   ((JsonArray)l)[2]!.GetValue<int>(), ((JsonArray)l)[3]!.GetValue<long>(),
                   ((JsonArray)l)[4]!.GetValue<int>());

        var outMap = new Dictionary<int, (string, int)>();
        var pendingSubLinks = new List<(long Lid, long OId, int OSlot, long TId, int TSlot)>();
        foreach (var l in links)
        {
            if (l is null) continue;
            var (lid, oId, oSlot, tId, tSlot) = Rec(l);
            var key = $"{linkNs}:{lid}";
            if (oId == -10)                              // from subgraph input
            {
                if (outerInputMap.TryGetValue(oSlot, out var src)) _links[key] = src;
            }
            else if (idMap.TryGetValue(oId, out var fid))
            {
                _links[key] = (fid, oSlot);
            }
            else
            {
                pendingSubLinks.Add((lid, oId, oSlot, tId, tSlot));
            }
            if (tId == -20 && _links.TryGetValue(key, out var s))  // to subgraph output
                outMap[tSlot] = s;
        }

        // expand each subgraph instance
        foreach (var inst in subInstances)
        {
            var sdef = _subgraphDefs[inst["type"]!.GetValue<string>()];
            // 1. map DEF input slots -> parent link sources, matched by NAME
            //    (instances may omit unlinked inputs; positional mapping mis-wires)
            var byName = new Dictionary<string, (string, int)>();
            foreach (var iput in (inst["inputs"] as JsonArray ?? []).OfType<JsonObject>())
            {
                var link = iput["link"];
                if (link is null) continue;
                if (_links.TryGetValue($"{linkNs}:{link.GetValue<long>()}", out var src))
                    byName[iput["name"]!.GetValue<string>()] = src;
            }
            var instInMap = new Dictionary<int, (string, int)>();
            var defInputs = sdef["inputs"] as JsonArray ?? [];
            for (var dIdx = 0; dIdx < defInputs.Count; dIdx++)
            {
                var name = (defInputs[dIdx] as JsonObject)?["name"]?.GetValue<string>();
                if (name is not null && byName.TryGetValue(name, out var src))
                    instInMap[dIdx] = src;
            }
            var ns = $"{linkNs}/{inst["id"]!.GetValue<long>()}";
            var subOut = Expand((JsonArray)sdef["nodes"]!, sdef["links"] as JsonArray ?? [],
                instInMap, ns);
            // 2. instance output slot i -> inner source; register under the
            //    parent link ids attached to that output
            var outputs = inst["outputs"] as JsonArray ?? [];
            for (var idx = 0; idx < outputs.Count; idx++)
            {
                if (!subOut.TryGetValue(idx, out var src)) continue;
                foreach (var l in ((outputs[idx] as JsonObject)?["links"] as JsonArray ?? []))
                    if (l is not null)
                        _links[$"{linkNs}:{l.GetValue<long>()}"] = src;
            }
        }

        // unresolved forward references (source was a subgraph, now known)
        foreach (var (lid, _, _, tId, tSlot) in pendingSubLinks)
        {
            var key = $"{linkNs}:{lid}";
            if (_links.TryGetValue(key, out var src) && tId == -20)
                outMap[tSlot] = src;
        }

        // record namespace on nodes for input resolution later
        foreach (var fid in idMap.Values)
            _nodes[fid]["_ns"] = linkNs;
        return outMap;
    }

    // ── virtual node resolution ────────────────────────────────────────

    private (string Fid, int Slot)? SourceOf(JsonObject node, int inputIndex = 0)
    {
        var ns = node["_ns"]?.GetValue<string>() ?? "top";
        var inputs = node["inputs"] as JsonArray ?? [];
        if (inputIndex >= inputs.Count) return null;
        var link = (inputs[inputIndex] as JsonObject)?["link"];
        if (link is null) return null;
        var key = link.GetValueKind() == System.Text.Json.JsonValueKind.String
            ? $"{ns}:{link.GetValue<string>()}"
            : $"{ns}:{link.GetValue<long>()}";
        return _links.TryGetValue(key, out var src) ? src : null;
    }

    private static string TitleOf(JsonObject node)
    {
        if (node["widgets_values"] is JsonArray wv && wv.Count > 0 && wv[0] is not null)
            return wv[0]!.ToString();
        return node["title"]?.GetValue<string>() ?? "";
    }

    private void ResolveSetGet()
    {
        // SetNode registers its input under a title; GetNode outputs it.
        _registry = [];
        foreach (var n in _nodes.Values)
        {
            if (n["type"]!.GetValue<string>() != "SetNode") continue;
            var src = SourceOf(n, 0);
            if (src is not null) _registry[TitleOf(n)] = src.Value;
        }
        foreach (var n in _nodes.Values)
        {
            if (n["type"]!.GetValue<string>() != "GetNode") continue;
            if (!_registry.TryGetValue(TitleOf(n), out var src)) continue;
            var ns = n["_ns"]?.GetValue<string>() ?? "top";
            foreach (var oput in (n["outputs"] as JsonArray ?? []).OfType<JsonObject>())
                foreach (var l in (oput["links"] as JsonArray ?? []))
                    if (l is not null)
                        _links[$"{ns}:{l.GetValue<long>()}"] = src;
        }
    }

    private void ResolveEverywhere()
    {
        // 'Anything Everywhere' broadcasts its input to every unlinked input
        // of the same TYPE anywhere in the graph.
        var broadcasts = new Dictionary<string, (string, int)>();
        foreach (var n in _nodes.Values)
        {
            if (!n["type"]!.GetValue<string>().StartsWith("Anything Everywhere")) continue;
            var inputs = n["inputs"] as JsonArray ?? [];
            for (var idx = 0; idx < inputs.Count; idx++)
            {
                var src = SourceOf(n, idx);
                if (src is not null)
                    broadcasts[(inputs[idx] as JsonObject)?["type"]?.GetValue<string>() ?? "*"] = src.Value;
            }
        }
        if (broadcasts.Count == 0) return;
        foreach (var n in _nodes.Values)
        {
            var type = n["type"]!.GetValue<string>();
            if (VirtualTypes.Contains(type) || type.StartsWith("Anything Everywhere")) continue;
            var ns = n["_ns"]?.GetValue<string>() ?? "top";
            foreach (var iput in (n["inputs"] as JsonArray ?? []).OfType<JsonObject>())
            {
                var itype = iput["type"]?.GetValue<string>();
                if (iput["link"] is null && itype is not null &&
                    broadcasts.TryGetValue(itype, out var src))
                {
                    var fake = $"ue{++_uid}";      // synthesize a link
                    iput["link"] = fake;
                    _links[$"{ns}:{fake}"] = src;
                }
            }
        }
    }

    private (string Fid, int Slot)? Trace((string Fid, int Slot)? src, int depth = 0)
    {
        // Follow a source through virtual/bypassed nodes until a real emitting
        // node is reached.
        if (src is null || depth > 40) return null;
        var (fid, slot) = src.Value;
        if (!_nodes.TryGetValue(fid, out var n)) return null;
        var ntype = n["type"]!.GetValue<string>();
        var mode = n["mode"]?.GetValue<int>() ?? 0;
        if (ntype == "Reroute") return Trace(SourceOf(n, 0), depth + 1);
        if (ntype == "GetNode")
            return Trace(_registry.TryGetValue(TitleOf(n), out var reg)
                ? reg : null, depth + 1);
        if (ntype == "SetNode") return Trace(SourceOf(n, 0), depth + 1);
        if (mode is 2 or 4)     // muted/bypassed: pass through matching input type
        {
            string? outType = null;
            var outs = n["outputs"] as JsonArray ?? [];
            if (slot < outs.Count) outType = (outs[slot] as JsonObject)?["type"]?.GetValue<string>();
            var inputs = n["inputs"] as JsonArray ?? [];
            for (var idx = 0; idx < inputs.Count; idx++)
            {
                var itype = (inputs[idx] as JsonObject)?["type"]?.GetValue<string>();
                if (itype == outType || outType is "*" or null)
                {
                    var nxt = SourceOf(n, idx);
                    if (nxt is not null) return Trace(nxt, depth + 1);
                }
            }
            return null;
        }
        return (fid, slot);
    }

    // ── emission ───────────────────────────────────────────────────────

    /// Ordered (name, isWidget) input names that consume widgets_values entries.
    private List<(string Name, bool IsWidget)> WidgetInputNames(string classType)
    {
        var names = new List<(string, bool)>();
        if (_objectInfo[classType]?["input"] is not JsonObject input) return names;
        foreach (var section in new[] { "required", "optional" })
        {
            if (input[section] is not JsonObject sec) continue;
            foreach (var (name, spec) in sec)
            {
                var typeSpec = spec is JsonArray arr && arr.Count > 0 ? arr[0] : spec;
                var opts = spec is JsonArray arr2 && arr2.Count > 1 && arr2[1] is JsonObject o2
                    ? o2 : null;
                var isWidget = typeSpec is JsonArray || (typeSpec is JsonValue v &&
                    v.GetValueKind() == System.Text.Json.JsonValueKind.String &&
                    v.GetValue<string>() is "INT" or "FLOAT" or "STRING" or "BOOLEAN" or "COMBO");
                if (opts?["forceInput"]?.GetValue<bool>() == true) isWidget = false;
                names.Add((name, isWidget));
            }
        }
        return names;
    }

    private JsonObject Emit()
    {
        var prompt = new JsonObject();
        foreach (var (fid, n) in _nodes)
        {
            var ntype = n["type"]!.GetValue<string>();
            if (VirtualTypes.Contains(ntype) || ntype.StartsWith("Anything Everywhere") ||
                _subgraphDefs.ContainsKey(ntype)) continue;
            if ((n["mode"]?.GetValue<int>() ?? 0) is 2 or 4) continue;

            var apiInputs = new JsonObject();
            var inputs = n["inputs"] as JsonArray ?? [];

            // linked inputs
            var linkedNames = new HashSet<string>();
            for (var idx = 0; idx < inputs.Count; idx++)
            {
                if ((inputs[idx] as JsonObject)?["link"] is null) continue;
                var src = Trace(SourceOf(n, idx));
                if (src is null) continue;
                var name = (inputs[idx] as JsonObject)!["name"]!.GetValue<string>();
                apiInputs[name] = new JsonArray(src.Value.Fid, src.Value.Slot);
                linkedNames.Add(name);
            }

            // widget inputs
            var wv = n["widgets_values"];
            if (wv is JsonObject wvObj)
            {
                foreach (var (k, v) in wvObj)
                    if (!linkedNames.Contains(k) && !k.StartsWith("videopreview"))
                        apiInputs[k] = v?.DeepClone();
            }
            else if (wv is JsonArray wvArr)
            {
                var names = WidgetInputNames(ntype);
                var wi = 0;
                foreach (var (name, isWidget) in names)
                {
                    if (!isWidget) continue;
                    if (wi >= wvArr.Count) break;
                    var val = wvArr[wi];
                    wi++;
                    if (!linkedNames.Contains(name)) apiInputs[name] = val?.DeepClone();
                    // skip UI-only seed-control companion value
                    if (name is "seed" or "noise_seed" && wi < wvArr.Count &&
                        wvArr[wi] is JsonValue sv &&
                        sv.GetValueKind() == System.Text.Json.JsonValueKind.String &&
                        SeedControlValues.Contains(sv.GetValue<string>()))
                        wi++;
                }
            }
            prompt[fid] = new JsonObject { ["class_type"] = ntype, ["inputs"] = apiInputs };
        }
        return prompt;
    }
}
