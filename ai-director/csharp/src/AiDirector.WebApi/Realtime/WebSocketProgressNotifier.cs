using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using AiDirector.Application.Abstractions;

namespace AiDirector.WebApi.Realtime;

/// IProgressNotifier backed by raw WebSockets, one group per project. Payload
/// shape mirrors the Python /ws/pipeline messages so frontend/index.html needs
/// no changes.
public sealed class WebSocketProgressNotifier : IProgressNotifier
{
    private readonly ConcurrentDictionary<string, ConcurrentDictionary<Guid, WebSocket>> _groups = new();

    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    public async Task Register(string projectId, WebSocket socket, CancellationToken ct)
    {
        var group = _groups.GetOrAdd(projectId, _ => new());
        var id = Guid.NewGuid();
        group[id] = socket;
        try
        {
            // Hold the connection open until the client disconnects.
            var buffer = new byte[1024];
            while (socket.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                var result = await socket.ReceiveAsync(buffer, ct);
                if (result.MessageType == WebSocketMessageType.Close) break;
            }
        }
        catch (OperationCanceledException) { /* shutting down */ }
        catch (WebSocketException) { /* client vanished */ }
        finally { group.TryRemove(id, out _); }
    }

    public async Task PublishAsync(string projectId, ProgressUpdate update, CancellationToken ct = default)
    {
        if (!_groups.TryGetValue(projectId, out var group)) return;
        var payload = JsonSerializer.SerializeToUtf8Bytes(new
        {
            type = "progress",
            project_id = projectId,
            update.Stage,
            update.Status,
            update.Percent,
            update.Message,
            scene_id = update.SceneId,
            scene_number = update.SceneNumber,
        }, Json);

        foreach (var (id, socket) in group)
        {
            if (socket.State != WebSocketState.Open) { group.TryRemove(id, out _); continue; }
            try { await socket.SendAsync(payload, WebSocketMessageType.Text, true, ct); }
            catch { group.TryRemove(id, out _); }
        }
    }
}
