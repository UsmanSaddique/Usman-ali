using System.Threading.Channels;

namespace AiDirector.Application.Pipeline;

/// Single-writer GPU work queue. The GPU is single-tenant, so pipeline runs are
/// processed one at a time by the hosted PipelineRunner — this makes that
/// explicit instead of relying on convention.
public interface IPipelineQueue
{
    ValueTask EnqueueAsync(string projectId, CancellationToken ct = default);
    IAsyncEnumerable<string> DequeueAllAsync(CancellationToken ct);
}

public sealed class PipelineQueue : IPipelineQueue
{
    private readonly Channel<string> _channel =
        Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true });

    // Projects sitting in the channel awaiting dequeue. Double-clicking Start (or
    // resume raced against full-auto) used to enqueue the same project twice and
    // run the pipeline back-to-back; duplicates are now dropped while queued.
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> _pending = new();

    public ValueTask EnqueueAsync(string projectId, CancellationToken ct = default)
    {
        if (!_pending.TryAdd(projectId, 0)) return ValueTask.CompletedTask;
        return _channel.Writer.WriteAsync(projectId, ct);
    }

    public async IAsyncEnumerable<string> DequeueAllAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        await foreach (var projectId in _channel.Reader.ReadAllAsync(ct))
        {
            _pending.TryRemove(projectId, out _);
            yield return projectId;
        }
    }
}
