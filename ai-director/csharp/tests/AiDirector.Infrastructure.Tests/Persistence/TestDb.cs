using AiDirector.Infrastructure.Persistence;
using Microsoft.EntityFrameworkCore;

namespace AiDirector.Infrastructure.Tests.Persistence;

/// Helpers for opening the real ai_director.db against the C# EF Core model.
/// Always works on a COPY so tests can never mutate the live database.
public static class TestDb
{
    /// The live Python database, located relative to the repo.
    public static string LiveDbPath()
    {
        // tests/AiDirector.Infrastructure.Tests/bin/Debug/net10.0 -> repo root is 6 up.
        var dir = AppContext.BaseDirectory;
        for (var i = 0; i < 8 && dir is not null; i++)
        {
            var candidate = Path.Combine(dir, "ai_director.db");
            if (File.Exists(candidate)) return candidate;
            dir = Directory.GetParent(dir)?.FullName;
        }
        // Fall back to the known absolute location.
        return @"C:\Users\PC\Desktop\VideoMaker\ai-director\ai_director.db";
    }

    /// Copy the live DB (+ WAL/SHM sidecars) to a throwaway temp file and open it.
    public static (AiDirectorDbContext Ctx, string TempPath) OpenCopy()
    {
        var live = LiveDbPath();
        var temp = Path.Combine(Path.GetTempPath(),
            $"aidir_test_{Guid.NewGuid():N}.db");
        File.Copy(live, temp, overwrite: true);
        foreach (var ext in new[] { "-wal", "-shm" })
            if (File.Exists(live + ext)) File.Copy(live + ext, temp + ext, true);

        var options = new DbContextOptionsBuilder<AiDirectorDbContext>()
            .UseSqlite($"Data Source={temp}")
            .Options;
        return (new AiDirectorDbContext(options), temp);
    }

    public static bool LiveDbExists() => File.Exists(LiveDbPath());
}
