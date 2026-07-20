using AiDirector.Domain.Enums;
using FluentAssertions;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace AiDirector.Infrastructure.Tests.Persistence;

/// Phase 1 acceptance: the C# EF Core model reads the REAL ai_director.db that
/// the Python/SQLAlchemy app writes — same tables, columns, enum spellings, and
/// JSON columns — without migration or data loss.
public sealed class DatabaseCompatibilityTests
{
    [Fact]
    public void Reads_projects_with_valid_enum_status()
    {
        if (!TestDb.LiveDbExists()) return; // skip if DB absent
        var (ctx, temp) = TestDb.OpenCopy();
        try
        {
            var projects = ctx.Projects.AsNoTracking().ToList();
            projects.Should().NotBeEmpty("the live DB has projects");
            // Every status string in the DB mapped to a defined enum value.
            projects.Should().OnlyContain(p => Enum.IsDefined(p.Status));
        }
        finally { ctx.Dispose(); Cleanup(temp); }
    }

    [Fact]
    public void Project_count_matches_raw_sql()
    {
        if (!TestDb.LiveDbExists()) return;
        var (ctx, temp) = TestDb.OpenCopy();
        try
        {
            var efCount = ctx.Projects.Count();
            using var conn = new SqliteConnection($"Data Source={temp}");
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = "SELECT COUNT(*) FROM projects";
            var rawCount = Convert.ToInt32(cmd.ExecuteScalar());
            efCount.Should().Be(rawCount);
        }
        finally { ctx.Dispose(); Cleanup(temp); }
    }

    [Fact]
    public void Reads_scenes_including_mixed_case_status_rows()
    {
        if (!TestDb.LiveDbExists()) return;
        var (ctx, temp) = TestDb.OpenCopy();
        try
        {
            // scenes.status in the live DB mixes Python names ("PENDING") and a
            // lowercase value ("draft"); the tolerant converter maps both.
            var scenes = ctx.Scenes.AsNoTracking().ToList();
            scenes.Should().OnlyContain(s => Enum.IsDefined(s.Status));
            scenes.Should().OnlyContain(s => Enum.IsDefined(s.SceneType));
        }
        finally { ctx.Dispose(); Cleanup(temp); }
    }

    [Fact]
    public void Reads_json_columns_into_typed_collections()
    {
        if (!TestDb.LiveDbExists()) return;
        var (ctx, temp) = TestDb.OpenCopy();
        try
        {
            // JSON columns can't be filtered in SQL, so materialize then inspect.
            var scene = ctx.Scenes.AsNoTracking().AsEnumerable()
                .FirstOrDefault(s => s.DirectorNotes.Count > 0);
            if (scene is not null)
            {
                // director_notes JSON deserialized to a populated dictionary.
                scene.DirectorNotes.Keys.Should().NotBeEmpty();
            }

            var gen = ctx.Generations.AsNoTracking().AsEnumerable()
                .FirstOrDefault(g => g.Parameters.Count > 0);
            if (gen is not null)
                gen.Parameters.Keys.Should().NotBeEmpty();

            // List columns never come back null (NULL -> empty list).
            ctx.Projects.AsNoTracking().ToList()
                .Should().OnlyContain(p => p.DefaultLoraIds != null);
        }
        finally { ctx.Dispose(); Cleanup(temp); }
    }

    [Fact]
    public void Navigates_project_to_scenes_to_generations()
    {
        if (!TestDb.LiveDbExists()) return;
        var (ctx, temp) = TestDb.OpenCopy();
        try
        {
            var project = ctx.Projects
                .Include(p => p.Scenes).ThenInclude(s => s.Generations)
                .AsNoTracking()
                .FirstOrDefault(p => p.Scenes.Count > 0);
            if (project is not null)
            {
                project.Scenes.Should().NotBeEmpty();
                // Navigation loads eagerly; scene_number is populated for each.
                project.Scenes.Should().OnlyContain(s => s.SceneNumber >= 0);
            }
        }
        finally { ctx.Dispose(); Cleanup(temp); }
    }

    private static void Cleanup(string temp)
    {
        foreach (var ext in new[] { "", "-wal", "-shm" })
            try { File.Delete(temp + ext); } catch { /* best effort */ }
    }
}
