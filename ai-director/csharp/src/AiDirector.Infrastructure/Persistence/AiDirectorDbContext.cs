using System.Text;
using System.Text.RegularExpressions;
using AiDirector.Domain.Entities;
using AiDirector.Domain.ValueObjects;
using AiDirector.Infrastructure.Persistence.Conversions;
using Microsoft.EntityFrameworkCore;

namespace AiDirector.Infrastructure.Persistence;

/// EF Core context mapped to the SAME schema the Python/SQLAlchemy app uses,
/// so the existing ai_director.db keeps working unchanged. Table names are the
/// Python __tablename__ values; column names are snake_case (SQLAlchemy default);
/// enums and JSON columns use the custom converters.
public sealed partial class AiDirectorDbContext(DbContextOptions<AiDirectorDbContext> options)
    : DbContext(options)
{
    public DbSet<Channel> Channels => Set<Channel>();
    public DbSet<Project> Projects => Set<Project>();
    public DbSet<Scene> Scenes => Set<Scene>();
    public DbSet<Generation> Generations => Set<Generation>();
    public DbSet<MusicTrack> MusicTracks => Set<MusicTrack>();
    public DbSet<RenderJob> RenderJobs => Set<RenderJob>();
    public DbSet<SafetyReport> SafetyReports => Set<SafetyReport>();
    public DbSet<LoRA> Loras => Set<LoRA>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        base.OnModelCreating(b);

        b.Entity<Channel>(e =>
        {
            e.ToTable("channels");
            e.HasKey(x => x.Id);
            e.Property(x => x.DefaultLoras).HasJsonColumn();
            e.HasMany(x => x.Projects).WithOne(p => p.Channel)
                .HasForeignKey(p => p.ChannelId);
        });

        b.Entity<Project>(e =>
        {
            e.ToTable("projects");
            e.HasKey(x => x.Id);
            e.Property(x => x.Status).HasConversion(PythonEnumConversions.ProjectStatusConverter());
            e.Property(x => x.DefaultLoraIds).HasJsonColumn();
            e.Property(x => x.DefaultLoraWeights).HasJsonColumn();
            e.HasMany(x => x.Scenes).WithOne(s => s.Project).HasForeignKey(s => s.ProjectId);
            e.HasMany(x => x.MusicTracks).WithOne(m => m.Project).HasForeignKey(m => m.ProjectId);
            e.HasMany(x => x.RenderJobs).WithOne(r => r.Project).HasForeignKey(r => r.ProjectId);
        });

        b.Entity<Scene>(e =>
        {
            e.ToTable("scenes");
            e.HasKey(x => x.Id);
            e.Ignore(x => x.ActiveGeneration);
            e.Ignore(x => x.LatestGeneration);
            e.Property(x => x.SceneType).HasConversion(PythonEnumConversions.SceneTypeConverter());
            e.Property(x => x.Status).HasConversion(PythonEnumConversions.SceneStatusConverter());
            e.Property(x => x.LoraIds).HasJsonColumn();
            e.Property(x => x.LoraWeights).HasJsonColumn();
            e.Property(x => x.DirectorNotes).HasJsonColumn();
            e.HasMany(x => x.Generations).WithOne(g => g.Scene).HasForeignKey(g => g.SceneId);
        });

        b.Entity<Generation>(e =>
        {
            e.ToTable("generations");
            e.HasKey(x => x.Id);
            e.Property(x => x.Status).HasConversion(PythonEnumConversions.GenerationStatusConverter());
            e.Property(x => x.Parameters).HasJsonColumn();
        });

        b.Entity<MusicTrack>(e =>
        {
            e.ToTable("music_tracks");
            e.HasKey(x => x.Id);
        });

        b.Entity<RenderJob>(e =>
        {
            e.ToTable("render_jobs");
            e.HasKey(x => x.Id);
            e.Property(x => x.Status).HasConversion(PythonEnumConversions.RenderStatusConverter());
            e.Property(x => x.RenderSettings).HasJsonColumn();
        });

        b.Entity<SafetyReport>(e =>
        {
            e.ToTable("safety_reports");
            e.HasKey(x => x.Id);
            e.Property(x => x.Verdict).HasConversion(PythonEnumConversions.SafetyVerdictConverter());
            e.Property(x => x.Issues).HasJsonColumn();
            e.Property(x => x.CheckedFields).HasJsonColumn();
            e.Property(x => x.AutoRevisions).HasJsonColumn();
        });

        b.Entity<LoRA>(e =>
        {
            e.ToTable("loras");
            e.HasKey(x => x.Id);
            e.Property(x => x.TriggerWords).HasJsonColumn();
        });

        // SQLAlchemy names columns after the attribute (snake_case). Apply the
        // same convention to every mapped column so we never drift from the DB.
        foreach (var entity in b.Model.GetEntityTypes())
            foreach (var prop in entity.GetProperties())
                prop.SetColumnName(ToSnakeCase(prop.Name));
    }

    internal static string ToSnakeCase(string name)
    {
        if (string.IsNullOrEmpty(name)) return name;
        // Insert '_' at lower→Upper and letter→digit-run boundaries, then lower.
        var s = SnakeBoundary().Replace(name, "$1_$2");
        return s.ToLowerInvariant();
    }

    [GeneratedRegex(@"([a-z0-9])([A-Z])")]
    private static partial Regex SnakeBoundary();
}

file static class JsonColumnExtensions
{
    /// Configure a collection/dictionary property as a JSON text column
    /// (converter + value comparer) in one call.
    public static void HasJsonColumn<T>(
        this Microsoft.EntityFrameworkCore.Metadata.Builders.PropertyBuilder<T> p)
    {
        p.HasConversion(JsonConversions.For<T>(), JsonConversions.Comparer<T>());
    }
}
