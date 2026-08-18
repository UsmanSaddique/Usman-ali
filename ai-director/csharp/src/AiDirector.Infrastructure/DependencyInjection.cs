using AiDirector.Application.Abstractions;
using AiDirector.Application.Archetypes;
using AiDirector.Application.Configuration;
using AiDirector.Infrastructure.Archetypes;
using AiDirector.Infrastructure.Audio;
using AiDirector.Infrastructure.ComfyUi;
using AiDirector.Infrastructure.Media;
using AiDirector.Infrastructure.Persistence;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace AiDirector.Infrastructure;

public static class DependencyInjection
{
    public static IServiceCollection AddInfrastructure(
        this IServiceCollection services, IConfiguration config)
    {
        services.AddOptions<AiDirectorOptions>()
            .Bind(config.GetSection(AiDirectorOptions.SectionName))
            .PostConfigure(o =>
            {
                // AIDIR_* env vars override a few common knobs (parity with config.py).
                if (Environment.GetEnvironmentVariable("AIDIR_AUTO_RESUME") is "1" or "true")
                    o.AutoResume = true;
            });

        var dbPath = config.GetSection(AiDirectorOptions.SectionName)["Paths:Database"]
                     ?? "ai_director.db";
        services.AddDbContext<AiDirectorDbContext>(opt =>
            opt.UseSqlite($"Data Source={dbPath}"));

        services.AddHttpClient<IComfyUiClient, ComfyUiClient>();
        services.AddSingleton<IWorkflowBuilder, WorkflowBuilder>();
        services.AddScoped<ILtxDirectorEngine, LtxDirectorEngine>();
        services.AddSingleton<IMediaRunner, MediaRunner>();
        services.AddScoped<IProjectRepository, ProjectRepository>();

        // Content archetypes: registry (cached YAML) + pure resolver.
        services.AddSingleton<IArchetypeRegistry, YamlArchetypeRegistry>();
        services.AddSingleton<ArchetypeResolver>();

        // Concrete services (injected directly into endpoints/pipeline).
        services.AddScoped<Assembler>();
        services.AddScoped<QaGates>();
        services.AddScoped<MusicEngine>();

        return services;
    }
}
