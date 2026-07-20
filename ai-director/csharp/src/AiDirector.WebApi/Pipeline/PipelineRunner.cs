using AiDirector.Application.Pipeline;
using AiDirector.Domain.Enums;
using AiDirector.Infrastructure.Persistence;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace AiDirector.WebApi.Pipeline;

/// Hosted service that drains the single-tenant GPU queue, running one pipeline
/// at a time. Each run gets its own DI scope (fresh DbContext).
public sealed class PipelineRunner(
    IPipelineQueue queue,
    IServiceScopeFactory scopes,
    ILogger<PipelineRunner> log) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        log.LogInformation("PipelineRunner started");
        await RecoverInterruptedProjectsAsync(stoppingToken);
        await foreach (var projectId in queue.DequeueAllAsync(stoppingToken))
        {
            try
            {
                using var scope = scopes.CreateScope();
                var orchestrator = scope.ServiceProvider.GetRequiredService<PipelineOrchestrator>();
                await orchestrator.RunAsync(projectId, stoppingToken);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested) { break; }
            catch (Exception e)
            {
                log.LogError(e, "Pipeline run failed for project {Id}", projectId);
            }
        }
    }

    /// Port of main.py::_recover_interrupted_projects. A process killed mid-run
    /// leaves projects/scenes in Generating; without this they display as
    /// "generating" forever after a restart. Rolls them back to their checkpoint
    /// (Approved / Pending) — deliberately does NOT auto-resume: the user does
    /// not want GPU runs starting on boot, recovered projects wait for an
    /// explicit Resume click.
    private async Task RecoverInterruptedProjectsAsync(CancellationToken ct)
    {
        try
        {
            using var scope = scopes.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<AiDirectorDbContext>();

            var stuck = await db.Projects.Include(p => p.Scenes)
                .Where(p => p.Status == ProjectStatus.Generating)
                .ToListAsync(ct);
            foreach (var project in stuck)
            {
                foreach (var s in project.Scenes.Where(s => s.Status == SceneStatus.Generating))
                    s.Status = SceneStatus.Pending;
                project.Status = ProjectStatus.Approved;
                log.LogWarning("[Recovery] {Title} was interrupted mid-generation — rolled back to " +
                               "checkpoint, waiting for explicit Resume", project.Title);
            }
            if (stuck.Count > 0) await db.SaveChangesAsync(ct);
        }
        catch (Exception e)
        {
            // Never let recovery stop the queue from starting.
            log.LogError(e, "[Recovery] startup checkpoint recovery failed");
        }
    }
}
