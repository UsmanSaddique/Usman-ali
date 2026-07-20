using AiDirector.Application.Abstractions;
using AiDirector.Application.Pipeline;
using AiDirector.Domain.Entities;
using AiDirector.Domain.Enums;
using AiDirector.WebApi.Contracts;
using Microsoft.EntityFrameworkCore;
using AiDirector.Infrastructure.Persistence;

namespace AiDirector.WebApi.Endpoints;

public static class ProjectEndpoints
{
    public static void MapProjectEndpoints(this IEndpointRouteBuilder app)
    {
        var g = app.MapGroup("/api/projects");

        g.MapGet("", async (IProjectRepository repo, CancellationToken ct) =>
            Results.Ok((await repo.ListAsync(ct)).Select(ProjectSummary.From)));

        g.MapGet("/{id}", async (string id, IProjectRepository repo, CancellationToken ct) =>
        {
            var p = await repo.GetAsync(id, includeGraph: true, ct);
            return p is null ? Results.NotFound() : Results.Ok(ProjectDetail.From(p));
        });

        g.MapPost("", async (CreateProjectRequest req, AiDirectorDbContext db, CancellationToken ct) =>
        {
            var channelExists = await db.Channels.AnyAsync(c => c.Id == req.ChannelId, ct);
            if (!channelExists) return Results.BadRequest(new { error = "unknown channel_id" });

            var project = new Project
            {
                Title = req.Title,
                ChannelId = req.ChannelId,
                DurationTarget = req.DurationTarget,
                ProjectType = req.ProjectType,
                Context = req.Context,
                Lyrics = req.Lyrics,
                Status = ProjectStatus.Draft,
            };
            db.Projects.Add(project);
            await db.SaveChangesAsync(ct);
            return Results.Created($"/api/projects/{project.Id}", ProjectSummary.From(project));
        });

        // Enqueue generation onto the single-tenant GPU queue. The orchestrator
        // only picks up Pending/Queued/Failed scenes, so this doubles as resume:
        // finished scenes are reused, never regenerated. "resume" and "full-auto"
        // are the names the frontend calls (parity with the Python routes).
        foreach (var route in new[] { "/{id}/start-generation", "/{id}/resume", "/{id}/full-auto" })
        {
            g.MapPost(route,
                async (string id, IProjectRepository repo, IPipelineQueue queue, CancellationToken ct) =>
            {
                var p = await repo.GetAsync(id, ct: ct);
                if (p is null) return Results.NotFound();

                // A Generating status left behind by a dead run is a ghost — the
                // queue is single-tenant and idle, so roll it back and continue
                // rather than stranding the project.
                if (p.Status is ProjectStatus.Generating)
                {
                    p.Status = ProjectStatus.Approved;
                    p.ErrorLog = null;
                    await repo.SaveAsync(ct);
                }

                await queue.EnqueueAsync(id, ct);
                return Results.Accepted($"/api/projects/{id}", new { status = "queued" });
            });
        }
    }
}
