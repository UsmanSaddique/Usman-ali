using AiDirector.Application.Abstractions;
using AiDirector.Domain.Entities;
using Microsoft.EntityFrameworkCore;

namespace AiDirector.Infrastructure.Persistence;

public sealed class ProjectRepository(AiDirectorDbContext db) : IProjectRepository
{
    public async Task<Project?> GetAsync(string id, bool includeGraph = false, CancellationToken ct = default)
    {
        IQueryable<Project> q = db.Projects;
        if (includeGraph)
            q = q.Include(p => p.Scenes).ThenInclude(s => s.Generations)
                 .Include(p => p.MusicTracks)
                 .Include(p => p.RenderJobs);
        return await q.FirstOrDefaultAsync(p => p.Id == id, ct);
    }

    public Task<List<Project>> ListAsync(CancellationToken ct = default) =>
        db.Projects.OrderByDescending(p => p.CreatedAt).ToListAsync(ct);

    public async Task AddAsync(Project project, CancellationToken ct = default) =>
        await db.Projects.AddAsync(project, ct);

    public Task SaveAsync(CancellationToken ct = default) => db.SaveChangesAsync(ct);
}
