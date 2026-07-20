using AiDirector.Application.Pipeline;
using AiDirector.Application.Safety;
using Microsoft.Extensions.DependencyInjection;

namespace AiDirector.Application;

public static class DependencyInjection
{
    public static IServiceCollection AddApplication(this IServiceCollection services)
    {
        services.AddSingleton<IPipelineQueue, PipelineQueue>();
        services.AddScoped<PipelineOrchestrator>();
        services.AddSingleton<YtSafetyGate>();
        return services;
    }
}
