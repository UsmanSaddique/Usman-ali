using System.Net.WebSockets;
using AiDirector.Application;
using AiDirector.Application.Abstractions;
using AiDirector.Infrastructure;
using AiDirector.WebApi.Endpoints;
using AiDirector.WebApi.Pipeline;
using AiDirector.WebApi.Realtime;
using Serilog;

var builder = WebApplication.CreateBuilder(args);

builder.Host.UseSerilog((ctx, cfg) => cfg
    .ReadFrom.Configuration(ctx.Configuration)
    .WriteTo.Console());

// The existing frontend expects snake_case JSON (Python/FastAPI parity), so the
// C# API must emit and accept snake_case property names to be a drop-in.
builder.Services.ConfigureHttpJsonOptions(o =>
{
    o.SerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.SnakeCaseLower;
    o.SerializerOptions.PropertyNameCaseInsensitive = true;
});

// Layers.
builder.Services.AddApplication();
builder.Services.AddInfrastructure(builder.Configuration);

// Real-time progress: one notifier instance shared by the pipeline and the WS endpoint.
builder.Services.AddSingleton<WebSocketProgressNotifier>();
builder.Services.AddSingleton<IProgressNotifier>(sp => sp.GetRequiredService<WebSocketProgressNotifier>());
builder.Services.AddHostedService<PipelineRunner>();

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddCors(o => o.AddDefaultPolicy(p =>
    p.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));

var app = builder.Build();

app.UseSerilogRequestLogging();
app.UseCors();
app.UseWebSockets();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// API.
app.MapProjectEndpoints();
app.MapSystemEndpoints();
app.MapProductionEndpoints();
app.MapMediaEndpoints();

// Pipeline progress WebSocket (payload shape matches the Python /ws/pipeline).
app.Map("/ws/pipeline/{projectId}", async (HttpContext ctx, string projectId,
    WebSocketProgressNotifier notifier) =>
{
    if (!ctx.WebSockets.IsWebSocketRequest) { ctx.Response.StatusCode = 400; return; }
    using WebSocket socket = await ctx.WebSockets.AcceptWebSocketAsync();
    await notifier.Register(projectId, socket, ctx.RequestAborted);
});

// Serve the existing frontend unchanged. Path is configurable; defaults to the
// repo's frontend/index.html so the C# backend drops in behind the same UI.
var frontend = app.Configuration["Frontend:IndexPath"] ?? ResolveFrontend(app.Environment.ContentRootPath);
app.MapGet("/", () => File.Exists(frontend)
    ? Results.Content(File.ReadAllText(frontend), "text/html")
    : Results.Content("<h1>AI Director (.NET)</h1><p>frontend/index.html not found.</p>", "text/html"));

app.Run();

static string ResolveFrontend(string contentRoot)
{
    // csharp/src/AiDirector.WebApi -> repo root (ai-director) is 3 up.
    var dir = contentRoot;
    for (var i = 0; i < 6 && dir is not null; i++)
    {
        var candidate = Path.Combine(dir, "frontend", "index.html");
        if (File.Exists(candidate)) return candidate;
        dir = Directory.GetParent(dir)?.FullName;
    }
    return Path.Combine(contentRoot, "frontend", "index.html");
}

// Exposed for integration tests (WebApplicationFactory).
public partial class Program;
