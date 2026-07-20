using System.Text.Json.Nodes;
using AiDirector.Application.Abstractions;
using AiDirector.Application.Configuration;
using AiDirector.Application.Pipeline;
using AiDirector.Domain.Entities;
using AiDirector.Domain.Enums;
using FluentAssertions;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using NSubstitute;
using NSubstitute.ExceptionExtensions;

namespace AiDirector.Application.Tests;

/// Pins the pipeline's failure semantics: scenes retry with a budget
/// (max_retries), permanent failures don't kill the run, and an interrupted run
/// rolls back to its checkpoint instead of ghosting Generating statuses.
public sealed class PipelineRetryTests
{
    private readonly IProjectRepository _repo = Substitute.For<IProjectRepository>();
    private readonly IComfyUiClient _comfy = Substitute.For<IComfyUiClient>();
    private readonly IWorkflowBuilder _workflows = Substitute.For<IWorkflowBuilder>();
    private readonly IProgressNotifier _notifier = Substitute.For<IProgressNotifier>();
    private readonly Project _project;

    public PipelineRetryTests()
    {
        _project = new Project
        {
            Title = "retry test", ChannelId = "ch", Status = ProjectStatus.Approved,
            VideoModel = "test-model.gguf",
        };
        _project.Scenes.Add(new Scene
        {
            ProjectId = _project.Id, SceneNumber = 1, Prompt = "p", NegativePrompt = "n",
            Duration = 5.0, Status = SceneStatus.Pending, MaxRetries = 3,
        });

        _repo.GetAsync(_project.Id, includeGraph: true, Arg.Any<CancellationToken>())
            .Returns(_project);
        _comfy.WaitReadyAsync(Arg.Any<double>(), Arg.Any<bool>(), Arg.Any<CancellationToken>())
            .Returns(true);
        _workflows.ZImage(Arg.Any<string>(), Arg.Any<int>(), Arg.Any<int>(), Arg.Any<int>(),
                Arg.Any<double>(), Arg.Any<long>(), Arg.Any<double>(), Arg.Any<string>())
            .Returns(new JsonObject());
        _workflows.LtxImage2Video(Arg.Any<string>(), Arg.Any<string>(), Arg.Any<string>(),
                Arg.Any<string>(), Arg.Any<int>(), Arg.Any<int>(), Arg.Any<int>(), Arg.Any<int>(),
                Arg.Any<double>(), Arg.Any<long>(), Arg.Any<int>(), Arg.Any<double>(),
                Arg.Any<IReadOnlyList<(string, double)>>(), Arg.Any<string>())
            .Returns(new JsonObject());
        _comfy.WaitForCompletionAsync(Arg.Any<string>(), Arg.Any<int?>(), Arg.Any<CancellationToken>())
            .Returns(new JsonObject());
        _comfy.CollectOutputAsync(Arg.Any<JsonObject>(), Arg.Any<string>(), Arg.Any<CancellationToken>())
            .Returns(ci => ci.Arg<string>());
    }

    private PipelineOrchestrator Orchestrator() => new(
        _repo, _comfy, _workflows, _notifier,
        Options.Create(new AiDirectorOptions()),
        NullLogger<PipelineOrchestrator>.Instance);

    [Fact]
    public async Task Scene_that_always_fails_stops_at_its_retry_budget()
    {
        _comfy.SubmitAsync(Arg.Any<JsonObject>(), Arg.Any<CancellationToken>())
            .ThrowsAsync(new InvalidOperationException("ComfyUI rejected workflow"));

        await Orchestrator().RunAsync(_project.Id);

        var scene = _project.Scenes.Single();
        scene.Status.Should().Be(SceneStatus.Failed);
        scene.RetryCount.Should().Be(3, "the max_retries budget must be honored, not one-and-done");
        _project.Status.Should().Be(ProjectStatus.Failed);
        // 3 attempts = 3 still submissions (the first submit of each attempt throws).
        await _comfy.Received(3).SubmitAsync(Arg.Any<JsonObject>(), Arg.Any<CancellationToken>());
    }

    [Fact]
    public async Task Scene_that_fails_once_then_succeeds_ends_generated()
    {
        var calls = 0;
        _comfy.SubmitAsync(Arg.Any<JsonObject>(), Arg.Any<CancellationToken>())
            .Returns(_ => ++calls == 1
                ? throw new InvalidOperationException("transient VRAM error")
                : Task.FromResult($"prompt-{calls}"));

        await Orchestrator().RunAsync(_project.Id);

        _project.Scenes.Single().Status.Should().Be(SceneStatus.Generated);
        _project.Status.Should().Be(ProjectStatus.Generated);
    }

    [Fact]
    public async Task Interrupted_run_rolls_back_to_checkpoint_instead_of_ghosting()
    {
        _comfy.SubmitAsync(Arg.Any<JsonObject>(), Arg.Any<CancellationToken>())
            .ThrowsAsync(new OperationCanceledException("app shutting down"));

        var act = () => Orchestrator().RunAsync(_project.Id);
        await act.Should().ThrowAsync<OperationCanceledException>();

        // The ghost bug: scene stuck Generating + project stuck Generating after a
        // killed run, needing manual surgery before resume worked.
        _project.Scenes.Single().Status.Should().Be(SceneStatus.Pending);
        _project.Status.Should().Be(ProjectStatus.Approved);
    }
}
