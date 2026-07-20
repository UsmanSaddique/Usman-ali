# AI Director — .NET 10 Port

C# / .NET 10 port of the Python FastAPI pipeline in `../app`, following Clean
Architecture. See [`../CSHARP_PORT_PLAN.md`](../CSHARP_PORT_PLAN.md) for the full
plan and QA strategy. The ComfyUI server, workflow JSONs, ffmpeg commands, and
embedded-Python subprocess scripts do **not** move — this port replaces only the
orchestrator.

## Stack (latest stable, as of 2026-07)

- .NET 10 (LTS) / C# 14
- EF Core 10 + SQLite (maps the existing `ai_director.db` unchanged)
- YamlDotNet 18, Serilog 10, Swashbuckle 10, Playwright 1.61
- xUnit + FluentAssertions + NSubstitute
- Central Package Management (`Directory.Packages.props`) — every version pinned
  in one place. SQLite native lib pinned to 2.1.12 (patches GHSA-2m69-gcr7-jv3q).

## Layout

```
src/
  AiDirector.Domain          entities, enums, value objects (no dependencies)
  AiDirector.Application      use cases, ports, pipeline/directing logic
  AiDirector.Infrastructure   EF Core, ComfyUI client, ffmpeg, audio, templates
  AiDirector.WebApi           minimal-API composition root + wwwroot frontend
tests/
  *.Tests                     unit + integration (incl. real-DB compatibility)
```

Dependency rule: `WebApi -> Infrastructure -> Application -> Domain`.

## Build & test

```bash
dotnet build
dotnet test
```

## Progress

- [x] **P1 foundation** — solution skeleton, Clean Architecture layout, CPM,
      latest packages, security-patched.
- [x] **Domain** — all 8 entities, 6 enums, value objects ported from
      `app/database.py`. Enums carry Python VALUE spellings for the JSON/API
      contract.
- [x] **Persistence** — `AiDirectorDbContext` maps the SAME schema (tables,
      snake_case columns). Custom converters store enums as Python UPPERCASE
      names and tolerate the live DB's mixed-case rows; JSON columns round-trip
      to typed collections.
- [x] **DB compatibility proven** — `DatabaseCompatibilityTests` opens a copy of
      the real `ai_director.db` and reads every table through EF Core (5 tests
      green): enum mapping, raw-SQL count match, JSON columns, navigation.

- [x] **ComfyUI client** — `ComfyUiClient` (HttpClient) ports comfyui_client.py:
      submit/history/queue polling, job-disappearance detection, auto-launch,
      VRAM free, output collection.
- [x] **Workflow builder** — `WorkflowBuilder` ports the build_* graph
      constructors (Z-Image, LTX txt2vid/img2vid, ESRGAN upscale, ACE-Step 1.5).
      Structural parity tests green (3).
- [x] **Pipeline** — `PipelineOrchestrator` runs the "clips" engine (still ->
      i2v -> Generation record -> status), fed by a single-tenant GPU queue
      (`PipelineQueue`) drained by the hosted `PipelineRunner`.
- [x] **WebApi** — minimal-API endpoints (projects, channels, system health/gpu/
      engine), `/ws/pipeline/{id}` WebSocket progress, serves the existing
      `frontend/index.html` unchanged. **Runs and reads the live DB** (verified:
      returns real channels + projects).

- [x] **Directing brain** — `LyricsParser` (section markers -> timed segments),
      `ScenePlanner` (segments -> Scene rows), `YtSafetyGate` (rule layer:
      lexicons, quote limits, made-for-kids strictness). Ports of
      lyrics_parser.py / lyric_scenes.py / yt_safety.py.
- [x] **Assembler + QA** — `Assembler` (ffmpeg xfade concat + audio mix +
      loudnorm + mux, filter-graph-as-script for the Windows cmdline cap),
      `QaGates` (freeze/black/duration via ffmpeg detect filters).
- [x] **Music** — `MusicEngine` (ACE-Step 1.5 via ComfyUI).
- [x] **More endpoints** — safety-check / safety-report, scenes-from-lyrics,
      generate-music, render, scene edit. **Verified live**: safety-check on a
      real project returns a verdict and persists a SafetyReport.
- [x] **Phase 0 parity tooling** — `parity/characterize.py` records real ComfyUI
      submissions + ffmpeg argv as golden fixtures; `parity/export_openapi.py`
      exported the 48-route contract to `parity/fixtures/openapi.json`. An
      `ApiContractTests` proves every implemented C# route mirrors a real Python
      route (path params normalized).

25 tests green across the solution.

### Remaining (smaller tail)

- TTS (Kokoro) + SFX (MMAudio) engines for narration mode (narration mode is
  unbuilt in the Python app too).
- The ~35 endpoints beyond the core 12 (script-gen variants, per-scene regen,
  version selection, LoRA registry, automation/produce).
- Capture golden fixtures from a real Python run, then flip the WorkflowBuilder
  / Assembler parity tests from structural to byte-exact.
- GPU smoke: one still + clip + song end-to-end against real ComfyUI.

## Run the app

```bash
cd src/AiDirector.WebApi
dotnet run --urls http://localhost:5080
# http://localhost:5080/  -> frontend ; /swagger -> API ; /api/projects -> live DB
```
