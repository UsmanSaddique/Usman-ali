# AI Director — C# / .NET Port Master Plan

Convert the Python FastAPI pipeline (`app/`) to a .NET 8 solution using Clean
Architecture, producing byte-identical outputs against the same ComfyUI server,
the same workflow JSONs, and the same ffmpeg commands.

**Guiding rule:** the GPU side does not move. ComfyUI, the workflow JSONs in
`app/knowledge/workflows/`, the embedded-Python subprocess scripts (ACE-Step,
Kokoro), and ffmpeg all stay exactly as they are. C# replaces only the
orchestrator. Parity is verifiable: same workflow JSON in → same video out.

---

## 1. Target Solution Layout (Clean Architecture)

```
AiDirector.sln
├── src/
│   ├── AiDirector.Domain            ← no dependencies on anything
│   │   ├── Entities/                Channel, Project, Scene, Generation,
│   │   │                            MusicTrack, RenderJob, SafetyReport, LoRA
│   │   ├── Enums/                   ProjectStatus, SceneType, SceneStatus,
│   │   │                            GenerationStatus, RenderStatus, SafetyVerdict
│   │   ├── ValueObjects/            SceneTiming, Resolution, SeedPolicy,
│   │   │                            NarrationSegment, QaVerdict
│   │   └── Events/                  PipelineStageChanged, SceneGenerated, …
│   │
│   ├── AiDirector.Application       ← depends only on Domain
│   │   ├── Abstractions/            IComfyUiClient, IFfmpegRunner, ITtsEngine,
│   │   │                            IMusicEngine, ISfxEngine, ITemplateRenderer,
│   │   │                            IUnitOfWork + repositories, IProgressNotifier,
│   │   │                            IClock, IFileStore, IGpuMonitor
│   │   ├── Pipeline/                PipelineOrchestrator (state machine),
│   │   │                            stage handlers, checkpoint/resume journal
│   │   ├── Directing/               Director, LtxDirector, LyricScenes,
│   │   │                            NarrationScenes, NarrationWriter, LyricSync,
│   │   │                            LyricsParser
│   │   ├── Safety/                  YtSafety rule engine
│   │   ├── Qa/                      QA gates (freeze/black/duration/audio checks)
│   │   └── UseCases/                one handler per API action (CreateProject,
│   │                                StartGeneration, RenderProject, …)
│   │
│   ├── AiDirector.Infrastructure    ← depends on Application (implements ports)
│   │   ├── Persistence/             EF Core + SQLite (WAL), migrations,
│   │   │                            repository implementations
│   │   ├── ComfyUi/                 ComfyUiClient (HttpClient + WebSocket),
│   │   │                            WorkflowGraph loader/patcher (the JSON
│   │   │                            node-rewiring logic from comfyui_client.py
│   │   │                            + ltx_director.py), subgraph flattener
│   │   ├── Media/                   FfmpegRunner, FfprobeInspector, Assembler,
│   │   │                            Upscaler (ComfyUI Real-ESRGAN route)
│   │   ├── Audio/                   AceStepMusicEngine (subprocess on ComfyUI
│   │   │                            embedded python), KokoroTtsEngine (subprocess),
│   │   │                            MmAudioSfxEngine (ComfyUI workflow)
│   │   ├── Templates/               HeadlessBrowserTemplateRenderer (Playwright
│   │   │                            for .NET replaces the Python headless route)
│   │   ├── Knowledge/               YAML channel/config loaders (YamlDotNet),
│   │   │                            workflow JSON catalog
│   │   └── System/                  GpuMonitor (nvidia-smi), EngineProcessManager
│   │                                (start/health-check ComfyUI headless)
│   │
│   └── AiDirector.WebApi            ← composition root
│       ├── Endpoints/               minimal-API groups mirroring main.py routes
│       ├── Hubs/                    SignalR (or raw WebSocket) for
│       │                            /ws/pipeline/{projectId} progress
│       ├── Contracts/               request/response DTOs (pydantic → records)
│       ├── wwwroot/                 frontend/index.html served as-is
│       └── Program.cs               DI wiring, hosted services, config binding
│
└── tests/
    ├── AiDirector.Domain.Tests
    ├── AiDirector.Application.Tests     (mocked ports — bulk of unit tests)
    ├── AiDirector.Infrastructure.Tests  (integration: SQLite, ffmpeg, ComfyUI)
    └── AiDirector.Parity.Tests          (golden-master vs Python outputs)
```

Dependency rule: `WebApi → Infrastructure → Application → Domain`. Domain and
Application never reference EF Core, HttpClient, or file paths — everything
external goes through the `Abstractions/` interfaces.

### Python → C# module map

| Python (`app/`)                  | C# home |
|----------------------------------|---------|
| `database.py` models/enums       | Domain.Entities / Enums; EF Core config in Infrastructure.Persistence |
| `config.py` (pydantic-settings)  | `appsettings.json` + strongly-typed `Options` classes |
| `main.py` endpoints              | WebApi.Endpoints + Application.UseCases |
| `pipeline.py`                    | Application.Pipeline.PipelineOrchestrator |
| `comfyui_client.py`              | Infrastructure.ComfyUi.ComfyUiClient + WorkflowGraph |
| `ltx_director.py`                | Application.Directing.LtxDirector (graph edits via WorkflowGraph port) |
| `director.py`, `lyric_scenes.py`, `narration_scenes.py`, `narration_writer.py`, `lyric_sync.py`, `lyrics_parser.py` | Application.Directing |
| `assembler.py`, `video_gen.py`, `upscaler.py` | Infrastructure.Media (commands orchestrated from Application) |
| `music_gen.py`, `tts.py`, `sfx_gen.py` | Infrastructure.Audio (subprocess/ComfyUI) |
| `template_renderer.py`           | Infrastructure.Templates (Playwright for .NET) |
| `qa.py`                          | Application.Qa (ffprobe results via IFfmpegRunner) |
| `yt_safety.py`                   | Application.Safety (pure rules — easiest port, fully unit-testable) |
| `model_manager.py`               | Infrastructure.System (VRAM/model lifecycle hints to ComfyUI) |
| `youtube_upload.py`              | Infrastructure (port `IYouTubeUploader`; stub until needed) |
| `frontend/index.html`            | copied unchanged to `wwwroot/` — the API contract is the compatibility line |

### Key technology choices

- **.NET 8 LTS**, minimal APIs (closest shape to FastAPI decorators).
- **EF Core + SQLite** with WAL on; reuse the *same schema and column names*
  so `ai_director.db` keeps working — write an EF model that maps the existing
  tables rather than regenerating them. Port `_migrate()` as idempotent
  startup migrations.
- **System.Text.Json** with `JsonNode` for workflow graph patching (mutable
  DOM — direct analogue of Python dict manipulation on workflow JSON).
- **SignalR or plain `WebSocketMiddleware`** for pipeline progress — keep the
  message payload shape identical so `index.html` needs zero changes.
- **Channels + BackgroundService** replace the Python thread-per-run model;
  a single-writer GPU queue enforces the "GPU is single-tenant" rule
  explicitly instead of by convention.
- **run_state.json journal**: keep the same file format so a run started in
  Python could in principle resume in C# (and it doubles as a parity check).

---

## 2. Conversion Phases

Ordered so every phase ends with something runnable and testable. Estimated
relative effort in parentheses.

### Phase 0 — Freeze & characterize the Python app (S)
- Tag the current Python commit as the **reference implementation**.
- Record golden fixtures: for 2–3 known projects (e.g. Baby Pooem duck song,
  a narration-mode project once built), capture every ComfyUI workflow JSON
  actually submitted, every ffmpeg command line executed, and hashes of
  outputs. A small Python shim that logs `POST /prompt` bodies and
  `subprocess` argv is enough. These fixtures are the contract for Phase P.
- Export the OpenAPI schema from FastAPI (`/openapi.json`) — this becomes the
  API-compatibility spec for the C# endpoints.

### Phase 1 — Skeleton + Domain + Persistence (M)
- Create the solution, projects, DI wiring, appsettings binding of `config.py`.
- Port entities/enums; EF Core mapping against the **existing** SQLite file;
  round-trip test: open the real `ai_director.db`, read all projects/scenes,
  assert counts and field values match a Python-dumped JSON snapshot.
- Health endpoints: `/api/system/health`, `/api/system/gpu-status`.

### Phase 2 — ComfyUI client + workflow graph engine (L, highest risk)
- `ComfyUiClient`: queue prompt, poll history, websocket progress, output
  file collection, timeout/retry behavior copied from `comfyui_client.py`.
- `WorkflowGraph`: load workflow JSON, set node inputs by title/class,
  rewire connections, flatten subgraphs, validate — this is the port of the
  trickiest Python logic. Test it **offline** with the Phase 0 fixtures:
  given the same template + parameters, the serialized graph must be
  deep-equal to what Python submitted.
- Milestone: C# generates one Z-Image still and one LTX clip end-to-end.

### Phase 3 — Directing brain (M)
- Port `director.py`, `ltx_director.py`, `lyric_scenes.py`,
  `narration_scenes.py`, `narration_writer.py`, `lyric_sync.py`,
  `lyrics_parser.py`, `yt_safety.py`.
- These are pure/near-pure logic: highest-value unit tests. Seeded RNG note:
  Python `random` and .NET `Random` differ — where seeds drive prompt/shot
  variety, port the selection logic to a deterministic hash-based choice (or
  embed a tiny xorshift PRNG shared by both) so scene plans are reproducible
  and comparable.

### Phase 4 — Audio engines (M)
- `KokoroTtsEngine`, `AceStepMusicEngine` (subprocess on ComfyUI embedded
  python — argv copied verbatim from `music_gen.py`/`tts.py`),
  `MmAudioSfxEngine`. Preserve the producer-brief formula and style
  passthrough behavior exactly (see ACE-Step memory: full style passthrough,
  cfg 2.5 / temp 0.75 defaults).

### Phase 5 — Assembly, upscale, QA (M)
- `Assembler`: port ffmpeg command construction (concat, crossfade, 5.04 s
  clip cap, audio mux, loudness). Commands must string-match Phase 0 fixtures
  modulo temp paths.
- `Upscaler` (Real-ESRGAN via ComfyUI — never lanczos), QA gates from `qa.py`
  (freeze-frame, black-frame, duration, audio-presence checks).

### Phase 6 — Pipeline orchestrator + API + WebSocket (L)
- Port the state machine in `pipeline.py`: stage ordering, pause/cancel/
  resume, checkpoint journal, retry-with-new-seed policy, startup checkpoint
  recovery (auto-resume OFF per user preference).
- Implement all `main.py` routes against the OpenAPI spec; serve
  `frontend/index.html` unchanged; wire WebSocket progress with identical
  payloads.
- Milestone: full 5-step wizard run (script → scenes → generate → music →
  render) completes from the existing frontend against the C# backend.

### Phase 7 — Automation & ops (S)
- `/api/automation/produce` full-auto flow, engine-start/preflight endpoints,
  ComfyUI headless process management, structured logging (Serilog) mirroring
  current log lines so existing debugging habits transfer.

### Phase P — Parity & cutover (runs alongside 2–7)
- Golden-master tests from Phase 0 fixtures run in CI on every phase.
- Shadow mode: run C# against the same ComfyUI + a copy of the DB, reproduce
  one real Baby Pooem episode; compare artifacts (see QA plan §3.4).
- Cutover: switch the launch script to `dotnet run`; keep the Python app
  in-tree, frozen, for one release cycle as fallback. Then archive it.

**Deliberately out of scope for v1:** rewriting the frontend, YouTube upload
OAuth (stub the port), and any change to models/workflows/prompts. Parity
first; improvements after cutover.

---

## 3. QA Master Plan

Four layers, cheapest first. The overriding idea: **the Python app is the
oracle.** Every ambiguous behavior question is answered by "what does the
reference implementation do," not by judgment.

### 3.1 Unit tests (Application + Domain — no GPU, milliseconds)
- **Directing logic:** scene planning, lyric parsing/sync, narration segment
  math, shot selection — table-driven tests using inputs dumped from the
  Python code paths (dump via a small pytest that serializes in/out pairs to
  JSON; C# tests consume the same JSON). Target ≥90 % coverage here — this is
  where port bugs will live.
- **Safety rules (`yt_safety`):** every rule gets a pass/revise/block case.
- **Workflow graph engine:** template + params → serialized graph deep-equal
  to recorded Python submissions (the single most important test suite in
  the repo).
- **State machine:** every legal and illegal status transition; pause/cancel
  /resume from each stage; checkpoint journal write/replay.
- **ffmpeg command builders:** assert exact argv strings against fixtures.

### 3.2 Integration tests (real SQLite, real ffmpeg, fake ComfyUI)
- **Fake ComfyUI server** (in-process ASP.NET test host): accepts `/prompt`,
  replays recorded histories, drops pre-made tiny mp4/png/wav files into the
  output dir. Lets the whole pipeline run in seconds without a GPU.
- **DB compatibility:** open a copy of the real `ai_director.db`; run
  migrations; assert Python's SQLAlchemy can still read it afterward
  (run the existing `scratch/query_db.py` against the migrated file).
- **API contract:** generate the C# OpenAPI doc and diff it against the
  frozen FastAPI `openapi.json` — routes, methods, and schema field names
  must match. Plus request/response snapshot tests for every endpoint.
- **WebSocket:** connect, drive a fake pipeline run, assert message sequence
  and payload shape match a recorded Python session.
- **Failure injection:** ComfyUI timeout, subprocess non-zero exit, ffmpeg
  failure mid-assembly, process kill mid-run → next start must recover to
  the correct checkpoint state (mirrors the resume-ops playbook).

### 3.3 GPU smoke suite (manual trigger, real ComfyUI, ~15 min)
A C# analogue of `scripts/native_qa_test.py` / `scripts/test_ltx_director.py`:
1. one Z-Image still, 2. one 832×480 LTX clip, 3. one Kokoro TTS line,
4. one ACE-Step 30 s song, 5. one template-rendered card, 6. assemble all of
it into a 20 s mp4, 7. run QA gates on the result. Pass = files exist,
durations correct, QA gates green, VRAM freed afterward. Run at the end of
Phases 2, 4, 5, 6.

### 3.4 Parity / golden-master (the acceptance bar)
For each reference project (song-mode, narration-mode, ltx_director-mode):

| Artifact | Comparison |
|---|---|
| Scene plan / script JSON | semantic diff (field-by-field; ordering-insensitive where Python is nondeterministic) |
| ComfyUI workflow submissions | deep JSON equality (after temp-path normalization) |
| ffmpeg command lines | string equality modulo temp paths |
| TTS/music/SFX WAVs | duration ±50 ms, sample rate, channel count, loudness (LUFS ±0.5) |
| Final MP4 | container metadata equal; duration ±1 frame; per-scene SSIM ≥0.98 vs Python render *with identical seeds*; audio track cross-correlation |
| DB state after run | row-by-row diff of project/scenes/generations (ignoring timestamps/ids) |

Acceptance criteria for cutover:
- All unit + integration suites green in CI (GitHub Actions, no GPU needed).
- GPU smoke suite green on the RTX 5070 Ti.
- One full real episode produced end-to-end by C# through the existing
  frontend, watched by you, judged equal to Python output.
- Kill-and-resume test passes on a real run.
- No VRAM regression: peak usage during I2V within margin of Python run
  (guards the 832×480-safe / 1152×640-spill boundary).

### 3.5 Ongoing QA after cutover
- CI on every commit: unit + integration (fake ComfyUI) suites.
- Nightly (optional, local): GPU smoke suite via scheduled task.
- Keep golden fixtures under `tests/fixtures/` and version them; any
  intentional behavior change must update a fixture in the same PR — that
  makes drift visible in review instead of silent.

---

## 4. Risk Register

| Risk | Mitigation |
|---|---|
| Workflow-graph patching subtly differs (subgraph flattening, node id rewiring) | Phase 2 offline deep-equality tests against recorded Python submissions before any GPU run |
| Python `random` vs .NET `Random` breaks seed reproducibility | deterministic hash-based selection shared by both implementations; seeds recorded in DB as today |
| EF Core migration corrupts the live DB | operate on copies until cutover; backup + WAL checkpoint before first real open; Python read-back test |
| ffmpeg quoting/escaping differences on Windows | build argv as arrays via `ProcessStartInfo.ArgumentList` (never a shell string); fixture string checks |
| Long-running pipeline vs ASP.NET request lifetime | pipeline runs in a hosted BackgroundService with its own scope; endpoints only enqueue — same split `pipeline.py` already has |
| WebSocket payload drift breaks the untouched frontend | recorded-session snapshot tests (§3.2) |
| Hidden Python behavior nobody remembers | Phase 0 characterization logging; when in doubt, the frozen Python app is the oracle |

---

## 5. Suggested Order of Work (checklist)

- [ ] P0: tag reference commit, add submission/argv logging shim, capture fixtures, export openapi.json
- [ ] P1: solution skeleton, Domain, EF Core mapping vs real DB, health endpoints
- [ ] P2: ComfyUiClient + WorkflowGraph + offline equality tests → first GPU clip
- [ ] P3: directing brain + yt_safety + unit-test corpus from Python dumps
- [ ] P4: TTS / music / SFX engines
- [ ] P5: assembler + upscaler + QA gates, ffmpeg fixture tests
- [ ] P6: pipeline orchestrator, all endpoints, WebSocket, frontend served
- [ ] P7: automation/produce, engine management, logging
- [ ] PP: shadow-mode episode, parity report, cutover decision
