# Channel Archetypes — Design Plan & Build Checklist

> Goal: make the pipeline handle every channel/niche type from the strategy brief
> (Tier 1 full-auto, Tier 2 human-review, Tier 3 do-not-automate) through **one
> configurable layer** instead of hardcoding per-channel behavior.
>
> Status: **DESIGN ONLY — nothing below is built yet.** Work top-to-bottom through
> the checklist. Each phase is independently shippable and leaves existing channels
> working unchanged.

---

## 0. Why this exists (read first)

The strategy brief sorts YouTube niches into three tiers by how safe they are to
automate locally. Mapping that onto this codebase, **~80% already exists** — the
missing piece is an explicit object that says, per niche: *which audio lane, which
visual mode, where the content comes from, and how strict the human gate is.*

Today that mapping is implicit inside each `channels/*.yaml`. We make it explicit
and reusable as a **Content Archetype**.

### What already exists (build ON this, do not rebuild)

| Capability | Where | Notes |
|---|---|---|
| Channel brand config | `app/database.py:82` (`Channel`) + `channels/*.yaml` | voice, look, SEO, music brief, gen presets, `made_for_kids` |
| Song lane (lyrics → scenes) | `app/services/pipeline.py:520` (`generate_song_*`) + `app/services/lyric_scenes.py` | `project_type="song"` |
| Narration lane (TTS → beats → scenes) | `app/services/pipeline.py:333` + `:417` + `app/services/narration_scenes.py` | `project_type="narration"`, Kokoro TTS |
| Two video engines | `Project.video_engine` = `clips` \| `ltx_director` | `app/services/ltx_director.py`, `app/services/director.py` |
| YT safety gate w/ verdicts | `app/database.py:250` (`SafetyReport`), `app/services/yt_safety.py` | pass/revise/block/**override**; already gates start-generation |
| Music / song gen | `app/services/music_gen.py` | ACE-Step / HeartMuLa |
| Assembler (audio-len → clip timing) | `app/services/assembler.py` | already lays clips to timestamps + bg track |
| Config singleton | `app/config.py` (`Settings`) | pydantic, env-prefixed `AIDIR_` |

### The gap

Nothing maps **niche → pipeline wiring + gate policy**. That is the archetype.

---

## 1. The Content Archetype concept

An **archetype** is a reusable recipe (a YAML file) that wires the pipeline for a
class of niches. A **channel** picks an archetype and overrides only brand-specific
bits. A **project** may override the channel's archetype.

Resolution order (most specific wins): `Project.content_archetype` → `Channel.content_archetype` → archetype defaults.

### Archetype schema (`archetypes/<id>.yaml`)

```yaml
id: ai_dreamscape                # unique slug, == filename
tier: 1                          # 1 | 2 | 3  (drives HITL defaults, see §3)
label: "AI Dreamscape / Satisfying surreal"
enabled: true                    # false => start-generation refuses (Tier 3)

audio_mode: ambient              # song | narration | ambient
                                 #   song      = lyrics drive vocals + scene timing (existing)
                                 #   narration = Kokoro TTS drives beats (existing)
                                 #   ambient   = music/soundscape bed, NO voice (NEW, phase 3)
video_engine: ltx_director       # clips | ltx_director  (existing values)
visual_mode: surreal_loop        # character_panels | surreal_loop | voice_over_bg | reddit_broll
character_consistency: false     # true => enforce Character-Bible descriptor repetition
scene_planner: freeform          # lyric_scenes | narration_scenes | freeform | qa_pairs
source: llm                      # llm | scrape | mixed

hitl:                            # human-in-the-loop policy (defaults from tier, see §3)
  script_review: optional        # optional | required | blocked
  safety_gate: standard          # standard | strict | blocked

seo_profile: ambient             # named SEO/title template set
```

### Data-model changes (additive, via existing `_migrate()` in `app/database.py:320`)

- `Channel.content_archetype` — `VARCHAR`, default `NULL` (falls back to legacy song/narration behavior).
- `Project.content_archetype` — `VARCHAR`, default `NULL` (inherits channel).

No destructive migration. Existing rows with `NULL` map to the `kids_poem` /
legacy behavior so nothing breaks.

---

## 2. Full archetype catalog (ALL niches)

Build the YAML for each. ✅ = fully covered by existing code once routed; 🔧 = needs
the new component named in the last column.

| Archetype id | Tier | audio_mode | video_engine | visual_mode | source | Reuses | New work |
|---|---|---|---|---|---|---|---|
| `kids_poem` | 1 | song | clips | character_panels | llm | song lane ✅ | none (baby-pooem already is this) |
| `ai_dreamscape` | 1 | ambient | ltx_director | surreal_loop | llm | ltx_director | 🔧 ambient audio (phase 3) |
| `funny_ai_qa` | 1 | narration | clips | surreal_loop | llm | narration lane | 🔧 qa_pairs planner (phase 5) |
| `reddit_story` | 1 | narration | clips | reddit_broll | scrape | narration TTS + assembler | 🔧 scraper + bg-loop (phase 4) |
| `edu_facts` | 2 | narration | clips | voice_over_bg | llm | narration lane | 🔧 required-review gate (phase 2) |
| `faith_kids` | 2 | song/narration | clips | character_panels | llm | song/narration | strict gate + IP block list (phase 2) |
| *(Tier 3 traps)* | 3 | — | — | — | — | — | `enabled: false` → refuse (phase 2) |

**Tier 3 = do NOT automate** (satisfying cats, luxury cars, sports legends, WWE/UFC/F1,
ASMR candy/toys, dancing babies). Do not create per-niche YAMLs that run. Instead a
single guard: any archetype with `enabled: false` or `tier: 3` → start-generation
refuses with an explanatory message (real physics / copyright / uncanny-valley).

---

## 3. Tier → HITL enforcement (mechanical version of the brief)

Tier sets the **default** `hitl` block; an archetype may override.

| Tier | `script_review` | `safety_gate` | Runtime behavior |
|---|---|---|---|
| 1 | optional | standard | Wizard can run end-to-end unattended |
| 2 | required | strict | Pipeline **hard-stops** after script; needs explicit human approve before audio/video. Safety gate runs the strict/LLM critic |
| 3 | blocked | blocked | `enabled:false`; start-generation refuses |

Wire this into the **existing** safety-gate refusal point at start-generation
(`app/services/yt_safety.py` + wherever start-generation checks the latest
`SafetyReport`). Tier 2/3 become gate *policies*, not new code paths.

⚠️ Known constraint (from memory `project_safety_gate_llm_crash`): the safety-gate
Qwen critic hard-crashes the app server on the 2nd `llama_cpp` load per process.
Strict-gate (Tier 2) runs must start on a **fresh server**. Note this in the gate code.

---

## 4. Pipeline routing change

Today the pipeline branches on `Project.project_type`. Generalize:

1. Add `app/services/archetypes.py`:
   - `load_archetypes()` — read `archetypes/*.yaml` into a dict (mirror channel-YAML loader).
   - `resolve(project) -> ResolvedRecipe` — merge project → channel → archetype defaults,
     returns `(project_type, video_engine, visual_mode, audio_mode, character_consistency, scene_planner, source, hitl, seo_profile)`.
2. At pipeline entry, call `resolve()` once. Keep `project_type` (`song`/`narration`)
   as the internal lane — the archetype just *chooses* it. Existing song/narration
   implementations are the two backends; most archetypes reuse them.
3. `audio_mode: ambient`, `visual_mode: reddit_broll/voice_over_bg`, and
   `source: scrape` are the only branches that need genuinely new handling
   (phases 3–4).

---

## 5. The only THREE genuinely new components

Everything else is config + routing. These are the real code:

1. **Ambient audio mode** (`audio_mode: ambient`) — generate a music/soundscape bed,
   skip TTS entirely, let scene timing follow fixed clip lengths or music structure.
   Small branch off `app/services/music_gen.py` + assembler.
2. **bg-loop / reddit_broll visual mode** — assembler variant that lays the narration
   WAV over a looping background clip instead of generating one clip per scene.
   `app/services/assembler.py` already computes audio length + lays clips to
   timestamps; this is a simpler special case.
3. **Reddit scraper** (`source: scrape`) — fetch top threads → feed
   `app/services/narration_writer.py`. Isolated, optional, only for `reddit_story`.
   (Respect Reddit API terms; no login/credentials automation.)

---

## 6. BUILD CHECKLIST (work in order)

> **STATUS 2026-07-23: ALL PHASES IMPLEMENTED (Python + C#).** Phases 0–6 done and
> verified (Python smoke tests + 27 C# Application tests + 10 Infra tests green).
> Remaining = end-to-end GPU runs of the two new lanes (ambient, reddit_broll),
> which need ComfyUI + models loaded; the code paths are complete and compile-clean.
>
> Phase 1 ✅ resolve() wired into `ensure_safety`, `run_full_auto` router,
> `_project_video_engine`; baby-pooem + gullu mapped to `kids_poem`. C#:
> `ArchetypeResolver` injected into `PipelineOrchestrator.RunAsync`.
> Phase 2 ✅ Tier-3 refusal (`ArchetypeBlocked`), Tier-2 hard-stop
> (`ScriptReviewRequired` + `reviewed` column, set only by `/approve-script`),
> strict gate + IP deny-list scan (py `_scan_ip_denylist`, C# `YtSafetyGate` param).
> Phase 3 ✅ `audio_mode=ambient`: `_run_full_auto_ambient_impl` +
> `_plan_ambient_scenes` + `_ensure_ambient_music_bed` (no TTS).
> Phase 4 ✅ `app/services/reddit_source.py` (public JSON, SFW, no auth) +
> `_ensure_scrape_context` + `_render_bg_loop` (ffmpeg loop) + `reddit-tales` channel.
> Phase 5 ✅ `_ensure_qa_directive` (funny_ai_qa); edu_facts/faith_kids ride Tier-2 gates.
> Phase 6 ✅ `GET /api/archetypes` (py + C#), channel list carries content_archetype,
> wizard shows a tier/gate badge (`wizArchetypeBadge`).

### Phase 0 — Scaffold (no behavior change) ✅ DONE (Python + C#, 2026-07-23)
Python:
- [x] Add `archetypes_dir` to `PathConfig` in `app/config.py` + `ensure_dirs()`.
- [x] Create `archetypes/` dir with all 7 YAMLs (kids_poem, ai_dreamscape, funny_ai_qa,
      reddit_story, edu_facts, faith_kids, authenticity_trap).
- [x] Add `content_archetype` column to `Channel` and `Project` in `app/database.py`,
      register both in the `_migrate()` list.
- [x] Create `app/services/archetypes.py` with `load_archetypes()` + `resolve()` +
      `ResolvedRecipe` dataclass + tier→HITL defaults + legacy fallback.
- [x] Map baby-pooem.yaml → `content_archetype: kids_poem`.
- [x] Smoke test `scratch_test_archetypes.py` (7 checks) + DB migration verified — ALL PASS.

C# parity (Clean Architecture):
- [x] `PathsOptions.ArchetypesDir` in `AiDirectorOptions.cs`.
- [x] `ContentArchetype` on `Channel` + `Project` entities (auto snake_case → `content_archetype`).
- [x] `Application/Archetypes/ArchetypeModels.cs` (`ContentArchetype`, `HitlPolicy`, `ResolvedRecipe`).
- [x] `Application/Archetypes/ArchetypeResolver.cs` (pure resolve) + `IArchetypeRegistry`.
- [x] `Infrastructure/Archetypes/YamlArchetypeRegistry.cs` (YamlDotNet loader, cached).
- [x] DI registration in `Infrastructure/DependencyInjection.cs`.
- [x] `tests/.../ArchetypeTests.cs` (7 tests) PASS; full Infra suite (10) still green.

**Not yet mapped (deliberately deferred to Phase 1 — mapping is inert until resolve()
is wired into the pipeline, and Tier-2 archetypes would change gate behavior):**
gullu-ka-gaon, little-muslim-nation, urdu-moral-stories, little-fairy-dreams,
kid-ggroups, ai-war, test-automation-channel.

### Phase 1 — Routing
- [ ] Call `resolve()` at pipeline entry; map recipe → existing `project_type` lane.
- [ ] Author the remaining archetype YAMLs from §2 (`ai_dreamscape`, `funny_ai_qa`,
      `reddit_story`, `edu_facts`, `faith_kids`).
- [ ] Point each existing channel YAML at an archetype via `content_archetype:`
      (baby-pooem → `kids_poem`, etc.). Verify resolution precedence.

### Phase 2 — Tier gates (Tier 2 & 3)
- [ ] Derive default `hitl` from `tier` in `resolve()`.
- [ ] `script_review: required` → pipeline hard-stops after script generation, sets a
      status that waits for explicit human approve (reuse the existing approve step /
      `ProjectStatus.APPROVED`).
- [ ] `enabled: false` OR `tier: 3` → start-generation refuses with explanatory message.
- [ ] `safety_gate: strict` → route to LLM critic; add fresh-server note re:
      `project_safety_gate_llm_crash`.
- [ ] `faith_kids`: add an IP/deny-list check (e.g. block "Bheem" and other copyrighted IP).

### Phase 3 — Ambient audio (`ai_dreamscape`) ← first new niche
- [ ] Add `audio_mode: ambient` branch: music/soundscape bed, no TTS, no lyrics vocals.
- [ ] Scene timing: fixed clip lengths or music-structure driven (freeform planner).
- [ ] Wire `video_engine: ltx_director` + `character_consistency: false` (no descriptor lock).
- [ ] End-to-end test one `ai_dreamscape` project → surreal looping video + ambient bed.

### Phase 4 — Reddit stories (`reddit_story`)
- [ ] `source: scrape` fetcher → top threads (respect API terms; no credential automation).
- [ ] Feed thread text to `narration_writer.py`; generate Kokoro narration master.
- [ ] `visual_mode: reddit_broll` assembler: narration WAV over a looping bg clip.
- [ ] End-to-end test.

### Phase 5 — Remaining Tier-1 + Tier-2
- [ ] `funny_ai_qa`: `qa_pairs` planner (LLM Q&A → surreal image prompts), narration lane.
- [ ] `edu_facts`: narration lane + `script_review: required` (confirm hard-stop works).
- [ ] `faith_kids`: song/narration + strict gate (confirm IP block + fresh-server note).

### Phase 6 — UX / wizard exposure
- [ ] Surface archetype pick in the create-project wizard (`frontend/index.html`).
- [ ] Show tier + gate status so the human knows when review is required.
- [ ] Optional: keep the C# port in parity (see memory `project_csharp_frontend_parity` —
      JSON shapes must match exactly).

---

## 7. Guardrails / gotchas for whoever continues

- **`debug`/reload MUST stay False** — auto-reload restarts the server and kills any
  in-flight GPU run on every `.py` save (`app/config.py:247`). Restart manually to
  apply edits.
- **Safety-gate 2nd-load crash** — strict Tier-2 runs need a fresh server process.
- **832x480 is the VRAM-safe clip size** on the 16GB card; 1152x640 spills (memory
  `project_wizard_slow_i2v_rootcause`). Ambient/ltx_director loops must respect this.
- **Additive migrations only** — never rewrite the schema; append to `_migrate()`.
- **C# parity** — if touching JSON shapes, mirror in `csharp/` or it fails silently.
- **No new dependency on real footage / copyrighted IP** — Tier 3 stays refused by
  design; do not "just try it."

---

## 8. Definition of done

- Every Tier-1 archetype runs end-to-end from the wizard with `script_review: optional`.
- Tier-2 archetypes hard-stop for human approval and use the strict gate.
- Tier-3 / `enabled:false` archetypes refuse at start-generation with a clear message.
- Existing channels (baby-pooem, gullu, etc.) run unchanged after mapping to archetypes.
- Adding a NEW niche = writing one `archetypes/*.yaml` (no code), unless it needs a
  genuinely new audio_mode / visual_mode / source.
