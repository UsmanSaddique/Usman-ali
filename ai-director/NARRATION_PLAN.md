# NARRATION-TO-VIDEO — MASTER PLAN

**Goal:** A second production mode alongside song-mode: *narration mode*. Topic in → researched script →
YT-policy-safe narration (best local TTS) → visuals generated **to the narration's timing** → per-clip
sound effects + ambient bed → mixed like a real doc → upscaled → captions/SEO/thumbnail out.

**Target formats (faceless, production-level):**
1. **AI/Tech explainers** — diagrams, motion graphics, clean VO
2. **Software tutorials / coding guides** — code-on-screen visuals, terminal mockups
3. **Documentaries / storytelling** — cinematic b-roll, maps, ambient soundscapes

**Hard constraints (from what's proven on this machine):**
- RTX 5070 Ti 16GB — ONE heavy model in VRAM at a time (ModelManager already enforces this)
- No paid API keys — everything local
- Video clips: 832×480 safe @ ~1min/clip, 121-frame (5.04s) cap, crossfade assembly (already fixed)
- Server: edits to .py require restart (reload=False); resume/checkpointing must keep working

---

## 0. Architecture principle: **narration is the timeline master**

Song mode syncs video to the music. Narration mode inverts nothing — same idea, different master:

```
TOPIC ──► SCRIPT (LLM) ──► YT-SAFETY GATE ──► TTS ──► FORCED ALIGNMENT (word timestamps)
                                                          │
                              ┌───────────────────────────┘
                              ▼
                    BEAT PLANNER (narration segments → visual beats with exact start/end)
                              ▼
        VISUAL GEN (per beat, duration-exact: b-roll / stills+KenBurns / motion graphics)
                              ▼
        SOUND DESIGN (MMAudio per-clip SFX + ambient bed + music bed)
                              ▼
        MIX (duck under VO, −14 LUFS) ──► ASSEMBLE ──► UPSCALE ──► SRT + SEO + THUMB
```

Every downstream artifact carries `(start_time, end_time)` derived from the narration WAV.
Video never dictates timing; it fills the slot exactly (chained 5.04s clips + crossfades, same
mechanism song mode uses today).

---

## 1. Models to download (all local, all fit 16GB one-at-a-time)

| Role | Model | Size | Why |
|---|---|---|---|
| Script writer | **Qwen3.6-27B Q3 (already installed)** | ~11.7GB | Already the director brain; good long-form writer |
| **Narration TTS (primary)** | **Kokoro-82M** | ~330MB | Best quality-per-cost local TTS; near-instant on GPU; multiple natural EN voices (US/UK, m/f); Apache-2.0 |
| Narration TTS (branded voice) | **Chatterbox (Resemble, 500M)** | ~2GB | MIT license; zero-shot voice cloning + emotion-exaggeration knob → a unique, consistent channel voice |
| Long-form TTS (optional, later) | VibeVoice-1.5B | ~5GB | 30+ min single pass, very consistent prosody for documentaries |
| Alignment | **WhisperX (already integrated)** | small | Word timestamps; also reused as QA (transcribe-back check) |
| **SFX (per-clip, synced)** | **MMAudio (video-to-audio)** | ~10GB peak | Watches the finished clip and generates *synchronized* foley/ambience — exactly "each clip has sound effects" |
| SFX one-shots / ambience | Stable Audio Open Small | ~3GB | Text→SFX (whoosh, keyboard, rain, crowd) for transitions and beds |
| Music bed | ACE-Step (already installed) | — | Instrumental-only prompts → royalty-free background bed |
| Visuals | Z-Image-Turbo + LTX-22B + ESRGAN (already installed) | — | Proven pipeline |

> Priority: Kokoro + MMAudio are the two downloads that unlock the feature. Chatterbox second
> (channel identity). VibeVoice/Stable-Audio are phase-2 nice-to-haves.
>
> **Status 2026-07-17:** Kokoro installed + downloaded + smoke-tested (14.6s test narration OK).
> faster-whisper installed (chosen over WhisperX — no torch-version conflict with the ComfyUI
> torch nightly). MMAudio custom node installed, weights downloading. LTX Director model set
> downloaded/present (dev GGUF Q4_K_S already on disk is used; Q4_K_M optional).

**Why not MMS-TTS (current tts.py)?** It's robotic — fine for kids-rhyme scaffolding, not
"production level." Keep it as last-resort fallback; put an engine switch in `TTSConfig`
(`engine: kokoro | chatterbox | mms`).

---

## 2. Pipeline stages (detailed)

### Stage A — Research & Script Engine (`app/services/narration_writer.py`)
- Input: topic + channel profile (new channel-type YAMLs: `explainer`, `tutorial`, `documentary`) + optional user notes/links pasted in.
- Two-pass LLM writing on the existing Qwen director brain:
  1. **Outline pass** — hook, chapters, retention curve (open loop in first 15s, payoff at end),
     target duration → word budget (≈150 wpm ⇒ 8-min video ≈ 1200 words).
  2. **Draft pass** — full narration per chapter with `[BEAT]` markers where the visual should change,
     plus per-beat visual intent (`broll | still | diagram | code | map | title_card`) and SFX intent
     ("keyboard clatter", "server room hum", "distant thunder").
- Output schema (extends existing `ScenePlan`): `chapters[] → beats[] → {narration_text, visual_type, visual_prompt, sfx_prompt, mood}`.
- SEO block (title/description/tags) reuses the existing director SEO machinery.

### Stage B — YT-Safety Gate (`app/services/yt_safety.py`) — runs BEFORE any GPU time is spent
**UNIVERSAL: this gate applies to EVERY project, regardless of type — narration AND song mode (and any
future mode).** It is wired into the shared pipeline as a mandatory phase right after script/lyrics
generation, not into narration mode specifically:
- **Narration projects:** checks the narration script + per-beat visual prompts.
- **Song projects:** checks the lyrics, the scene visual prompts, and the style/producer brief
  (no copyrighted lyric fragments, no interpolated melodies referenced by name, kids-content
  compliance for the kids channels — "made for kids" flag correctness matters for COPPA).
- **All projects:** checks title/description/tags/thumbnail prompt (no misleading claims vs content),
  and emits the AI-disclosure flag into the release-assets bundle.
- A `safety_reports` row is stored per project per run; `start-generation` (and `/full-auto`) refuse
  to spend GPU time until the latest report's verdict is `pass` (override flag available in the wizard
  for manual sign-off, which is itself recorded in the report).

Two layers, produces a stored report + auto-revise loop:
1. **Rule layer (deterministic):** profanity/slur lexicon; medical-misinfo trigger phrases; violence
   descriptors above threshold; no quoted lyrics; no verbatim quotes >90 chars; word-count-vs-duration
   sanity; CTA spam check.
2. **LLM critic layer:** same Qwen model, different system prompt = YouTube policy reviewer scoring
   against: advertiser-friendly guidelines (violence, shocking content, drugs, dangerous acts, sensitive
   events), medical/elections misinformation policies, "reused content" / "repetitious content"
   (originality: script must be transformative, narration must add commentary/analysis — this is the
   #1 reason faceless channels get demonetized), child-safety flags.
   Output: `{verdict: pass|revise|block, issues[], suggested_rewrites[]}` → auto-rewrite flagged
   sentences (max 2 loops) → human sees final diff in the wizard.
3. **Metadata layer:** enforce AI-disclosure flag (YouTube requires disclosure for realistic synthetic
   media — expose a checkbox that flows into the release-assets bundle), no misleading
   title/thumbnail claims vs script content.

### Stage C — Narration Audio (`tts.py` upgrade)
- Engine registry: `kokoro` (default), `chatterbox` (cloned channel voice + emotion control per beat
  mood), `mms` (fallback). Per-channel voice config in the channel YAML.
- Generate **per-beat WAVs** → concat with breath-pauses (pause length by punctuation/chapter break:
  0.3s sentence, 0.8s chapter).
- **WhisperX forced alignment** on the master WAV (already implemented) → word timestamps →
  exact `(start, end)` per beat.
- **Audio QA gate:** transcribe-back with faster-whisper, WER vs script > 8% ⇒ regen that beat
  (catches TTS skips/hallucinated words); loudness normalize VO to −16 LUFS mono.

### Stage D — Beat Planner (`app/services/narration_scenes.py`)
- Maps aligned beats → renderable scenes. Rules:
  - Beat duration = alignment end − start (+ pause share). A 12s beat ⇒ 3 chained 5.04s clips
    (existing crossfade logic), or 1 clip + Ken Burns hold — planner chooses by `visual_type` and
    motion budget.
  - Visual pacing guard: no shot >8s on screen for explainer/tutorial (retention), documentaries
    allow up to 12s slow shots.
  - Reuse budget: B-roll of the same subject may be re-shown once (cheaper), never back-to-back.
- Writes standard `Scene` rows so the ENTIRE existing generation/QA/resume/upscale machinery works
  unchanged. `duration` comes from narration, never from defaults.

### Stage E — Visual Asset Classes (the "not a kids project" part)
| Class | Engine | Status |
|---|---|---|
| Cinematic b-roll | Z-Image still → LTX img2vid (existing) | ✅ have |
| Stills + Ken Burns | existing | ✅ have |
| **Motion graphics / diagrams** | NEW: HTML/CSS/JS templates rendered headless (Playwright + screen-capture at 24fps, or CSS-animation → ffmpeg) — deterministic, razor-sharp text, zero VRAM | build |
| **Code / terminal scenes** | NEW: code-walkthrough template (Monaco/highlight.js typewriter effect, terminal emulator look) rendered same way | build |
| **Title cards / lower thirds / animated maps** | NEW: template pack in same renderer (map pans via Leaflet/static tiles + CSS transform) | build |
| User-supplied assets | NEW: per-project `assets_in/` folder — screen recordings, charts; planner slots them into beats by name | build |
- The template renderer is a huge lever: text in AI video models is garbage, but explainer channels are
  *mostly text/diagram shots*. Headless-browser rendering gives pixel-perfect 4K-native graphics for free,
  runs on CPU **in parallel with GPU video gen** (no VRAM contention).
- Style tokens per channel (colors/fonts/logo) in the channel YAML so every video looks on-brand.

### Stage F — Sound Design (`app/services/sfx_gen.py`)
- **Per-clip SFX:** MMAudio conditioned on the *rendered clip* + the beat's `sfx_prompt` →
  synchronized foley (footsteps land on footfalls, keyboard sounds on typing shots). Runs as its own
  VRAM phase after video gen (ModelManager slot).
- **Ambient beds:** per-chapter ambience (rain, city, server hum) via Stable Audio Open or MMAudio
  on a loop, −30 LUFS under everything.
- **Transition one-shots:** whoosh/riser library — generate ~30 once with Stable Audio, cache in
  `assets_generated/sfx_library/`, reuse forever (consistent channel sound).
- **Music bed:** ACE-Step instrumental prompt from chapter mood; documentary = sparse drones,
  explainer = light percussion.

### Stage G — Mix & Assembly (`assembler.py` extension)
- 4-bus mix in one ffmpeg filtergraph: VO (master) / music / ambience / SFX.
- **Sidechain ducking:** music+ambience duck −7 dB under VO (`sidechaincompress`), 250ms release.
- Master to **−14 LUFS integrated** (YouTube normalization target), true peak −1 dBTP (`loudnorm`).
- Video: existing crossfade concat; add straight-cuts option (docs/explainers cut on beat boundaries;
  crossfades read "slideshow").
- Outputs: master MP4 → existing ESRGAN upscale path (1080p/4K), **SRT from word timestamps**
  (already have `_write_lyrics_srt` pattern — narration version is trivial and boosts YT search),
  chapters file (`00:00 Intro…`) from chapter starts for the description, thumbnail via existing picker
  + optional Z-Image thumbnail prompt from the hook.

### Stage E2 — LTX Director multi-segment engine (long video + NATIVE audio) — ✅ integrated 2026-07-17
An alternative video engine alongside the per-scene clip pipeline, built on the
WhatDreamsCost `LTXDirector` ComfyUI node (already installed as custom nodes
`WhatDreamsCost-ComfyUI` + `LTXDirector-Extender`):
- **What it does:** feed N segments (reference still + rich prompt + optional spoken
  line each) → LTX-2.3-dev generates ONE continuous long video where characters
  actually SPEAK their dialogue (the model generates audio natively via the LTX audio
  VAE), with smooth segment transitions (transition LoRA) and a latent-upscale detail
  pass (IC detailer LoRA).
- **When to use it:** story/documentary videos where continuity + native dialogue beat
  per-clip control. The per-scene pipeline stays best for music videos and QA-gated
  per-clip retries; LTX Director is best for "attach images + prompts, get a film."
- **Models** (all downloaded): `ltx-2-3-22b-dev-Q4_K_M.gguf` (unsloth), official
  `ltx-2.3-22b-distilled-lora-384-1.1` @0.5 (fast-distill), `ltx2.3-transition` @0.7,
  `ltx-2-19b-ic-lora-detailer` @0.4, LTX23 video+audio VAEs, taeltx preview VAE,
  Gemma-3-12B fp4 text encoder, spatial upscaler ×2.
- **Implementation:** `app/services/ltx_director.py` — template
  (`app/knowledge/workflows/ltx_director_multi.json`, 2 chained LTXDirector nodes)
  + a generic UI-graph→API converter (subgraph expansion, Set/Get nodes,
  Anything-Everywhere broadcasts, reroutes, bypass passthrough, widget mapping via
  live `/object_info`; converter validated structurally against the running ComfyUI).
  Segments come from the project's scenes: still + prompt + `narration_text` as the
  spoken line. Endpoints: `POST /api/projects/{id}/ltx-director` (safety-gated,
  model-checked), `GET .../ltx-director/status`. Wizard: step-5 button.
- **Safety:** goes through the same universal YT-safety gate as every other
  generation path.

### Stage H — App integration (native flows, per your preference — no loose scripts)
- `Project.project_type = "song" | "narration"` (DB migration + API field).
- New tables: `narration_beats` (or extend Scene with `narration_start/end`, `visual_type`, `sfx_prompt`),
  `safety_reports`.
- Pipeline phases added to `PipelinePhase`: `SCRIPT → SAFETY → TTS → ALIGN → PLAN → VISUALS → SFX → MIX → UPSCALE` — all journaled in `run_state.json` so restart/resume works identically.
- **Wizard: narration mode (5 steps mirroring song mode):**
  1. Topic + channel + target length → script draft + safety report (inline diff of auto-fixes)
  2. Voice pick + listen to narration; per-beat re-read button
  3. Beat board: timeline of beats with visual-type chips, edit prompts, upload assets
  4. Generate visuals (existing scene grid, batch mode, QA badges) + SFX pass
  5. Mix preview → render → upscale → release assets (SRT, chapters, SEO, AI-disclosure flag)
- Full-auto endpoint (`/full-auto` narration variant) for overnight runs, honoring auto_resume=False.

---

## 3. VRAM sequencing (16GB, one model at a time)

`Qwen 27B (script+safety) → unload → TTS (tiny, coexists) → WhisperX (small) → Z-Image stills batch →
unload → LTX batch (resident, batch mode) → unload → MMAudio SFX batch → unload → ESRGAN upscale`

Motion-graphics rendering is CPU/browser — run it concurrently with the LTX batch to hide its cost.

---

## 4. Build order (each phase ships something usable)

| Phase | Deliverable | Effort |
|---|---|---|
| **P1** | Schema + `project_type` + narration writer + **universal safety gate (wired into BOTH song and narration pipelines as a mandatory pre-GPU phase)** | 2–3 sessions |
| **P2** | Kokoro TTS engine + alignment + audio QA; full narration WAV with word timestamps | 1–2 sessions |
| **P3** | Beat planner → existing video pipeline renders a complete narration video (b-roll + stills only) with VO — **first end-to-end video here** | 2 sessions |
| **P4** | Mix bus (ducking + LUFS) + ACE-Step bed + SRT/chapters/release assets | 1 session |
| **P5** | MMAudio per-clip SFX + SFX library + ambience | 1–2 sessions |
| **P6** | Motion-graphics/code/map template renderer + asset import | 2–3 sessions |
| **P7** | Wizard narration mode UI (5 steps) + full-auto + resume hardening | 2 sessions |
| **P8** | Chatterbox channel voice, VibeVoice long-form, per-channel style packs, benchmarks | ongoing |

**QA gates carried through every phase:** frozen-clip check (existing), WER transcribe-back, safety
verdict stored, duration drift <100ms between narration WAV and final video, LUFS verification on the
master.

---

## 5. Key decisions locked in this plan
1. Narration WAV is the single source of truth for all timing.
2. Kokoro primary TTS, Chatterbox for channel identity, MMS demoted to fallback.
3. MMAudio for clip-synced SFX (video-conditioned — the only local model that truly "watches" the clip).
4. Text/diagram shots use a headless-browser template renderer, NOT diffusion (sharp text, 0 VRAM, parallel).
5. Safety gate is **universal — every generated video in every mode (song AND narration) passes it**,
   before GPU spend, with stored reports + auto-revise, covering advertiser-friendliness,
   originality/reused-content, kids-content/COPPA correctness, and AI-disclosure metadata.
   `start-generation` is blocked until the verdict is `pass` (recorded manual override available).
6. Everything is app-native (DB, phases, wizard, resume) — no one-off scripts.
