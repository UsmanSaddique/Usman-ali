# AI Director — Status & Resume Guide
_Last verified: 2026-06-20_

## Product direction (the goal)
A **multi-channel local video factory**: define channels modeled on real successful
ones, pick a channel, and the AI autonomously produces one full video (script → title →
scenes → narration → music → clips → assembly) in that channel's style. One video per
channel per day / overnight. Everything local on the RTX 5070 Ti (16GB).

Target channels being cloned (Urdu kids / animated moral & Islamic stories):
Neela Tota, Kidzone, Little Muslim Nation (LMN Urdu). Niche: **Urdu/English, kids,
animated moral storytelling with talking-animal characters.**

This is the **single source of truth** for "where are we / what works / what to do next".
Written so work can resume in a fresh session without re-discovering anything.

---

## TL;DR — the core pipeline WORKS

On 2026-06-20 a real video clip was generated **end-to-end** through the actual
ComfyUI backend in 44 seconds. The "does it actually generate video?" question is
answered: **yes.**

```
[OK] 768x512, 49 frames, 2.04s, valid H.264 mp4 — assets_generated/smoke_clip.mp4
```

---

## How to run

The app runs on **ComfyUI's embedded Python** (it has torch/llama-cpp/fastapi),
NOT the local `venv` (which lacks fastapi).

```bat
run_server.bat        :: starts the web server on http://localhost:8000
```

Prerequisite: **ComfyUI must be running** at `127.0.0.1:8188` for video generation.
(Start ComfyUI separately — its portable launcher.)

Interpreter path used everywhere:
`C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\python_embeded\python.exe`

---

## Tests (run these to confirm health)

| File | What it proves | Needs ComfyUI? |
|------|----------------|----------------|
| `test_smoke.py` | imports, config, routes, ken_burns + real ffmpeg render | no |
| `test_workflow_validate.py` | LTX/Wan workflow JSON matches live ComfyUI nodes + model files | yes (read-only) |
| `test_generate_one.py` | one real short clip end-to-end | yes (GPU ~45s) |
| `test_director.py [slug] [title]` | real script gen — checks lang split, char consistency, scene mix | needs Qwen (GPU) |
| `seed_channels.py` | upsert all `channels/*.yaml` into the DB (run after adding a profile) | no |

```bash
PY="C:\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable\python_embeded\python.exe"
"$PY" test_smoke.py             # 7/7 passing as of 2026-06-20
"$PY" test_workflow_validate.py # workflow valid: all 9 nodes + files present
"$PY" test_generate_one.py      # generated a real clip
```

---

## Bugs fixed this session (2026-06-20)

1. **`VideoGenService.ken_burns()` was missing** — `pipeline.py` calls it for
   STILL_PAN scenes; a linter had deleted it. Restored (with the `self` param the
   old version was missing → it would have crashed even if present). CPU/ffmpeg
   only. `app/services/video_gen.py`.
2. **Stale `config.video.ltx_python` refs** in `model_manager.py` (legacy direct-LTX
   loader) — the field no longer exists; would `AttributeError` if that loader ran.
   Guarded with `getattr` so the obsolete path fails gracefully; video now goes
   through ComfyUI anyway. `app/services/model_manager.py`.
3. **ffmpeg not on PATH** — config hardcoded `"ffmpeg"`. Now auto-resolves:
   PATH → bundled `imageio_ffmpeg` (ships in ComfyUI's python) → `"ffmpeg"`.
   Without this, assembly (Phase 6) and Ken Burns silently failed.
   `app/config.py` (`_resolve_ffmpeg`).
4. **`init_db()` default pointed at `D:/ai-director/ai_director.db`** — a drive that
   may not exist. Any `get_session()` before the server's explicit init hit the wrong
   DB. Now defaults to `settings.paths.database`. `app/database.py`.

## Known bug NOT yet fixed (flagged for later)

- **`build_wan_workflow` is defined twice** in `app/services/comfyui_client.py`
  (lines ~293 and ~443). Python keeps the second; it uses different node types
  (`DualCLIPLoader`/`EmptyWanVLatentVideo` with t5xxl/clip_l) that are likely wrong
  for Wan 2.2. The LTX path is unaffected and is the default, so this is low priority
  until you actually want to use Wan. Fix: delete the duplicate, keep/validate the
  correct one with `test_workflow_validate.py` after setting a Wan model as default.
- **Intermittent LLM crash `'NoneType' object has no attribute 'n_tokens'`** during
  script generation — the Qwen loader sometimes returns None (one project failed with
  this; another scripted 12 scenes fine). Investigate `model_manager` LLM loader /
  llama-cpp init for `Qwen3.6-27B-Q3_K_S.gguf`. Likely VRAM/offload or a load race.

---

## Channel-cloning system (built 2026-06-20)

The "director brain" was upgraded to elite level and is **channel + language aware**:
- `app/services/director.py` — `BASE_SYSTEM_PROMPT` / `YT_KNOWLEDGE_PROMPT` rewritten to think
  like a top-0.1% YouTube strategist + kids story writer + AI art director, with hard local-AI
  realism (talking-animal characters, English visual prompts, repetition-based character
  consistency, VRAM-aware scene-type strategy, retention/SEO engineering).
- `build_system_prompt()` injects a **Language Directive**: narration + title in the channel's
  language (e.g. Urdu script), visual prompts always English.
- Channel profiles (rich YAML, drive everything): `channels/urdu-moral-stories.yaml`,
  `channels/little-muslim-nation.yaml` (+ existing `little-fairy-dreams.yaml`). Schema includes
  language, character_bible, art_style_phrase, still_ratio, narration, music, title_formulas,
  seo_themes, and a `generation:` block with 16GB VRAM presets.
- `seed_channels.py` registers every profile into the DB.

**Verified 2026-06-20:** generated an 8-scene Urdu script for "greedy parrot who learned to
share" → Urdu title w/ English search tail, 75% still_pan, exact character descriptor repeated
every scene, English prompts + Urdu narration. See `last_script.json` for the sample output.
To add a new cloned channel: write a `channels/<slug>.yaml`, run `seed_channels.py`, done.

## Session 2026-06-20 (part 2): speed, language, SEO, QA

**LLM speed fixed (5-6 min → ~40s).** Root cause was CPU-spill: the 27B-Q3 (~11.7GB)
plus a huge `n_batch=4096` compute buffer overflowed 16GB. Fixes in `config.py`/`model_manager.py`:
`n_batch=512`, `n_threads=8`, `flash_attn`, full offload, and **ComfyUI `/free` is called
before loading the LLM** (`comfyui_client.free_vram()` from `director.generate_script`) so the
LLM gets the whole card. Verified: all 64 layers on CUDA0, ~40s.

**Script JSON bug fixed.** The model sometimes returned a single scene object instead of the
`{title, scenes:[...]}` wrapper → 0 scenes. Fixed by (a) switching script gen to non-streaming
(the streaming accumulator dropped this hybrid-SSM model's chunks), (b) an explicit output-shape
demand in the user message, (c) parser resilience (`_parse_script_response` wraps a bare scene/list).
Verified: clean 5-scene scripts.

**Language = Roman Urdu + English (multi-lang).** `language: "Roman Urdu"` in the Urdu profiles →
narration/title/tags in Latin-script Urdu ("Aik dafa ka zikr hai..."), visual prompts stay English.
Set a profile's `language:` to `English` (or anything) to switch. Director honors it via the
Language Directive in `build_system_prompt`.

**SEO added.** `VideoScript` now has `description`, `tags`, `hashtags`; the director emits a hooky
description + 15-25 mixed-language tags + hashtags. Verified: 20 tags, 363-char description.

**Custom audio / BYO song.** `render(project_id, narration_path=, music_path=)` already muxes any
audio file — pass a path to attach your own song; or let Phase-5 generate music.

### ⚠️ Two environmental blockers found during QA (NOT code bugs)
1. **No SDXL image model installed** (`models/checkpoints/` has no SD/SDXL checkpoint). So every
   `still_pan`/`img2vid` scene falls back to LTX video — no Ken Burns, slower, and worse for
   character close-ups. **Action: install an SDXL checkpoint** (e.g. SDXL base or Juggernaut XL)
   and point `config.image.path` at it. This is the biggest quality unlock.
2. **Single-GPU VRAM contention.** Loading the 27B LLM (~13GB) while ComfyUI is resident can crash
   ComfyUI on 16GB. Mitigated by `free_vram()` + LLM unload, and `video_gen` now waits up to 60s
   for ComfyUI (`wait_ready`) instead of failing instantly. For overnight runs keep nothing else
   on the GPU. A smaller scripting LLM (7-14B) would remove the contention entirely.
   ComfyUI must be launched with `PYTHONIOENCODING=utf-8` or it crashes on an emoji in its logs.

### LTX clip limits on 16GB (relevant to "40s / 5 clips / 720p")
LTX 22B at 1280x720 uses ~15.8GB and **spills VRAM → crawls (8+ min/clip)**. Use a VRAM-safe base
of **768x512** (~39s/clip, verified) and let the upscaler take it to 1080p. For a full 40s you
need more short clips (≈8), or SDXL stills + Ken Burns (any length, cheap) — another reason to
install SDXL.

### ✅ FULL E2E PRODUCED A REAL VIDEO (2026-06-20)
`resume_e2e.py` + `finish_e2e.py` on the scripted project ran the whole chain:
script → 5 LTX clips @768x512 (195s) → upscale to 1080p → assemble with a custom song.
**Result: valid 1920x1080, 22.7s, H.264 + audio, 11.3MB.** (22.7s because 5 clips ≈4.5s each;
add clips or SDXL Ken Burns for 40s.)

Upscaler fix: Real-ESRGAN weights aren't installed, so `upscaler.py` now falls back to an
**ffmpeg lanczos upscale** to reach 1080p (and `_get_video_fps`/`_video_has_audio`/`_get_duration`
fall back to parsing `ffmpeg -i` since **ffprobe isn't installed** either — only ffmpeg is).
Installing Real-ESRGAN weights + an SDXL checkpoint are the two upgrades for max quality.

Test files: `test_e2e.py` (full, needs LLM+ComfyUI), `resume_e2e.py <id>` (clips→render, skips
script), `finish_e2e.py <id>` (upscale→render on existing clips).

## Session 2026-06-20 (part 3): quality + resolution tuning

**Consistency root cause + fix.** Clips looked totally different scene-to-scene because (a) the
pipeline used a RANDOM seed per clip and (b) there's no image model, so every scene is an
independent txt2vid generation. Fix: `pipeline._generate_scene` now locks ONE seed per project
(`md5(project.id)`) across all clips → shared palette/lighting/character look. **True character
consistency still requires an image model** (SDXL stills with locked seed + Ken Burns, or img2vid
from one reference, or a character LoRA). Installing SDXL remains the #1 quality unlock.

**Resolution measured + raised.** LTX `FAMILY_DEFAULTS` now **1152x640 @ 97 frames (~4s)** —
measured peak ~15.8GB, ~85s/clip, no VRAM spill (720p spills and crawls 8+ min). Upscales cleanly
to 1080p. Budget for a 5-min video: ~75 clips × 85s ≈ 1.8h gen + overhead ≈ ~2.5h (within 3.5h goal).

**Audio reality.** The "beep" in the QA video was a placeholder sine wave (`sine=...`) I synthesized
because no TTS/music model is installed. Real audio needs: TTS narration (WanGP server on :5000),
music (ACE-Step — not installed), or attach your own song file via `render(music_path=...)`.

## Audio system (2026-06-20 pt4)

Three audio sources, resolved in `pipeline._find_user_audio` + `generate_music`:
1. **Your own song/voiceover (works now)** — drop `music.*` / `voice.*` into
   `projects/<id>/audio_in/`, OR set `music_file:`/`voice_file:` in the channel YAML
   (file in `assets_generated/music/`). User files always win over generation.
2. **Generated music — ✅ VERIFIED.** ComfyUI ACE-Step (`build_acestep_workflow`,
   `MusicGenService._generate_comfyui`). Model `ace_step_v1_3.5b.safetensors` (7.17GB,
   1903 tensors) in ComfyUI `models/checkpoints/`. Generated a 30s vocal song (44.1kHz
   stereo) in ~14s at 60 steps. Does instrumental (instrumental=True) AND vocal songs
   with lyrics (instrumental=False, pass `lyrics`), multilingual (English/Urdu/Hindi).
   NOTE: first download was a truncated/corrupt partial (4.9 of 7.17GB) → "shape invalid"
   crash; re-download WITHOUT `curl -C -` fixed it. A hung curl can lock the file (kill it
   to delete).
3. **Voiceover (TTS) — ✅ VERIFIED, local.** Rewrote `tts.py` to use Meta MMS-TTS via
   transformers VitsModel (no server, no API key). English (`mms-tts-eng`) verified; Urdu
   has NO standalone MMS voice (HF 401) so "urdu"→`mms-tts-hin` (Hindi=Hindustani, sounds
   like spoken Urdu, needs Devanagari input). Samples in `assets_generated/tts_samples/`.
   Channels switched to English narration; `language:` in YAML controls it.

Two model downloads in progress: SDXL (`sd_xl_base_1.0.safetensors`, visual consistency)
and ACE-Step (music). `config.image.path` now points at the SDXL file.

## Pipeline status by phase

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Script (Qwen) | ✅ **elite + verified** | channel/language-aware; intermittent `n_tokens` load crash still possible |
| 2. TTS (voiceover) | ⛔ needs WanGP server on :5000 | non-fatal: pipeline skips if down |
| 3. Video gen (ComfyUI) | ✅ **verified end-to-end** | 5-clip @768x512 run; 720p OOMs on 16GB |
| 3. Ken Burns (stills) | ⚠️ needs SDXL | code works; no SDXL installed → falls back to LTX |
| 4. Upscale → 1080p | ✅ **verified** | ffmpeg-lanczos fallback (no ESRGAN weights installed) |
| 5. Music (ACE-Step) | ⛔ needs server | BYO-song path verified; generated-music needs server |
| 6. Assembly (ffmpeg) | ✅ **verified** | produced 1080p 22.7s video w/ custom audio |
| 7. YouTube upload | ⛔ needs credentials | SEO metadata code exists |

---

## Next steps toward the overnight-automation goal

Ordered by value. The dream: describe ideas → wake up to finished, SEO'd videos.

1. **Full multi-scene render of the existing scripted project.** There is already a
   project SCRIPTED with 12 PENDING scenes ("little cute girl visiting dinosaurs").
   Run `start_generation` → `start_upscale` → `render` on it to exercise phases 4 & 6
   on real data. This is the next concrete milestone.
2. **Fix the LLM `n_tokens` crash** so script generation is reliable for unattended runs.
3. **Validate the upscale step** (Phase 4) on `smoke_clip.mp4`.
4. **Overnight batch runner**: a script/endpoint that takes a list of ideas, and for
   each calls `PipelineOrchestrator.run_full_auto` sequentially overnight. `run_full_auto`
   already exists and skips completed phases — wrap it in a queue + idea generator.
5. **SEO metadata**: have the director LLM emit title/description/tags per video
   (hook into Phase 7). Code scaffolding exists in `youtube_upload.py`.
6. **Optional servers** for full automation: WanGP TTS (:5000) for narration,
   ACE-Step for music. Pipeline already degrades gracefully without them.

---

## Environment facts (don't re-derive)

- GPU: RTX 5070 Ti 16GB; 64GB RAM. Single GPU → one model in VRAM at a time.
- Models live under `C:\ComfyUI_windows_portable_nvidia_cu126\...\ComfyUI\models`.
- Default video model: `LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf` (8 steps, cfg 1.0).
- DB: `ai_director.db` in project root.
- ComfyUI output dir (clips copied from): `...\ComfyUI\output`.
