# AI Director — Improvement Plan
_Created 2026-07-05. Goal: "describe ideas → wake up to finished, SEO'd, uploadable videos" with zero babysitting._

Priorities are ordered by value toward that goal on the current hardware
(RTX 5070 Ti 16GB, 64GB RAM, i5-14400F — pipeline is already tuned to this; nothing here needs new hardware).

---

## Tier 1 — Reliability for unattended overnight runs (do first)

The pipeline works, but one hang at 1am currently kills the whole night. These make it survive alone.

### 1.1 Overnight batch runner with a watchdog
- New `overnight_runner.py`: reads a queue of jobs (song configs / video ideas from a YAML or `jobs/` folder), runs them sequentially via the existing `run_pipeline()` / `run_full_auto`.
- **Watchdog:** per-phase hard timeouts; if ComfyUI stops responding or a job exceeds its budget → cancel ComfyUI queue, `POST /free`, restart the ComfyUI process, resume from checkpoint (checkpointing already exists — this is the missing half).
- On startup: clear any zombie ComfyUI queue entries (known issue: zombie jobs survive client kills and spill VRAM).
- End-of-night `report.md`: per-job status, timings, output paths, failures.
- Windows Task Scheduler entry (or `schtasks` script) to launch it nightly at a set hour.

### 1.2 Make script/prompt generation crash-proof
- The intermittent Qwen `'NoneType' object has no attribute 'n_tokens'` crash (STATUS.md known bug) is fatal for unattended runs. Add retry-once-then-fallback: LLM prompts → template prompts (`phase_prompts_template` already exists as fallback in song_to_video.py — wire the same pattern into director.py).
- Consider a smaller prompt LLM (7–14B) for overnight runs: removes the 27B↔ComfyUI VRAM contention entirely, which is the likely root cause of the crash.

### 1.3 Port/process hygiene
- Known gotcha: force-killed servers leave orphaned LISTEN sockets on :8000. Runner should detect and pick a free port; add a `stop_all.bat` that cleanly kills app + ComfyUI.

---

## Tier 2 — Output quality (the difference between "generated" and "watchable")

### 2.1 Real lyric↔video sync (biggest quality win for song videos)
- `lyrics_parser.py` currently *estimates* segment timing by evenly distributing lines. Chorus visuals can drift seconds off the actual vocals.
- Add WhisperX (or faster-whisper) forced alignment on the generated song → word/line timestamps → cut scenes exactly on vocal phrases and section changes. CPU-friendly, runs while GPU is busy.

### 2.2 Automatic clip QA + retry
- Right now a broken clip (black frames, frozen motion, deformed character) goes straight into the final video.
- Cheap heuristics first: detect near-black frames, near-zero inter-frame motion, extreme blur (ffmpeg/PIL, no GPU). Fail → regenerate with a new seed (retry cap 2).
- Later: Qwen-VL scoring pass (already in PLAN.md Phase F) for aesthetic/consistency checks.

### 2.3 Character consistency via a character LoRA
- Locked seeds + character-bible prompt repetition gets ~80% consistency; a trained SDXL LoRA of each channel's hero character gets the rest. `train_lora.py` and `gen_training_set.py` already exist — productionize: generate a training set from the best stills, train per-channel LoRA, add to channel YAML `default_loras`.

### 2.4 Faster clips option: Wan 2.2 5B tier per channel
- Wan 2.2 ti2v 5B stays VRAM-resident → ~73s/clip with no model reloads, vs LTX-22B's 2–4 min/scene reload tax. Add `quality_tier: fast|best` to channel YAML so high-volume channels use Wan and flagship videos use LTX.
- Prerequisite cleanup: delete the duplicate/wrong `build_wan_workflow` in `comfyui_client.py` (known bug), keep the validated `build_wan22_ti2v_workflow`.

---

## Tier 3 — Close the loop to YouTube

### 3.1 SEO metadata alongside every render
- The director already emits `description`, `tags`, `hashtags` for story videos — the song pipeline doesn't. Emit `metadata.json` (title, description, tags, hashtags, made_for_kids flag) next to every `final_render.mp4`.

### 3.2 Thumbnail generation
- Pick the best still (or render one dedicated SDXL image at 1280×720 with a thumbnail-specific prompt: big character face, high contrast) + optional title text overlay via PIL. Thumbnails drive CTR more than video quality does.

### 3.3 YouTube upload (Phase 7)
- `youtube_upload.py` scaffolding exists; needs OAuth credentials + upload call + `made_for_kids` flag + scheduled publish time. Start with "upload as private/scheduled" so nothing goes live unreviewed.

---

## Tier 4 — Maintainability (cheap, do opportunistically)

### 4.1 Repo cleanup
- ~60 one-off scripts, logs, PNGs, and 3 SQLite DBs (`ai_director.db`, `app.db`, `director.db`) sit in the repo root. Move keepers to `scripts/`, archive experiments, gitignore `*.log`/`*.png`/`*.db`, delete the stale DBs.

### 4.2 Job configs out of code
- Song definitions live as Python literals inside `song_to_video.py`. Move them to `jobs/*.yaml` so adding tonight's videos never means editing pipeline code (also what the Tier 1 runner consumes).

### 4.3 Test suite for the song pipeline
- `test_smoke.py` covers the story pipeline; add an equivalent for song_to_video (lyrics parser unit tests need no GPU).

---

## Suggested order of attack

| Step | Item | Effort | Payoff |
|------|------|--------|--------|
| 1 | 1.1 Overnight runner + watchdog | ~1 session | Every night becomes productive |
| 2 | 1.2 LLM fallback hardening | small | No dead runs |
| 3 | 2.1 WhisperX lyric sync | ~1 session | Videos feel professionally edited |
| 4 | 3.1 + 3.2 metadata + thumbnails | small | Upload-ready output |
| 5 | 2.2 clip QA heuristics | medium | No broken scenes in finals |
| 6 | 2.3 character LoRA | ~1 session + training time | Channel identity |
| 7 | 3.3 YouTube upload (private first) | medium | Full loop closed |
| 8 | Tier 4 cleanup | ongoing | Sanity |
