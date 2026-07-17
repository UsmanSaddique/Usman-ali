# AI Director — Improvement Plan
_Updated 2026-07-15. Goal: "describe ideas → wake up to finished, SEO'd, uploadable videos" with zero babysitting._

Hardware baseline unchanged: RTX 5070 Ti 16GB, 64GB RAM — the pipeline is tuned to it.

---

## ✅ Done since the 2026-07-05 plan

- 5-step Studio Wizard + song mode + music audition round (10 variants, pick one).
- QA layer: preflight gate, script lint, per-clip motion check + retry, final report.
- Pause (graceful) vs Cancel (hard) on every heavy phase.
- ACE-Step 1.5 XL Turbo primary music engine; HeartMuLa improved.
- Z-Image-Turbo stills (character fidelity), 4x-UltraSharp/anime ESRGAN upscale path.
- **Extreme resumability (2026-07-15)** — see below.

## ✅ Tier 0 — Extreme resumability (DONE 2026-07-15)

The contract is now: **no crash, restart, or power cut ever loses finished work,
and the server heals itself on the next boot.**

1. **Startup recovery → checkpoint, not FAILED** (`app/main.py`
   `_recover_interrupted_projects`): projects stuck in GENERATING / UPSCALING /
   MUSIC / ASSEMBLING are rolled back to their last resumable status
   (APPROVED / GENERATED) with an `[interrupted]` note.
2. **Auto-resume on startup** (`_auto_resume_worker`, config
   `auto_resume: bool = True`, env `AIDIR_AUTO_RESUME=0` to disable): waits up
   to 240s for ComfyUI (auto-launching it), then runs `run_full_auto` on each
   interrupted project sequentially. Every phase already skips finished work
   (generated scenes, upscaled clips, active music track), so only the missing
   pieces are produced.
3. **Ghost-status healing on `/resume`**: if the DB says "running" but the
   pipeline is idle (dead thread), the endpoint now rolls the status back and
   resumes instead of 409-ing (previously required manual DB surgery).
4. **Run journal** (`projects/<id>/run_state.json`): atomically rewritten on
   every progress tick — phase, scene N/total, %, message, pid, timestamp.
   After any kill you can see exactly where it died.
5. **SQLite WAL + busy_timeout** (`app/database.py`): crash-safe journaling,
   no more "database is locked" between pipeline thread and API reads.
6. Already in place and load-bearing: `debug=False` (uvicorn reload was killing
   runs on .py edits), per-scene skip logic, orphaned-RUNNING-generation repair,
   ComfyUI auto-launch in `wait_ready`.

**Still worth adding later (watchdog tier):** per-clip hard timeout that
cancels the ComfyUI queue + restarts the ComfyUI process on hang (today a hung
ComfyUI job stalls the run until manually restarted — but now a manual app
restart auto-heals everything, which covers most of the pain).

---

## ✅ Premium opening + production reports (DONE 2026-07-15)

**Premium opening** (`config.video.premium_*`): scenes starting in the first
20s render at **960×544 @ 16 steps** instead of 832×480 @ 8 — the hook always
looks best. Benched on the 16GB card (121-frame img2vid): 832×480@8 = 46s,
960×544@8 = 55s, **960×544@16 = 88s (chosen — VRAM-safe)**, 1024×576 = 574s
(spills, rejected). ~4 premium clips add only ~3 min per video. Tune via
`premium_open_seconds/width/height/steps`; `premium_open_seconds: 0` disables.

**Production report**: every render now writes `report.md` +
`production_report.json` (total wall clock, GPU clip time, avg s/clip,
clip resolution/steps profiles, upscale wall clock, music variants, render
stats). Backfilled for both existing videos.

## Tier 1 — Output quality (the difference between "generated" and "watchable")

### 1.1 Real lyric↔video sync (biggest win for song videos)
`lyrics_parser.py` distributes lines evenly — chorus visuals drift seconds off
the vocals. Add faster-whisper/WhisperX forced alignment on the generated song
→ word timestamps → cut scenes exactly on phrases. CPU-friendly, can run while
the GPU renders clips.

### 1.2 IC-LoRA character consistency A/B (files already in place)
43GB BF16 LTX + 1.3GB IC-LoRA downloaded; reference-sheet recipe documented.
Run the official workflow + A/B against current Z-Image + locked-seed approach;
adopt if clearly better.

### 1.3 Frames-vs-fps bug (open, known)
Clip duration math mismatch flagged in the wizard sessions — audit
`num_frames`/`fps` handoff from scene duration to LTX/Wan workflows so a "5s"
scene is actually 5s. Cheap fix, affects every video's sync.

### 1.4 Re-download the corrupt ACE-Step Turbo model
Turbo engine fails at UNETLoader (truncated download + leftover .tmp). SFT
works, so this only blocks the fastest music engine.

## Tier 2 — Close the loop to YouTube

### 2.1 `metadata.json` next to every render
Title, description, tags, hashtags, made_for_kids — the director emits these
for story projects; song projects don't. Small, unblocks upload automation.

### 2.2 Thumbnail generation
One dedicated Z-Image still (1280×720, big character face, high contrast) +
optional PIL title overlay. Thumbnails drive CTR more than video quality.

### 2.3 YouTube upload — private/scheduled first
`youtube_upload.py` scaffolding exists; needs OAuth + upload call. Nothing
goes public unreviewed.

## Tier 3 — Overnight scale

### 3.1 Job queue for multi-video nights
`autoproduce.py` handles one video. Add a `jobs/*.yaml` queue the server (not a
script — app-native, per your direction) drains sequentially: each job =
channel + title/lyrics + duration. With Tier 0 done, a failed job no longer
kills the night — the next boot resumes it.

### 3.2 End-of-night report
`report.md` per night: per-job status, timings, output paths, QA summaries —
one glance over coffee.

## Tier 4 — Maintainability (opportunistic)

- Repo root has ~80 one-off scripts/logs/PNGs and 3 SQLite DBs. Move keepers to
  `scripts/`, gitignore `*.log/*.png/*.db`, delete `app.db`/`director.db`.
- Commit discipline: `ai_director.db` is tracked in git and constantly dirty —
  gitignore it (it's runtime state, not source).
- Song-pipeline unit tests (lyrics parser needs no GPU).

---

## Suggested order of attack

| Step | Item | Effort | Payoff |
|------|------|--------|--------|
| ✅ | Tier 0 extreme resumability | done | Crashes stop costing nights |
| 1 | 1.3 frames-vs-fps audit | small | Every video's timing correct |
| 2 | 1.1 WhisperX lyric sync | ~1 session | Videos feel professionally edited |
| 3 | 2.1 + 2.2 metadata + thumbnails | small | Upload-ready output |
| 4 | 1.2 IC-LoRA A/B | ~1 session | Channel identity |
| 5 | 3.1 job queue | medium | Multi-video nights |
| 6 | 2.3 YouTube upload (private) | medium | Full loop closed |
| 7 | 1.4 turbo re-download + Tier 4 | background | Sanity |
