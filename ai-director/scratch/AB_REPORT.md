# A/B Test — Clip-by-clip vs LTX Director (same 6 stills, same prompts, 30s fairy story)

**Winner: B (LTX Director) — faster AND better.**

| | A: clip-by-clip | B: LTX Director (1 node) |
|---|---|---|
| Wall time | **16.4 min** (6 stills + 6 clips; 4 "premium" clips @960×544/16 steps) | **12.2 min** (video + 1080p upscale; reused A's stills) |
| Output | 30.25s, 960×544, **no audio**, 1.8 Mbps | 31.7s, **1920×1080**, **native audio**, 11 Mbps |
| Per-clip cost | 44–222s per 5s clip | one pass for all 30s |
| Motion/continuity | hard cuts between 6 independent clips | continuous takes, smooth transitions (transition LoRA) |
| Detail (frames at 3s/15s/27s) | soft, flatter lighting, simpler backgrounds | much richer: dew on petals, sparkle particles, fur detail, cinematic light |

Files:
- A: `scratch/A_clipbyclip_30s.mp4`
- B: `projects/2ea03c9c-eb5b-46e8-9f86-437226ab4877/final_render.mp4`
- Frames: `scratch/cmp_A_*.png` / `scratch/cmp_B_*.png`

Notes: B renders native 1280×720 then Lanczos→1080p; A clips were 832×480/960×544 raw (no upscale pass run). B also generated ambient audio natively. A's number includes ~4 min of still generation which B reused — but even video-only A (~12.7 min) is no faster than B while producing 2.6× fewer pixels and no audio.
