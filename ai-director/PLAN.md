# AI Director — Local AI YouTube Video Generation Pipeline

## Vision
A fully local, single-GPU AI pipeline that generates complete YouTube videos from just a title, duration, and context. An LLM "director" plans every scene, dispatches work to specialized generation models (image, video, music), and assembles the final render — all with a web UI for human approval at every stage.

---

## Hardware Baseline
| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5070 Ti — 16 GB VRAM |
| RAM | 64 GB DDR4 |
| Storage | 1 TB Gen4 NVMe (fast) + planned 4-8 TB HDD (assets) |
| CPU | Intel i5-14400F (10C/16T) |
| OS | Windows (WSL2 or native Python) |

**Key constraint:** single GPU → only ONE model in VRAM at a time. The entire pipeline is sequential: load model → generate → unload → next model.

---

## Model Registry

### 1. Director Brain — LLM
| Model | Quant | File Size | VRAM Usage | Role |
|-------|-------|-----------|------------|------|
| **Qwen2.5-32B-Instruct-GGUF** | Q4_K_M | ~20 GB | 10-14 GB (partial offload) | Primary director — scene planning, prompt engineering, quality evaluation |
| Qwen2.5-14B-Instruct-GGUF | Q5_K_M | ~10 GB | ~10 GB (full GPU) | Fallback — faster, fits entirely on GPU |
| Qwen2.5-VL-7B-Instruct-GGUF | Q5_K_M | ~5 GB | ~5 GB | Optional — visual QA to evaluate generated clips |

**Source:** `https://huggingface.co/Qwen/` (search for GGUF variants by bartowski or similar quantizers)  
**Runtime:** `llama-cpp-python` with CUDA backend  
**Strategy:** Load with `n_gpu_layers` tuned to fit VRAM. 32B Q4_K_M needs ~40 layers; offload ~10 to CPU (64GB RAM absorbs this easily). Unload completely before generation phase.

### 2. Image Generation
| Model | Size | VRAM | Role |
|-------|------|------|------|
| **SDXL 1.0 Base** | ~6.5 GB (fp16) | ~8 GB | Primary image gen — 1024×1024 or 768×1344 |
| Juggernaut XL v9 | ~6.5 GB | ~8 GB | Alternative — photorealistic fine-tune of SDXL |
| DreamShaper XL | ~6.5 GB | ~8 GB | Alternative — artistic/fantasy fine-tune |

**Source:** `https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0` or CivitAI  
**Runtime:** `diffusers` library with `torch.float16`  
**LoRA support:** Yes — load/unload per-scene via diffusers' `load_lora_weights()`  
**Output:** 1024×1024 base → crop/resize to 720p (1280×720) → upscale to 1080p

### 3. Video Generation
| Model | Quant | VRAM | Output | Role |
|-------|-------|------|--------|------|
| **LTX-Video 2.3 22B Distilled** | FP8 | ~14 GB | 480-544p, 5-7s | Primary — fast, good motion |
| LTX-Video 2.3 GGUF | Q3_K_S | ~10 GB | 480p, 5-7s | Lighter alternative |
| Wan2.1-14B | FP8/GGUF | ~12-14 GB | 480p, 5-7s | Alternative — different style |
| CogVideoX-5B | FP16 | ~12 GB | 480p, 6s | Alternative — good for characters |

**Source:** LTX already at `C:\Users\PC\Documents\LTx models\`  
**Runtime:** LTX via DistilledPipeline + SingleGPUModelBuilder (NOT diffusers LTX2Pipeline). Wan via diffusers.  
**Strategy:** Director chooses txt2vid or img2vid per scene. Generate at max resolution that fits VRAM → upscale.

### 4. Upscaling
| Model | Size | VRAM | Scale |
|-------|------|------|-------|
| **Real-ESRGAN x4plus** | ~65 MB | ~500 MB | 4× (480p→1920p, crop to 1080p) |
| Real-ESRGAN x4plus-anime | ~65 MB | ~500 MB | 4× (better for cartoon/anime styles) |
| Real-ESRGAN x2plus | ~65 MB | ~300 MB | 2× (720p→1440p, crop to 1080p) |

**Source:** `https://github.com/xinntao/Real-ESRGAN`  
**Runtime:** `realesrgan-ncnn-vulkan` (standalone binary) or `basicsr` Python package  
**Strategy:** For video clips — extract frames, upscale batch, reassemble. For images — single pass.

### 5. Music Generation
| Model | Size | VRAM | Output |
|-------|------|------|--------|
| **ACE-Step v1.5** | ~3 GB | ~6 GB | Full songs with lyrics/instrumental |

**Source:** `https://huggingface.co/ACE-Step/`  
**Runtime:** Custom Python inference  
**Strategy:** Generate background music per video style. Cache music tracks for reuse across videos.

### 6. TTS (Text-to-Speech)
| Model | Size | VRAM | Output |
|-------|------|------|--------|
| **WanGP Omnivoice** | varies | ~4 GB | Natural speech with emotion |
| Silero TTS v4 | ~100 MB | CPU only | Fallback — lightweight |

**Already configured** in existing pipeline via `wangp_client.py`.

### 7. Audio Processing
| Tool | Role |
|------|------|
| **WhisperX** | Forced alignment of TTS to get word-level timestamps |
| **FFmpeg** | Audio mixing, video assembly, final render |

---

## Pipeline Architecture

```
USER INPUT (title, duration, context, channel)
        │
        ▼
┌─────────────────────────────────┐
│  PHASE 1: SCRIPT GENERATION     │  ← Qwen 32B in VRAM
│  - Analyze title & context      │
│  - Generate scene breakdown     │
│  - Assign scene types           │
│  - Write prompts + negatives    │
│  - Plan camera motions          │
│  - Suggest LoRAs per scene      │
│  Output: SceneList JSON         │
└──────────┬──────────────────────┘
           │  ★ USER APPROVAL GATE ★
           ▼
┌─────────────────────────────────┐
│  PHASE 2: TTS GENERATION        │  ← TTS model in VRAM
│  - Generate narration audio     │
│  - WhisperX forced alignment    │
│  - Extract word timestamps      │
│  Output: narration.wav + align  │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│  PHASE 3: ASSET GENERATION      │  ← One model at a time
│                                 │
│  For each scene (sequential):   │
│  ┌─ img2vid? ──────────────┐    │
│  │  Load SDXL → gen image  │    │
│  │  Unload SDXL            │    │
│  │  Load LTX → img2vid     │    │
│  │  Unload LTX             │    │
│  └──────────────────────────┘    │
│  ┌─ txt2vid? ──────────────┐    │
│  │  Load LTX → txt2vid     │    │
│  │  Unload LTX             │    │
│  └──────────────────────────┘    │
│  ┌─ still + pan? ──────────┐    │
│  │  Load SDXL → gen image  │    │
│  │  Unload SDXL            │    │
│  │  Ken Burns via FFmpeg   │    │
│  └──────────────────────────┘    │
│                                 │
│  After each clip:               │
│  - Save to version history      │
│  - Optional: Qwen QA check      │
│  Output: clips/ folder          │
└──────────┬──────────────────────┘
           │  ★ USER REVIEW GATE ★
           │  (approve/retry/regenerate)
           ▼
┌─────────────────────────────────┐
│  PHASE 4: UPSCALE               │  ← Real-ESRGAN in VRAM
│  - Frame extraction per clip    │
│  - Batch upscale 480p → 1080p   │
│  - Reassemble clips             │
│  Output: clips_hd/ folder       │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│  PHASE 5: MUSIC                  │  ← ACE-Step in VRAM
│  - Generate background track    │
│  - Match duration to video      │
│  Output: music.wav              │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│  PHASE 6: ASSEMBLY + RENDER     │  ← FFmpeg (CPU)
│  - Sequence all HD clips        │
│  - Mix narration + music        │
│  - Add transitions (crossfade)  │
│  - Render final 1080p/2K MP4    │
│  Output: final_video.mp4        │
└──────────┬──────────────────────┘
           │  ★ USER FINAL REVIEW ★
           ▼
┌─────────────────────────────────┐
│  PHASE 7: UPLOAD (optional)     │
│  - YouTube Data API v3          │
│  - Auto title, description, tags│
│  - Thumbnail from best frame    │
│  - Schedule publish time        │
└─────────────────────────────────┘
```

---

## Database Schema

```
projects
├── id (UUID)
├── title
├── channel_id (FK → channels)
├── duration_target (seconds)
├── context (user input text)
├── status (draft | scripted | approved | generating | upscaling | assembling | rendered | uploaded)
├── created_at
└── updated_at

channels
├── id (UUID)
├── name ("Little Fairy Dreams")
├── profile_yaml (path to channel config)
├── system_prompt (preloaded LLM context)
├── default_loras (JSON array)
├── still_ratio (0.0 - 1.0)
├── target_resolution (1080p | 2k)
└── made_for_kids (bool)

scenes
├── id (UUID)
├── project_id (FK)
├── scene_number (ordering)
├── scene_type (txt2vid | img2vid | still_pan | narration_only)
├── prompt
├── negative_prompt
├── duration (seconds)
├── camera_motion (static | pan_left | pan_right | zoom_in | zoom_out | tilt_up)
├── lora_ids (JSON array)
├── status (pending | generating | generated | approved | failed)
├── retry_count
├── active_generation_id (FK → generations, which version is "selected")
└── director_notes (JSON — style cues, transition hints)

generations
├── id (UUID)
├── scene_id (FK)
├── version (1, 2, 3...)
├── model_used (ltx-2.3 | sdxl | wan-14b)
├── output_path (raw clip)
├── upscaled_path (HD clip)
├── thumbnail_path
├── prompt_used (actual prompt sent to model)
├── negative_prompt_used
├── seed
├── parameters (JSON — steps, cfg, resolution, fps, etc.)
├── quality_score (0-100, from Qwen VL evaluation or manual)
├── status (completed | failed | superseded)
├── error_log
├── generation_time_seconds
└── created_at

music_tracks
├── id (UUID)
├── project_id (FK)
├── style_prompt
├── output_path
├── duration
└── created_at

render_jobs
├── id (UUID)
├── project_id (FK)
├── resolution (1080p | 2k)
├── output_path
├── status (queued | rendering | completed | failed)
├── progress_pct
└── created_at
```

---

## API Endpoints

### Projects
```
POST   /api/projects              — Create new video project
GET    /api/projects              — List all projects
GET    /api/projects/{id}         — Get project with scenes
PUT    /api/projects/{id}         — Update project
DELETE /api/projects/{id}         — Delete project
```

### Pipeline Control
```
POST   /api/projects/{id}/generate-script   — Phase 1: Director generates scenes
POST   /api/projects/{id}/approve-script    — User approves scene list
POST   /api/projects/{id}/start-generation  — Phase 3: Begin asset generation
POST   /api/projects/{id}/start-upscale     — Phase 4: Upscale all clips
POST   /api/projects/{id}/generate-music    — Phase 5: Generate music
POST   /api/projects/{id}/render            — Phase 6: Final assembly
POST   /api/projects/{id}/upload            — Phase 7: YouTube upload
```

### Scene Management
```
PUT    /api/scenes/{id}                    — Edit scene prompt/settings
POST   /api/scenes/{id}/regenerate         — Retry generation for a scene
POST   /api/scenes/{id}/approve            — Approve a scene clip
GET    /api/scenes/{id}/versions           — Get all generation versions
PUT    /api/scenes/{id}/select-version     — Pick a specific version
```

### System
```
GET    /api/system/gpu-status      — Current VRAM usage, loaded model
GET    /api/system/queue            — Generation queue status
GET    /api/channels                — List channel profiles
WS     /ws/pipeline/{project_id}   — Real-time progress updates
```

---

## Web UI Pages

1. **Dashboard** — All projects, status overview, GPU status
2. **New Project** — Title, duration, channel, context input
3. **Script Review** — Scene-by-scene breakdown, edit prompts, approve/reject
4. **Generation Monitor** — Live progress, current scene, ETA, preview thumbnails
5. **Clip Review** — Grid of all generated clips with video preview, approve/retry/regenerate per clip
6. **Version History** — Side-by-side comparison of generation versions for a scene
7. **Render Settings** — Resolution, transitions, music track, final preview
8. **Channel Management** — Channel profiles, LoRA library, system prompts

---

## Channel Profile Example: Little Fairy Dreams

```yaml
name: Little Fairy Dreams
audience: kids_under_5_girls
language: en
made_for_kids: true
style:
  color_palette: pastel (soft pink, lavender, mint, sky blue, peach)
  visual_style: dreamy storybook illustration, watercolor, magical
  avoid: human faces, text in frames, dark themes, scary elements, sharp edges
  prefer: fairy gardens, unicorns, butterflies, castles, rainbows, flowers, stars
narration:
  voice: soft warm female
  pace: slow and gentle
  style: minimal narration, more ambient
music:
  style: soft lullaby, gentle piano, music box, wind chimes
  tempo: slow (60-80 bpm)
  mood: dreamy, peaceful, magical
video:
  target_duration: 300-480 seconds (5-8 min)
  clip_duration: 5-7 seconds
  still_ratio: 0.4 (40% stills with Ken Burns, 60% motion)
  transitions: soft crossfade (0.5s)
  fps: 24
default_loras: []
monetization:
  cpm_range: $2-5 USD
  strategy: autoplay volume, high retention through gentle pacing
  upload_frequency: 1/day
```

---

## VRAM Budget Per Phase

| Phase | Model | VRAM | Duration (est.) |
|-------|-------|------|-----------------|
| 1. Script | Qwen 32B Q4_K_M | ~12 GB (partial offload) | 30-60s |
| 2. TTS | WanGP Omnivoice | ~4 GB | 10-30s |
| 3a. Image | SDXL fp16 | ~8 GB | 15-30s per image |
| 3b. Video | LTX 2.3 fp8 | ~14 GB | 90-180s per 5s clip |
| 4. Upscale | Real-ESRGAN | ~500 MB | 5-15s per clip |
| 5. Music | ACE-Step | ~6 GB | 30-60s per track |
| 6. Render | FFmpeg (CPU) | 0 | 60-120s |

**Total for 5-min video (60 clips):**
- Script: ~1 min
- TTS: ~1 min
- Images (24 stills): ~12 min
- Videos (36 motion clips): ~90 min (worst case, ~2.5 min each)
- Upscale: ~15 min
- Music: ~1 min
- Render: ~2 min
- **Total: ~2-3 hours per video** (run overnight for daily upload)

---

## Implementation Phases

### Phase A: Foundation (Week 1-2)
- [x] Project structure
- [ ] Config management + model registry
- [ ] ModelManager (VRAM orchestrator)
- [ ] Database models + migrations
- [ ] FastAPI skeleton + WebSocket
- [ ] Basic web UI shell

### Phase B: Director Brain (Week 2-3)
- [ ] Qwen integration via llama-cpp-python
- [ ] Channel profile loading
- [ ] Scene breakdown generation
- [ ] Prompt engineering pipeline
- [ ] Script approval workflow

### Phase C: Generation Engine (Week 3-5)
- [ ] SDXL image generation wrapper
- [ ] LTX video generation wrapper (adapt existing code)
- [ ] Ken Burns still-to-video processor
- [ ] Real-ESRGAN upscaler
- [ ] Retry logic + version tracking

### Phase D: Assembly (Week 5-6)
- [ ] ACE-Step music generation
- [ ] FFmpeg assembly pipeline
- [ ] Transition effects
- [ ] Audio mixing (narration + music)
- [ ] Final render at target resolution

### Phase E: Web UI (Week 6-8)
- [ ] Full clip review interface
- [ ] Version comparison
- [ ] Generation monitor with live preview
- [ ] Channel management
- [ ] Render settings

### Phase F: Automation (Week 8-10)
- [ ] YouTube Data API upload
- [ ] Scheduling system
- [ ] Multi-channel support
- [ ] Quality auto-evaluation (Qwen-VL)
- [ ] Historical performance tracking
