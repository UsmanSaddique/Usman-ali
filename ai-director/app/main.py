"""
AI Director — FastAPI Application
REST API + WebSocket for the web UI.
"""
import json
import logging
import threading
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database import (
    init_db, get_db, Channel, Project, Scene, Generation, LoRA,
    ProjectStatus, SceneStatus, GenerationStatus,
)
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator, PipelineProgress

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

# ── Globals ────────────────────────────────────────────────────────────────

model_manager = ModelManager()
pipeline: Optional[PipelineOrchestrator] = None
ws_clients: list[WebSocket] = []


def _repair_stuck_projects():
    """Reset any projects stuck in active pipeline states to FAILED on startup."""
    from app.database import get_session, Project, ProjectStatus
    session = get_session()
    try:
        stuck_projects = session.query(Project).filter(
            Project.status.in_([
                ProjectStatus.GENERATING,
                ProjectStatus.UPSCALING,
                ProjectStatus.ASSEMBLING,
            ])
        ).all()
        for proj in stuck_projects:
            logger.info(f"[Main] Resetting stuck project {proj.id} from {proj.status.value} to FAILED on startup")
            proj.status = ProjectStatus.FAILED
            proj.error_log = "Server restarted or recovered from ghost state."
        if stuck_projects:
            session.commit()
    except Exception as e:
        logger.error(f"Failed to repair stuck projects: {e}")
    finally:
        session.close()


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, main_loop
    main_loop = asyncio.get_running_loop()
    # Startup
    import os
    # Suppress bitsandbytes CUDA binary warnings — we don't use quantization
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    os.environ.setdefault("BNB_CUDA_VERSION", "132")  # Match your CUDA version
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    
    # Register WebSocketLogHandler to root logger
    ws_log_handler = WebSocketLogHandler()
    ws_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logging.getLogger().addHandler(ws_log_handler)

    # Global PyTorch optimizations (TF32 on GPU)
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("[Main] Global PyTorch optimizations enabled: TF32 allowed on GPU")
    except ImportError:
        pass

    init_db(str(settings.paths.database))
    _repair_stuck_projects()
    register_all_loaders(model_manager, settings)
    pipeline = PipelineOrchestrator(settings, model_manager)
    pipeline.on_progress(broadcast_progress_sync)
    logger.info("AI Director started")
    _seed_default_channel()
    yield
    # Shutdown
    model_manager.unload()
    logger.info("AI Director stopped")


app = FastAPI(title="AI Director", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket Progress & Logging ───────────────────────────────────────────

main_loop = None

def broadcast_log(message: str):
    """Broadcast log message to all connected WebSocket clients."""
    data = {
        "type": "log",
        "message": message
    }
    if not main_loop:
        return
    for ws in ws_clients[:]:
        try:
            import asyncio
            asyncio.run_coroutine_threadsafe(ws.send_json(data), main_loop)
        except Exception:
            pass


class WebSocketLogHandler(logging.Handler):
    """Custom logging handler that streams logs to active WebSocket clients."""
    def emit(self, record):
        try:
            log_message = self.format(record)
            broadcast_log(log_message)
        except Exception:
            pass


def broadcast_progress_sync(progress: PipelineProgress):
    """Called from pipeline thread — schedule async broadcast."""
    import asyncio
    data = {
        "type": "progress",
        "phase": progress.phase.value,
        "project_id": progress.project_id,
        "current_scene": progress.current_scene,
        "total_scenes": progress.total_scenes,
        "percent": progress.percent,
        "eta_seconds": progress.eta_seconds,
        "message": progress.message,
        "error": progress.error,
    }
    # Store latest for new connections
    app.state.last_progress = data
    # Non-blocking broadcast (best effort from sync context)
    if not main_loop:
        return
    for ws in ws_clients[:]:
        try:
            asyncio.run_coroutine_threadsafe(ws.send_json(data), main_loop)
        except Exception:
            pass


@app.websocket("/ws/pipeline/{project_id}")
async def ws_pipeline(websocket: WebSocket, project_id: str):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        # Send last known progress
        if hasattr(app.state, 'last_progress'):
            await websocket.send_json(app.state.last_progress)
        while True:
            # Keep alive — client can also send commands
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("command") == "cancel":
                from app.services.pipeline import PipelinePhase
                is_running = pipeline._progress.phase not in [PipelinePhase.IDLE, PipelinePhase.DONE, PipelinePhase.ERROR]
                pipeline.cancel()
                
                if not is_running:
                    from app.database import get_db, Project, ProjectStatus
                    db = next(get_db())
                    try:
                        project = db.query(Project).get(project_id)
                        if project and project.status in [ProjectStatus.GENERATING, ProjectStatus.UPSCALING, ProjectStatus.ASSEMBLING]:
                            logger.info(f"Force resetting ghost project {project_id} to FAILED since pipeline is IDLE")
                            project.status = ProjectStatus.FAILED
                            project.error_log = "Force reset to FAILED by user (cancellation on inactive pipeline)."
                            db.commit()
                            pipeline._emit_progress(
                                phase=PipelinePhase.ERROR,
                                error="Cancelled",
                                message="Pipeline was inactive. Status reset to FAILED.",
                            )
                    except Exception as e:
                        logger.error(f"Failed to force reset stuck project status: {e}")
                    finally:
                        db.close()
                await websocket.send_json({"type": "info", "message": "Cancellation requested"})
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.remove(websocket)


# ── Request/Response Models ────────────────────────────────────────────────

class CreateProjectReq(BaseModel):
    title: str
    channel_slug: str
    duration: int  # seconds
    context: str = ""
    num_scenes: Optional[int] = None
    video_model: Optional[str] = None
    lora_ids: Optional[list[str]] = None
    lora_weights: Optional[list[float]] = None
    lyrics: Optional[str] = None          # song mode: drives music vocals + scene timing
    music_style: Optional[str] = None
    music_model: Optional[str] = None     # auto|sft|turbo|heartmula|ace1
    upscale_inline: Optional[bool] = None


class MusicGenReq(BaseModel):
    """Optional overrides for the music step (wizard step 2)."""
    engine: Optional[str] = None          # auto|sft|turbo|heartmula|ace1
    style: Optional[str] = None
    lyrics: Optional[str] = None
    vocals: Optional[bool] = None


class MusicVariantsReq(MusicGenReq):
    """Song audition round: N versions, same lyrics, tweaked styles."""
    count: int = 10
    resume: bool = False   # continue a paused/incomplete batch (server computes the offset)


class ScenesFromLyricsReq(BaseModel):
    num_clips: Optional[int] = None

class ManualScenesReq(BaseModel):
    prompts: list[str]

class UpdateProjectReq(BaseModel):
    title: Optional[str] = None
    duration: Optional[int] = None  # seconds
    context: Optional[str] = None
    num_scenes: Optional[int] = None

class GenerateScenesReq(BaseModel):
    scene_ids: list[str]
    video_model: str
    lora_ids: list[str] = []
    lora_weights: list[float] = []
    width: Optional[int] = None
    height: Optional[int] = None
    batch: bool = False   # keep the video model resident across scenes (skip per-scene unload)
    upscale_inline: bool = False  # upscale each clip right after it generates

class UpdateSceneReq(BaseModel):
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    scene_type: Optional[str] = None
    duration: Optional[float] = None
    camera_motion: Optional[str] = None

class ProduceReq(BaseModel):
    """One-call automation: create a project and run the full pipeline."""
    channel_slug: str
    title: str
    duration: int = 60            # seconds
    num_scenes: Optional[int] = None
    context: str = ""
    video_model: Optional[str] = None


class CreateChannelReq(BaseModel):
    name: str
    slug: str
    system_prompt: str = ""
    still_ratio: float = 0.4
    target_resolution: str = "1080p"
    made_for_kids: bool = False

class RegisterLoRAReq(BaseModel):
    name: str
    path: str
    model_type: str  # sdxl, ltx, wan
    trigger_words: list[str] = []
    description: str = ""
    default_weight: float = 0.7


# ── Project Endpoints ──────────────────────────────────────────────────────

@app.post("/api/projects")
def create_project(req: CreateProjectReq, db: Session = Depends(get_db)):
    channel = db.query(Channel).filter(Channel.slug == req.channel_slug).first()
    if not channel:
        raise HTTPException(404, f"Channel '{req.channel_slug}' not found")

    project = Project(
        title=req.title,
        channel_id=channel.id,
        duration_target=req.duration,
        context=req.context,
        num_scenes_target=req.num_scenes,
        video_model=req.video_model or "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
        default_lora_ids=req.lora_ids or [],
        default_lora_weights=req.lora_weights or [],
        lyrics=req.lyrics,
        music_style=req.music_style,
        music_model=req.music_model or "auto",
        upscale_inline=True if req.upscale_inline is None else req.upscale_inline,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    return {"id": project.id, "title": project.title, "status": project.status.value}


@app.get("/api/projects")
def list_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return [
        {
            "id": p.id, "title": p.title,
            "status": p.status.value,
            "duration": p.duration_target,
            "total_scenes": p.total_scenes,
            "completed_scenes": p.completed_scenes,
            "channel": p.channel.name if p.channel else None,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in projects
    ]


def _media_url(path: Optional[str]) -> Optional[str]:
    """Absolute file path under projects/ → served URL (/projects/...)."""
    if not path:
        return None
    try:
        projects_root = (Path(__file__).parent.parent / "projects").resolve()
        rel = Path(path).resolve().relative_to(projects_root)
        return "/projects/" + str(rel).replace("\\", "/")
    except Exception:
        return None


@app.get("/api/projects/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)):
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    scenes_data = []
    for s in project.scenes:
        gen = s.active_generation
        clip_path = (gen.upscaled_path or gen.output_path) if gen else None
        scenes_data.append({
            "id": s.id,
            "scene_number": s.scene_number,
            "scene_type": s.scene_type.value,
            "prompt": s.prompt,
            "negative_prompt": s.negative_prompt,
            "duration": s.duration,
            "camera_motion": s.camera_motion,
            "status": s.status.value,
            "retry_count": s.retry_count,
            "narration_text": s.narration_text,
            "versions": len(s.generations),
            "active_version": gen.version if gen else None,
            "clip_path": clip_path,
            "clip_url": _media_url(clip_path),
            "thumbnail_path": gen.thumbnail_path if gen else None,
            "thumb_url": _media_url(gen.thumbnail_path if gen else None),
            "upscaled": bool(gen and gen.upscaled_path and gen.upscaled_path != gen.output_path),
        })

    # Music tracks — the active one plays in the wizard; the full list is the
    # song-audition picker (10 versions, same lyrics, tweaked styles)
    from app.database import MusicTrack
    all_tracks = db.query(MusicTrack).filter(
        MusicTrack.project_id == project_id,
    ).order_by(MusicTrack.created_at).all()
    music = next((t for t in reversed(all_tracks) if t.is_active), None)
    music_info = {
        "url": _media_url(music.output_path) if music else None,
        "path": music.output_path if music else None,
        "duration": music.duration if music else 0,
        "generating": project_id in _music_jobs,
    }
    batch = _music_batches.get(project_id)
    if batch:
        music_info["batch"] = {
            "total": batch["total"],
            "done": max(0, len(all_tracks) - batch["baseline"]),
            "running": project_id in _music_jobs,
        }
    music_variants = [{
        "id": t.id,
        "url": _media_url(t.output_path),
        "style": t.style_prompt,
        "duration": t.duration,
        "active": bool(t.is_active),
    } for t in all_tracks if t.output_path and Path(t.output_path).exists()]

    return {
        "id": project.id,
        "title": project.title,
        "status": project.status.value,
        "duration_target": project.duration_target,
        "num_scenes_target": project.num_scenes_target,
        "context": project.context,
        "lyrics": project.lyrics,
        "music_style": project.music_style,
        "music_model": project.music_model,
        "upscale_inline": project.upscale_inline,
        "music": music_info,
        "music_variants": music_variants,
        "total_scenes": project.total_scenes,
        "completed_scenes": project.completed_scenes,
        "output_path": project.output_path,
        "error_log": project.error_log,
        "video_model": project.video_model,
        "default_lora_ids": project.default_lora_ids or [],
        "default_lora_weights": project.default_lora_weights or [],
        "channel": {"name": project.channel.name, "slug": project.channel.slug},
        "scenes": scenes_data,
    }


@app.put("/api/projects/{project_id}")
def update_project(project_id: str, req: UpdateProjectReq, db: Session = Depends(get_db)):
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    
    if project.status not in [ProjectStatus.DRAFT, ProjectStatus.SCRIPTED]:
        raise HTTPException(400, "Project cannot be updated in its current status")
        
    if req.title is not None:
        project.title = req.title
    if req.duration is not None:
        project.duration_target = req.duration
    if req.num_scenes is not None:
        project.num_scenes_target = req.num_scenes
    if req.context is not None:
        project.context = req.context
        
    db.commit()
    db.refresh(project)
    return {
        "id": project.id,
        "title": project.title,
        "duration_target": project.duration_target,
        "num_scenes_target": project.num_scenes_target,
        "context": project.context
    }


# ── Pipeline Control Endpoints ─────────────────────────────────────────────

@app.post("/api/projects/{project_id}/generate-script")
def generate_script(project_id: str, db: Session = Depends(get_db)):
    """Phase 1: Director generates scene breakdown."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    def run():
        try:
            pipeline.generate_script(project_id)
        except Exception as e:
            logger.error(f"Script generation failed: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "message": "Script generation started"}


@app.post("/api/projects/{project_id}/approve-script")
def approve_script(project_id: str, db: Session = Depends(get_db)):
    """User approves the generated script — mark all scenes as ready."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    scenes = db.query(Scene).filter(Scene.project_id == project_id).all()
    for s in scenes:
        if s.status == SceneStatus.PENDING:
            pass  # keep pending for generation
    project.status = ProjectStatus.APPROVED
    db.commit()

    return {"status": "approved", "scene_count": len(scenes)}


@app.post("/api/projects/{project_id}/start-generation")
def start_generation(project_id: str, db: Session = Depends(get_db)):
    """Phase 3: Begin generating all scene assets."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    def run():
        from app.services.pipeline import PipelinePaused
        try:
            pipeline.start_generation(project_id)
        except PipelinePaused:
            logger.info(f"Generation paused for project {project_id}")
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "message": "Asset generation started"}


@app.post("/api/projects/{project_id}/generate-scenes")
def generate_scenes(project_id: str, req: GenerateScenesReq, db: Session = Depends(get_db)):
    """Phase 3 (Selective): Update models and generate specific scenes."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    logger.info(f"generate_scenes called for project {project_id} with req: {req.model_dump()}")

    # Update scene configs
    from app.database import Scene, SceneStatus
    scenes = db.query(Scene).filter(Scene.id.in_(req.scene_ids), Scene.project_id == project_id).all()
    for s in scenes:
        s.video_model = req.video_model
        s.lora_ids = req.lora_ids
        s.lora_weights = req.lora_weights
        s.status = SceneStatus.PENDING
    
    project.status = ProjectStatus.GENERATING
    db.commit()

    def run():
        from app.services.pipeline import PipelinePaused
        try:
            pipeline.start_generation(project_id, scene_ids=req.scene_ids,
                                      width=req.width, height=req.height, batch=req.batch,
                                      upscale_inline=req.upscale_inline)
        except PipelinePaused:
            logger.info(f"Generation paused for project {project_id}")
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "message": f"Asset generation started for {len(scenes)} scenes"}


@app.post("/api/projects/{project_id}/start-upscale")
def start_upscale(project_id: str, db: Session = Depends(get_db)):
    """Phase 4: Upscale all generated clips."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    def run():
        try:
            pipeline.start_upscale(project_id)
        except Exception as e:
            logger.error(f"Upscale failed: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "message": "Upscaling started"}


@app.post("/api/projects/{project_id}/render")
def render_project(project_id: str, db: Session = Depends(get_db)):
    """Phase 6: Final assembly and render."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    def run():
        try:
            pipeline.render(project_id)
        except Exception as e:
            logger.error(f"Render failed: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "message": "Rendering started"}


@app.post("/api/projects/{project_id}/generate-tts")
def generate_tts(project_id: str, db: Session = Depends(get_db)):
    """Phase 2: Generate narration audio via TTS."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    def run():
        try:
            pipeline.generate_tts(project_id)
        except Exception as e:
            logger.error(f"TTS generation failed: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "message": "TTS generation started"}


# project ids with a music job currently running (drives the wizard spinner)
_music_jobs: set = set()
# project id → {"total": N, "baseline": tracks before batch} for audition progress
_music_batches: dict = {}


@app.post("/api/projects/{project_id}/generate-music")
def generate_music(project_id: str, req: Optional[MusicGenReq] = None, db: Session = Depends(get_db)):
    """Phase 5 / wizard step 2: generate the music track.
    Optional body overrides engine/style/lyrics/vocals for this run."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if project_id in _music_jobs:
        return {"status": "already_running", "message": "Music generation already in progress"}

    r = req or MusicGenReq()
    # Persist the creative choices so full-auto / resume uses the same sound
    if r.style:
        project.music_style = r.style
    if r.engine:
        project.music_model = r.engine
    if r.lyrics is not None and r.lyrics.strip():
        project.lyrics = r.lyrics
    db.commit()

    _music_jobs.add(project_id)

    def run():
        try:
            pipeline.generate_music(
                project_id,
                style=r.style, lyrics=r.lyrics,
                engine=r.engine, vocals=r.vocals,
            )
        except Exception as e:
            logger.error(f"Music generation failed: {e}", exc_info=True)
        finally:
            _music_jobs.discard(project_id)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "message": "Music generation started"}


@app.post("/api/projects/{project_id}/generate-music-variants")
def generate_music_variants(project_id: str, req: Optional[MusicVariantsReq] = None, db: Session = Depends(get_db)):
    """Song audition round: generate N versions of the song (same lyrics,
    tweaked styles). None becomes active until POST select-music/{track_id}."""
    from app.database import MusicTrack
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if project_id in _music_jobs:
        return {"status": "already_running", "message": "Music generation already in progress"}

    r = req or MusicVariantsReq()
    # Persist the creative choices so resume / full-auto uses the same sound
    if r.style:
        project.music_style = r.style
    if r.engine:
        project.music_model = r.engine
    if r.lyrics is not None and r.lyrics.strip():
        project.lyrics = r.lyrics
    db.commit()

    current_count = db.query(MusicTrack).filter(MusicTrack.project_id == project_id).count()
    count = max(1, min(r.count, 10))
    offset = 0
    if r.resume:
        # continue where the paused/incomplete batch stopped — never repeat a style
        prev = _music_batches.get(project_id)
        if prev:
            done = max(0, current_count - prev["baseline"])
            offset = prev.get("offset", 0) + done
            count = max(1, prev["total"] - done)
    _music_batches[project_id] = {"total": count, "baseline": current_count, "offset": offset}
    _music_jobs.add(project_id)

    def run():
        try:
            pipeline.generate_music_variants(
                project_id, count=count, offset=offset,
                style=r.style, lyrics=r.lyrics,
                engine=r.engine, vocals=r.vocals,
            )
        except Exception as e:
            logger.error(f"Music variants generation failed: {e}", exc_info=True)
        finally:
            _music_jobs.discard(project_id)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "count": count, "offset": offset}


@app.post("/api/projects/{project_id}/pause")
def pause_pipeline(project_id: str):
    """Gracefully pause the running pipeline after the current item (scene /
    song version). Everything finished is kept; the project stays resumable —
    the same step button (or /resume) continues where it stopped."""
    pipeline.request_pause()
    return {"status": "pausing", "message": "Pausing after the current item finishes"}


@app.post("/api/projects/{project_id}/cancel")
def cancel_pipeline(project_id: str):
    """Hard-stop the running pipeline (marks the project failed — it can still
    be resumed later, but pause is the friendlier option)."""
    pipeline.cancel()
    return {"status": "cancelling", "message": "Stopping the pipeline"}


@app.post("/api/projects/{project_id}/select-music/{track_id}")
def select_music(project_id: str, track_id: str, db: Session = Depends(get_db)):
    """Pick the winning song version — it becomes the project's music track
    (the final render always uses the active track)."""
    from app.database import MusicTrack
    track = db.query(MusicTrack).filter(
        MusicTrack.id == track_id,
        MusicTrack.project_id == project_id,
    ).first()
    if not track:
        raise HTTPException(404, "Music track not found")
    db.query(MusicTrack).filter(
        MusicTrack.project_id == project_id,
        MusicTrack.is_active == True,
    ).update({"is_active": False})
    track.is_active = True
    db.commit()
    return {"status": "ok", "selected": track_id}


@app.post("/api/projects/{project_id}/scenes-from-lyrics")
def scenes_from_lyrics(project_id: str, req: Optional[ScenesFromLyricsReq] = None, db: Session = Depends(get_db)):
    """Wizard step 3 (fast path): build beat-synced scenes from the project lyrics — no LLM."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    try:
        n = pipeline.generate_scenes_from_lyrics(
            project_id, num_clips=(req.num_clips if req else None))
        return {"status": "ok", "scene_count": n}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/projects/{project_id}/scenes-manual")
def scenes_manual(project_id: str, req: ManualScenesReq, db: Session = Depends(get_db)):
    """Wizard step 3: Manually add scenes from a list of prompts."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    from app.database import Scene, SceneType, SceneStatus
    current_count = db.query(Scene).filter(Scene.project_id == project_id).count()

    new_scenes = []
    for i, p in enumerate(req.prompts):
        if not p.strip(): continue
        scene = Scene(
            project_id=project_id,
            scene_number=current_count + i + 1,
            scene_type=SceneType.TXT2VID,
            prompt=p.strip(),
            duration=4.0,
            status=SceneStatus.PENDING,
        )
        db.add(scene)
        new_scenes.append(scene)
    
    db.commit()
    return {"status": "ok", "scene_count": current_count + len(new_scenes), "added": len(new_scenes)}


@app.post("/api/projects/{project_id}/resume")
def resume_project(project_id: str, db: Session = Depends(get_db)):
    """Resume an interrupted/failed project from wherever it stopped.
    Finished scenes, upscales and the music track are all reused — only the
    missing pieces are generated."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    if project.status in (ProjectStatus.GENERATING, ProjectStatus.UPSCALING, ProjectStatus.ASSEMBLING):
        raise HTTPException(409, "Pipeline already running for this project")

    def run():
        try:
            pipeline.run_full_auto(project_id)
        except Exception as e:
            logger.error(f"Resume failed: {e}", exc_info=True)

    threading.Thread(target=run, daemon=True).start()
    return {"status": "resumed", "message": "Continuing from the last completed step"}


@app.post("/api/projects/{project_id}/full-auto")
def full_auto(project_id: str, db: Session = Depends(get_db)):
    """Run entire pipeline automatically (overnight mode)."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    def run():
        try:
            pipeline.run_full_auto(project_id)
        except Exception as e:
            logger.error(f"Full auto failed: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "message": "Full auto pipeline started"}


# ── Scene Endpoints ────────────────────────────────────────────────────────

@app.put("/api/scenes/{scene_id}")
def update_scene(scene_id: str, req: UpdateSceneReq, db: Session = Depends(get_db)):
    scene = db.query(Scene).get(scene_id)
    if not scene:
        raise HTTPException(404, "Scene not found")

    if req.prompt is not None:
        scene.prompt = req.prompt
    if req.negative_prompt is not None:
        scene.negative_prompt = req.negative_prompt
    if req.scene_type is not None:
        scene.scene_type = req.scene_type
    if req.duration is not None:
        scene.duration = req.duration
    if req.camera_motion is not None:
        scene.camera_motion = req.camera_motion

    db.commit()
    return {"status": "updated"}


@app.post("/api/scenes/{scene_id}/regenerate")
def regenerate_scene(scene_id: str, db: Session = Depends(get_db)):
    """Regenerate a single scene (retry with current prompt)."""
    scene = db.query(Scene).get(scene_id)
    if not scene:
        raise HTTPException(404, "Scene not found")

    scene.status = SceneStatus.PENDING
    scene.retry_count = 0
    db.commit()

    def run():
        try:
            pipeline.start_generation(scene.project_id)
        except Exception as e:
            logger.error(f"Regeneration failed: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "regenerating"}


@app.post("/api/scenes/{scene_id}/approve")
def approve_scene(scene_id: str, db: Session = Depends(get_db)):
    scene = db.query(Scene).get(scene_id)
    if not scene:
        raise HTTPException(404, "Scene not found")
    scene.status = SceneStatus.APPROVED
    db.commit()
    return {"status": "approved"}


@app.get("/api/scenes/{scene_id}/versions")
def get_scene_versions(scene_id: str, db: Session = Depends(get_db)):
    scene = db.query(Scene).get(scene_id)
    if not scene:
        raise HTTPException(404, "Scene not found")

    return [
        {
            "id": g.id,
            "version": g.version,
            "model_used": g.model_used,
            "output_path": g.output_path,
            "upscaled_path": g.upscaled_path,
            "thumbnail_path": g.thumbnail_path,
            "seed": g.seed,
            "quality_score": g.quality_score,
            "status": g.status.value,
            "generation_time": g.generation_time_sec,
            "created_at": g.created_at.isoformat() if g.created_at else None,
        }
        for g in scene.generations
    ]


@app.put("/api/scenes/{scene_id}/select-version")
def select_version(scene_id: str, generation_id: str, db: Session = Depends(get_db)):
    scene = db.query(Scene).get(scene_id)
    if not scene:
        raise HTTPException(404, "Scene not found")
    gen = db.query(Generation).get(generation_id)
    if not gen or gen.scene_id != scene_id:
        raise HTTPException(404, "Generation not found for this scene")
    scene.active_generation_id = generation_id
    db.commit()
    return {"status": "selected", "version": gen.version}


# ── Channel & LoRA Endpoints ──────────────────────────────────────────────

@app.get("/api/channels")
def list_channels(db: Session = Depends(get_db)):
    channels = db.query(Channel).all()
    return [
        {"id": c.id, "name": c.name, "slug": c.slug,
         "still_ratio": c.still_ratio, "resolution": c.target_resolution,
         "made_for_kids": c.made_for_kids}
        for c in channels
    ]


@app.post("/api/channels")
def create_channel(req: CreateChannelReq, db: Session = Depends(get_db)):
    channel = Channel(
        name=req.name,
        slug=req.slug,
        system_prompt=req.system_prompt,
        still_ratio=req.still_ratio,
        target_resolution=req.target_resolution,
        made_for_kids=req.made_for_kids,
    )
    db.add(channel)
    db.commit()
    return {"id": channel.id, "slug": channel.slug}


@app.get("/api/loras")
def list_loras(db: Session = Depends(get_db)):
    loras = db.query(LoRA).all()
    return [
        {"id": l.id, "name": l.name, "path": l.path,
         "model_type": l.model_type, "trigger_words": l.trigger_words,
         "default_weight": l.default_weight}
        for l in loras
    ]


@app.post("/api/loras")
def register_lora(req: RegisterLoRAReq, db: Session = Depends(get_db)):
    lora = LoRA(
        name=req.name, path=req.path, model_type=req.model_type,
        trigger_words=req.trigger_words, description=req.description,
        default_weight=req.default_weight,
    )
    db.add(lora)
    db.commit()
    return {"id": lora.id, "name": lora.name}


# ── System Endpoints ──────────────────────────────────────────────────────

@app.get("/api/system/gpu-status")
def gpu_status():
    stats = ModelManager.get_gpu_stats()
    current = model_manager.get_current()
    return {
        "gpu": stats,
        "loaded_model": {
            "type": current.model_type.value if current else None,
            "name": current.name if current else None,
            "vram_mb": current.vram_mb if current else 0,
        }
    }


@app.get("/api/system/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# ── Model / LoRA Discovery & ComfyUI Status ──────────────────────────────

@app.get("/api/available-models")
def available_models():
    """Scan ComfyUI diffusion_models folder for available video models."""
    models_dir = settings.paths.models_dir / "diffusion_models"
    results = []
    if models_dir.exists():
        for f in sorted(models_dir.iterdir()):
            if f.is_file() and f.suffix in (".gguf", ".safetensors"):
                fn = f.name
                fn_lower = fn.lower()
                family = "ltx" if "ltx" in fn_lower else "wan" if "wan" in fn_lower else "other"
                results.append({
                    "filename": fn,
                    "family": family,
                    "size_gb": round(f.stat().st_size / (1024**3), 2),
                })
    return results


@app.get("/api/available-loras")
def available_loras():
    """Scan ComfyUI loras folder for available LoRAs."""
    loras_dir = settings.paths.loras_dir
    results = []
    if loras_dir.exists():
        for f in sorted(loras_dir.iterdir()):
            if f.is_file() and f.suffix == ".safetensors":
                fn = f.name
                fn_lower = fn.lower()
                family = "ltx" if "ltx" in fn_lower else "wan" if "wan" in fn_lower else "other"
                results.append({
                    "filename": fn,
                    "family": family,
                    "size_mb": round(f.stat().st_size / (1024**2), 1),
                })
    return results


@app.get("/api/system/engine-status")
def engine_status():
    """Check ComfyUI engine availability (replaces old LTX subprocess check)."""
    from app.services.comfyui_client import ComfyUIClient, is_starting
    client = ComfyUIClient()
    running = client.ping()
    return {
        "running": running,
        "starting": (not running) and is_starting(),
        "engine": "ComfyUI API",
        "url": "http://127.0.0.1:8188",
    }


@app.post("/api/system/engine-start")
def engine_start():
    """Launch the ComfyUI engine if it isn't running (auto-launch also happens
    lazily whenever the pipeline needs it — this is the manual button)."""
    from app.services.comfyui_client import ComfyUIClient, is_starting
    client = ComfyUIClient()
    if client.ping():
        return {"status": "running"}
    if not client.ensure_running():
        raise HTTPException(500, "ComfyUI portable install not found — check the path in comfyui_client.py")
    return {"status": "starting" if is_starting() else "running"}


@app.get("/api/system/preflight")
def system_preflight():
    """Full automation-readiness check: engines, models, ffmpeg, disk, LLM."""
    from app.services.qa import preflight
    return preflight(settings).to_dict()


# ── Automation ─────────────────────────────────────────────────────────────

@app.post("/api/automation/produce")
def automation_produce(req: ProduceReq, db: Session = Depends(get_db)):
    """One-call hands-off production: preflight-gate, create the project, and
    run the full pipeline (script → scenes → clips → upscale → music → render)
    in the background. Poll GET /api/projects/{id} for status."""
    from app.services.qa import preflight
    report = preflight(settings)
    if not report.ok:
        raise HTTPException(503, f"Preflight failed: {report.summary()}")

    channel = db.query(Channel).filter(Channel.slug == req.channel_slug).first()
    if not channel:
        raise HTTPException(404, f"Channel '{req.channel_slug}' not found")

    project = Project(
        title=req.title,
        channel_id=channel.id,
        duration_target=req.duration,
        context=req.context,
        num_scenes_target=req.num_scenes,
        video_model=req.video_model or "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    project_id = project.id

    def run():
        try:
            pipeline.run_full_auto(project_id)
        except Exception as e:
            logger.error(f"Automation produce failed: {e}", exc_info=True)

    threading.Thread(target=run, daemon=True).start()

    return {
        "id": project_id,
        "title": project.title,
        "status": "started",
        "preflight": report.to_dict(),
        "poll": f"/api/projects/{project_id}",
    }


# ── Seed Data ──────────────────────────────────────────────────────────────

def _seed_default_channel():
    """Create the Little Fairy Dreams channel if it doesn't exist."""
    from app.database import get_session
    session = get_session()
    existing = session.query(Channel).filter(Channel.slug == "little-fairy-dreams").first()
    if not existing:
        channel = Channel(
            name="Little Fairy Dreams",
            slug="little-fairy-dreams",
            system_prompt="Kids under 5, girls/toddlers. Magical fairy/fantasy content.",
            still_ratio=0.4,
            target_resolution="1080p",
            made_for_kids=True,
        )
        session.add(channel)
        session.commit()
        logger.info("Seeded default channel: Little Fairy Dreams")
    session.close()


# ── Static Files (Web UI) ─────────────────────────────────────────────────

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    projects_dir = Path(__file__).parent.parent / "projects"
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    app.mount("/projects", StaticFiles(directory=str(projects_dir)), name="projects")

    @app.get("/")
    def serve_ui():
        return FileResponse(str(frontend_dir / "index.html"))
