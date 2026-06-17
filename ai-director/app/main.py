"""
AI Director — FastAPI Application
REST API + WebSocket for the web UI.
"""
import json
import logging
import threading
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

# ── Globals ────────────────────────────────────────────────────────────────

model_manager = ModelManager()
pipeline: Optional[PipelineOrchestrator] = None
ws_clients: list[WebSocket] = []


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    # Startup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    init_db(str(settings.paths.database))
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


# ── WebSocket Progress ─────────────────────────────────────────────────────

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
    for ws in ws_clients[:]:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(ws.send_json(data))
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
                pipeline.cancel()
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

class UpdateSceneReq(BaseModel):
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    scene_type: Optional[str] = None
    duration: Optional[float] = None
    camera_motion: Optional[str] = None

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


@app.get("/api/projects/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)):
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    scenes_data = []
    for s in project.scenes:
        gen = s.active_generation
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
            "clip_path": (gen.upscaled_path or gen.output_path) if gen else None,
            "thumbnail_path": gen.thumbnail_path if gen else None,
        })

    return {
        "id": project.id,
        "title": project.title,
        "status": project.status.value,
        "duration_target": project.duration_target,
        "context": project.context,
        "total_scenes": project.total_scenes,
        "completed_scenes": project.completed_scenes,
        "output_path": project.output_path,
        "channel": {"name": project.channel.name, "slug": project.channel.slug},
        "scenes": scenes_data,
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
        try:
            pipeline.start_generation(project_id)
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "message": "Asset generation started"}


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


@app.post("/api/projects/{project_id}/generate-music")
def generate_music(project_id: str, db: Session = Depends(get_db)):
    """Phase 5: Generate background music via ACE-Step."""
    project = db.query(Project).get(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    def run():
        try:
            pipeline.generate_music(project_id)
        except Exception as e:
            logger.error(f"Music generation failed: {e}", exc_info=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"status": "started", "message": "Music generation started"}


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
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse(str(frontend_dir / "index.html"))
