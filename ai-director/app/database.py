"""
AI Director — Database Models
SQLAlchemy ORM with async SQLite.
"""
import uuid
import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    Text, DateTime, ForeignKey, Enum as SAEnum, JSON
)
from sqlalchemy.orm import (
    DeclarativeBase, relationship, sessionmaker, Session
)
import enum


# ── Enums ──────────────────────────────────────────────────────────────────

class ProjectStatus(str, enum.Enum):
    DRAFT = "draft"
    SCRIPTED = "scripted"
    APPROVED = "approved"
    GENERATING = "generating"
    UPSCALING = "upscaling"
    MUSIC = "music"
    ASSEMBLING = "assembling"
    RENDERED = "rendered"
    UPLOADED = "uploaded"
    FAILED = "failed"


class SceneType(str, enum.Enum):
    TXT2VID = "txt2vid"
    IMG2VID = "img2vid"
    STILL_PAN = "still_pan"       # Ken Burns effect on still image
    NARRATION_ONLY = "narration"  # black/gradient screen with narration


class SceneStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    GENERATING = "generating"
    GENERATED = "generated"
    APPROVED = "approved"
    FAILED = "failed"
    SKIPPED = "skipped"


class GenerationStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SUPERSEDED = "superseded"  # replaced by a newer version


class RenderStatus(str, enum.Enum):
    QUEUED = "queued"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Base ───────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def generate_uuid() -> str:
    return str(uuid.uuid4())


# ── Models ─────────────────────────────────────────────────────────────────

class Channel(Base):
    __tablename__ = "channels"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False, unique=True)
    slug = Column(String, nullable=False, unique=True)  # "little-fairy-dreams"
    profile_path = Column(String)       # path to YAML config
    system_prompt = Column(Text)        # preloaded LLM context
    default_loras = Column(JSON, default=list)
    still_ratio = Column(Float, default=0.4)
    target_resolution = Column(String, default="1080p")
    made_for_kids = Column(Boolean, default=False)
    default_video_model = Column(String, default="ltx-2.3")
    default_image_model = Column(String, default="sdxl")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    projects = relationship("Project", back_populates="channel")


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False)
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)
    duration_target = Column(Integer, nullable=False)  # seconds
    context = Column(Text)              # user-provided context/notes
    script_raw = Column(Text)           # raw LLM output
    status = Column(SAEnum(ProjectStatus), default=ProjectStatus.DRAFT)
    total_scenes = Column(Integer, default=0)
    completed_scenes = Column(Integer, default=0)
    output_path = Column(String)        # final rendered video
    thumbnail_path = Column(String)
    error_log = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    channel = relationship("Channel", back_populates="projects")
    scenes = relationship("Scene", back_populates="project",
                          order_by="Scene.scene_number")
    music_tracks = relationship("MusicTrack", back_populates="project")
    render_jobs = relationship("RenderJob", back_populates="project")


class Scene(Base):
    __tablename__ = "scenes"

    id = Column(String, primary_key=True, default=generate_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    scene_number = Column(Integer, nullable=False)
    scene_type = Column(SAEnum(SceneType), nullable=False)
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text, default="")
    duration = Column(Float, default=5.0)       # seconds
    camera_motion = Column(String, default="static")
    lora_ids = Column(JSON, default=list)       # list of LoRA file paths
    lora_weights = Column(JSON, default=list)   # matching weights
    status = Column(SAEnum(SceneStatus), default=SceneStatus.PENDING)
    active_generation_id = Column(String, nullable=True)  # FK set after generation
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    director_notes = Column(JSON, default=dict) # style cues, transition hints
    narration_text = Column(Text)               # if this scene has narration
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="scenes")
    generations = relationship("Generation", back_populates="scene",
                               order_by="Generation.version")

    @property
    def active_generation(self) -> Optional["Generation"]:
        if not self.active_generation_id:
            return None
        for g in self.generations:
            if g.id == self.active_generation_id:
                return g
        return None

    @property
    def latest_generation(self) -> Optional["Generation"]:
        if self.generations:
            return self.generations[-1]
        return None


class Generation(Base):
    __tablename__ = "generations"

    id = Column(String, primary_key=True, default=generate_uuid)
    scene_id = Column(String, ForeignKey("scenes.id"), nullable=False)
    version = Column(Integer, nullable=False)
    model_used = Column(String, nullable=False)   # "ltx-2.3", "sdxl", "wan-14b"
    output_path = Column(String)                  # raw clip/image path
    upscaled_path = Column(String)                # HD version path
    thumbnail_path = Column(String)
    prompt_used = Column(Text)                    # actual prompt (with channel prefix etc.)
    negative_prompt_used = Column(Text)
    seed = Column(Integer)
    parameters = Column(JSON, default=dict)       # steps, cfg, resolution, fps, etc.
    quality_score = Column(Float, nullable=True)  # 0-100
    status = Column(SAEnum(GenerationStatus), default=GenerationStatus.QUEUED)
    error_log = Column(Text)
    generation_time_sec = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    scene = relationship("Scene", back_populates="generations")


class MusicTrack(Base):
    __tablename__ = "music_tracks"

    id = Column(String, primary_key=True, default=generate_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    style_prompt = Column(Text)
    output_path = Column(String)
    duration = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="music_tracks")


class RenderJob(Base):
    __tablename__ = "render_jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    resolution = Column(String, default="1080p")
    output_path = Column(String)
    status = Column(SAEnum(RenderStatus), default=RenderStatus.QUEUED)
    progress_pct = Column(Float, default=0.0)
    render_settings = Column(JSON, default=dict)  # transitions, audio mix levels
    error_log = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="render_jobs")


class LoRA(Base):
    """Registry of available LoRA weights."""
    __tablename__ = "loras"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    model_type = Column(String, nullable=False)   # "sdxl", "ltx", "wan"
    trigger_words = Column(JSON, default=list)
    description = Column(Text)
    default_weight = Column(Float, default=0.7)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Database Setup ─────────────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def init_db(db_path: str = "D:/ai-director/ai_director.db"):
    global _engine, _SessionLocal
    _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)
    return _engine


def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def get_db():
    """FastAPI dependency."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()
