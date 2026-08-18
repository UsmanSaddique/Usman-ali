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
    GENERATED = "generated"
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
    TEMPLATE = "template"         # headless-browser rendered (diagram/code/map/title_card)
    USER_ASSET = "user_asset"     # user-provided file from assets_in/


class SceneStatus(str, enum.Enum):
    DRAFT = "draft"
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
    content_archetype = Column(String, nullable=True)  # archetype id (archetypes/*.yaml); NULL=legacy
    orientation = Column(String, default="landscape")  # landscape(16:9) | vertical(9:16 shorts/reels) | square(1:1)
    default_video_model = Column(String, default="ltx-2.3")
    default_image_model = Column(String, default="sdxl")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    projects = relationship("Project", back_populates="channel")


class SafetyVerdict(str, enum.Enum):
    PASS = "pass"
    REVISE = "revise"
    BLOCK = "block"
    OVERRIDE = "override"   # human signed off despite a non-pass verdict


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False)
    channel_id = Column(String, ForeignKey("channels.id"), nullable=False)
    project_type = Column(String, default="song")   # "song" | "narration" (internal lane; chosen by archetype)
    content_archetype = Column(String, nullable=True)  # archetype id override; NULL=inherit channel
    orientation = Column(String, nullable=True)  # override channel orientation; NULL=inherit (landscape|vertical|square)
    reviewed = Column(Boolean, default=False)  # human script sign-off (Tier-2 required-review gate); only the approve endpoint sets it
    duration_target = Column(Integer, nullable=False)  # seconds
    context = Column(Text)              # user-provided context/notes
    narration_script = Column(Text)     # narration mode: JSON {chapters:[{beats:[...]}], seo:{...}}
    narration_voice = Column(String)    # kokoro voice id (e.g. "am_michael") or engine override
    narration_audio_path = Column(String)  # master narration WAV after TTS+alignment
    lyrics = Column(Text)               # song lyrics — drive music vocals + scene timing
    lyrics_urdu = Column(Text)          # Hindi/Urdu lyric version — same video, second soundtrack
    music_style = Column(Text)          # style tags for the music engine
    music_model = Column(String, default="auto")   # auto|sft|turbo|heartmula|ace1
    upscale_inline = Column(Boolean, default=True)  # upscale each clip right after generation
    # How video is generated: "clips" = one 5-6s ComfyUI job per scene;
    # "ltx_director" = multi-director single workflow (20-30s per director
    # node, native audio, reference stills attached per segment)
    video_engine = Column(String, default="clips")
    num_scenes_target = Column(Integer, nullable=True) # user-customized target scene count
    video_model = Column(String, default="LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf")
    default_lora_ids = Column(JSON, default=list)      # ["lora_file.safetensors", ...]
    default_lora_weights = Column(JSON, default=list)  # [0.8, 0.7, ...]
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
    prompt = Column(Text, nullable=False)        # IMAGE prompt: the opening-frame composition (still gen)
    motion_prompt = Column(Text, nullable=True)  # VIDEO prompt: what MOVES/happens over the clip (authored, distinct from the still)
    negative_prompt = Column(Text, default="")
    duration = Column(Float, default=5.0)       # seconds
    camera_motion = Column(String, default="static")
    video_model = Column(String, nullable=True) # per-scene model
    lora_ids = Column(JSON, default=list)       # list of LoRA file paths
    lora_weights = Column(JSON, default=list)   # matching weights
    status = Column(SAEnum(SceneStatus), default=SceneStatus.PENDING)
    active_generation_id = Column(String, nullable=True)  # FK set after generation
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    director_notes = Column(JSON, default=dict) # style cues, transition hints
    narration_text = Column(Text)               # if this scene has narration
    narration_start = Column(Float, nullable=True)  # narration mode: exact beat start (s) in master WAV
    narration_end = Column(Float, nullable=True)    # narration mode: exact beat end (s)
    visual_type = Column(String, nullable=True)     # broll|still|diagram|code|map|title_card
    sfx_prompt = Column(Text, nullable=True)        # sound-design intent for this beat
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


class SafetyReport(Base):
    """YT-policy safety gate result. One row per gate run, per project.
    The LATEST row is authoritative: start-generation refuses to run unless
    its verdict is pass/override (override = recorded human sign-off)."""
    __tablename__ = "safety_reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    verdict = Column(SAEnum(SafetyVerdict), nullable=False)
    issues = Column(JSON, default=list)        # [{severity, category, where, detail, suggestion}]
    checked_fields = Column(JSON, default=dict)  # what was scanned (lyrics/narration/prompts/seo)
    auto_revisions = Column(JSON, default=list)  # [{where, before, after}] applied by the gate
    override_note = Column(Text)               # human reason when verdict=override
    llm_used = Column(Boolean, default=False)  # LLM critic layer ran (vs rules-only)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


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


def init_db(db_path: str = None):
    global _engine, _SessionLocal
    if db_path is None:
        # Fall back to the configured DB path rather than a hardcoded drive,
        # so standalone scripts/tests that call get_session() hit the same file
        # the server uses.
        from app.config import settings
        db_path = str(settings.paths.database)
    _engine = create_engine(
        f"sqlite:///{db_path}", echo=False,
        # A pipeline thread holding a write while an API request reads used to
        # raise "database is locked"; wait instead of failing.
        connect_args={"timeout": 30},
    )

    from sqlalchemy import event

    @event.listens_for(_engine, "connect")
    def _sqlite_crash_safety(dbapi_conn, _record):
        # WAL survives process kills / power loss without corrupting the DB,
        # and lets readers proceed while a pipeline thread writes checkpoints.
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)
    _migrate(_engine)
    return _engine


def _migrate(engine):
    """Add columns missing from older DB schemas."""
    import sqlalchemy
    with engine.connect() as conn:
        for table, col, col_type, default in [
            ("projects", "video_model", "VARCHAR", "'LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf'"),
            ("projects", "default_lora_ids", "JSON", "'[]'"),
            ("projects", "default_lora_weights", "JSON", "'[]'"),
            ("projects", "lyrics", "TEXT", "NULL"),
            ("projects", "music_style", "TEXT", "NULL"),
            ("projects", "music_model", "VARCHAR", "'auto'"),
            ("projects", "upscale_inline", "BOOLEAN", "1"),
            ("projects", "video_engine", "VARCHAR", "'clips'"),
            ("projects", "lyrics_urdu", "TEXT", "NULL"),
            ("projects", "project_type", "VARCHAR", "'song'"),
            ("projects", "narration_script", "TEXT", "NULL"),
            ("projects", "narration_voice", "VARCHAR", "NULL"),
            ("projects", "narration_audio_path", "VARCHAR", "NULL"),
            ("projects", "content_archetype", "VARCHAR", "NULL"),
            ("projects", "orientation", "VARCHAR", "NULL"),
            ("projects", "reviewed", "BOOLEAN", "0"),
            ("channels", "content_archetype", "VARCHAR", "NULL"),
            ("channels", "orientation", "VARCHAR", "'landscape'"),
            ("scenes", "video_model", "VARCHAR", "NULL"),
            ("scenes", "motion_prompt", "TEXT", "NULL"),
            ("scenes", "narration_start", "FLOAT", "NULL"),
            ("scenes", "narration_end", "FLOAT", "NULL"),
            ("scenes", "visual_type", "VARCHAR", "NULL"),
            ("scenes", "sfx_prompt", "TEXT", "NULL"),
        ]:
            try:
                conn.execute(sqlalchemy.text(
                    f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT {default}"
                ))
                conn.commit()
            except Exception:
                pass


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
