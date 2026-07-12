"""
AI Director — Pipeline Orchestrator
Coordinates the full video generation workflow across all services.
Manages state transitions, error handling, retries, and progress reporting.
"""
import os
import time
import logging
import asyncio
import threading
from enum import Enum
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

from app.config import Settings
from app.database import (
    get_session, Project, Scene, Generation, MusicTrack, RenderJob,
    ProjectStatus, SceneType, SceneStatus, GenerationStatus, RenderStatus,
)
from app.services.model_manager import ModelManager, ModelType
from app.services.director import DirectorService, VideoScript
from app.services.image_gen import ImageGenService
from app.services.video_gen import VideoGenService
from app.services.upscaler import UpscalerService
from app.services.assembler import AssemblerService, ClipEntry
from app.services.music_gen import MusicGenService
from app.services.tts import TTSService

logger = logging.getLogger(__name__)


# ── Progress Reporting ─────────────────────────────────────────────────────

class PipelinePhase(str, Enum):
    IDLE = "idle"
    SCRIPTING = "scripting"
    TTS = "tts"
    GENERATING = "generating"
    UPSCALING = "upscaling"
    MUSIC = "music"
    ASSEMBLING = "assembling"
    DONE = "done"
    ERROR = "error"


@dataclass
class PipelineProgress:
    phase: PipelinePhase = PipelinePhase.IDLE
    project_id: str = ""
    current_scene: int = 0
    total_scenes: int = 0
    scene_status: str = ""
    percent: float = 0.0
    eta_seconds: float = 0.0
    message: str = ""
    error: Optional[str] = None


# ── Pipeline Orchestrator ──────────────────────────────────────────────────

class PipelineOrchestrator:
    """
    The master conductor. Drives the entire video generation pipeline.

    Workflow:
    1. generate_script() → Qwen plans all scenes → user reviews
    2. start_generation() → generate assets scene by scene
    3. start_upscale() → upscale all approved clips
    4. generate_music() → ACE-Step background track
    5. render() → FFmpeg assembly

    Each phase updates the database and broadcasts progress via callbacks.
    """

    def __init__(self, config: Settings, model_manager: ModelManager):
        self.config = config
        self.manager = model_manager
        self.director = DirectorService(model_manager, config)
        self.image_gen = ImageGenService(model_manager, config)
        self.video_gen = VideoGenService(model_manager, config)
        self.upscaler = UpscalerService(model_manager, config)
        self.assembler = AssemblerService(config)
        self.music_gen = MusicGenService(model_manager, config)
        self.tts = TTSService(config)

        self._progress = PipelineProgress()
        self._progress_callbacks: list[Callable[[PipelineProgress], None]] = []
        self._cancel_requested = False
        self._pause_requested = False
        # QA accumulator — filled across phases, written as qa_report.json at render
        self._qa_notes: dict = {"scenes": []}
        # Exclusivity: only ONE heavy phase (generation/upscale/render) may run
        # at a time. Concurrent starts (e.g. bulk-clicking 50 per-scene upscales)
        # used to spawn parallel threads that fought over the GPU and crashed.
        # RLock so run_full_auto can call the phase methods it wraps.
        self._phase_lock = threading.RLock()
        self._lock_depth = 0
        self._busy_phase: Optional[str] = None

    @property
    def busy_phase(self) -> Optional[str]:
        """Name of the running exclusive phase, or None when idle."""
        return self._busy_phase

    @contextmanager
    def _exclusive(self, phase: str):
        """Guard a heavy phase. Raises PipelineBusy immediately (no blocking)
        if another thread is mid-phase, instead of letting two GPU pipelines
        trample each other."""
        if not self._phase_lock.acquire(blocking=False):
            raise PipelineBusy(
                f"Pipeline is busy ({self._busy_phase or 'unknown phase'}) — "
                f"wait for it to finish or cancel it first")
        self._lock_depth += 1
        if self._lock_depth == 1:
            self._busy_phase = phase
        try:
            yield
        finally:
            self._lock_depth -= 1
            if self._lock_depth == 0:
                self._busy_phase = None
            self._phase_lock.release()

    # ── Progress Management ────────────────────────────────────────────

    def on_progress(self, callback: Callable[[PipelineProgress], None]):
        """Register a progress callback (e.g., WebSocket broadcaster)."""
        self._progress_callbacks.append(callback)

    def _emit_progress(self, **kwargs):
        """Update and broadcast progress."""
        for k, v in kwargs.items():
            setattr(self._progress, k, v)
        for cb in self._progress_callbacks:
            try:
                cb(self._progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def cancel(self):
        """Request cancellation of the current pipeline run."""
        self._cancel_requested = True
        logger.info("[Pipeline] Cancellation requested")

    def _check_cancel(self):
        if self._cancel_requested:
            self._cancel_requested = False
            raise PipelineCancelled("Pipeline cancelled by user")

    def request_pause(self):
        """Request a graceful pause — the pipeline stops after the current item
        (scene / song version) WITHOUT marking the project failed. All finished
        work is kept; the same step button resumes from where it stopped."""
        self._pause_requested = True
        logger.info("[Pipeline] Pause requested — stopping after the current item")

    def _check_pause(self):
        if self._pause_requested:
            self._pause_requested = False
            raise PipelinePaused("Pipeline paused by user")

    # ── Phase 1: Script Generation ─────────────────────────────────────

    def generate_script(self, project_id: str) -> VideoScript:
        """
        Load Qwen → generate scene breakdown → save to DB → unload Qwen.
        Returns the script for user review.
        """
        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            channel = project.channel

            self._emit_progress(
                phase=PipelinePhase.SCRIPTING,
                project_id=project_id,
                message="Director is planning your video...",
                percent=0.0,
            )

            # Get available LoRAs for this channel
            from app.database import LoRA
            loras = session.query(LoRA).filter(
                LoRA.model_type.in_(["sdxl", "ltx"])
            ).all()
            lora_dicts = [
                {"name": l.name, "description": l.description,
                 "trigger_words": l.trigger_words}
                for l in loras
            ]

            # Generate script
            script = self.director.generate_script(
                title=project.title,
                duration=project.duration_target,
                context=project.context or "",
                channel_slug=channel.slug,
                available_loras=lora_dicts if lora_dicts else None,
                num_scenes=project.num_scenes_target,
            )

            # Unload LLM to free VRAM
            self.manager.unload()

            # Save scenes to database
            # First clear any existing scenes
            session.query(Scene).filter(Scene.project_id == project_id).delete()

            for sp in script.scenes:
                scene = Scene(
                    project_id=project_id,
                    scene_number=sp.scene_number,
                    scene_type=SceneType(sp.scene_type),
                    prompt=sp.prompt,
                    negative_prompt=sp.negative_prompt,
                    duration=sp.duration,
                    camera_motion=sp.camera_motion,
                    lora_ids=sp.loras,
                    lora_weights=sp.lora_weights,
                    narration_text=sp.narration_text,
                    director_notes=sp.director_notes,
                    status=SceneStatus.PENDING,
                )
                session.add(scene)

            import json
            project.script_raw = json.dumps({
                "music_style": script.music_style,
                "music_mood": script.music_mood,
                "thumbnail_prompt": script.thumbnail_prompt,
            })
            project.total_scenes = len(script.scenes)
            project.status = ProjectStatus.SCRIPTED
            session.commit()

            self._emit_progress(
                phase=PipelinePhase.IDLE,
                message=f"Script ready: {len(script.scenes)} scenes",
                percent=100.0,
            )

            return script

        except PipelineCancelled as ce:
            logger.info(f"[Pipeline] Scripting cancelled: {ce}")
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = "Scripting cancelled by user"
                    session.commit()
            except Exception as dbe:
                logger.error(f"Failed to update project status on cancel: {dbe}")
            
            self._emit_progress(
                phase=PipelinePhase.ERROR,
                error="Cancelled",
                message="Scripting cancelled by user",
            )
            raise

        except Exception as e:
            logger.error(f"[Pipeline] Scripting failed: {e}", exc_info=True)
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = str(e)
                    session.commit()
            except Exception as dbe:
                logger.error(f"Failed to update project status on error: {dbe}")
            
            self._emit_progress(
                phase=PipelinePhase.ERROR,
                error=str(e),
                message=f"Scripting failed: {e}",
            )
            raise

        finally:
            self.manager.unload()
            session.close()

    # ── Phase 1b: Scenes from Lyrics (no LLM — instant, beat-synced) ────

    def generate_scenes_from_lyrics(self, project_id: str, num_clips: Optional[int] = None) -> int:
        """Build the scene list directly from the project's lyrics: parse into
        timed segments, one channel-styled img2vid scene per segment. Instant
        (no LLM load) and the scene timing follows the song structure."""
        from app.services.lyrics_parser import parse_lyrics
        from app.services import lyric_scenes

        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            lyrics = (project.lyrics or "").strip()
            if not lyrics:
                raise ValueError("Project has no lyrics — paste lyrics or use the AI Director script instead")

            duration = float(project.duration_target)
            n = num_clips or project.num_scenes_target or max(1, int(duration // 5))
            segments = parse_lyrics(
                lyrics, duration,
                max_segments=n,
                target_segment_sec=duration / n,
            )
            profile = self.director.load_channel_profile(project.channel.slug) or {}
            prompts = lyric_scenes.build_prompts(segments, profile, project_id)

            session.query(Scene).filter(Scene.project_id == project_id).delete()
            for p in prompts:
                session.add(Scene(
                    project_id=project_id,
                    scene_number=p["segment_index"] + 1,
                    scene_type=SceneType.IMG2VID,
                    prompt=p["prompt"],
                    negative_prompt=p["negative_prompt"],
                    duration=p["duration"],
                    camera_motion=p["camera_motion"],
                    narration_text="",   # the song carries the audio
                    status=SceneStatus.PENDING,
                ))
            project.total_scenes = len(prompts)
            project.completed_scenes = 0
            project.status = ProjectStatus.SCRIPTED
            session.commit()
            logger.info(f"[Pipeline] {len(prompts)} scenes built from lyrics for {project_id}")
            return len(prompts)
        finally:
            session.close()

    # ── Phase 2: TTS Generation ──────────────────────────────────

    def generate_tts(self, project_id: str) -> Optional[str]:
        """
        Generate narration audio for all scenes with narration_text.
        Returns the combined narration audio path, or None if no narration.
        Gracefully returns None if TTS server is unavailable.
        """
        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            scenes = session.query(Scene).filter(
                Scene.project_id == project_id
            ).order_by(Scene.scene_number).all()

            # Collect scene dicts with narration
            scene_dicts = [
                {"scene_number": s.scene_number, "narration_text": s.narration_text or ""}
                for s in scenes
            ]

            has_narration = any(d["narration_text"].strip() for d in scene_dicts)
            if not has_narration:
                logger.info("[Pipeline] No narration text in any scene, skipping TTS")
                return None

            # TTS is now LOCAL (Meta MMS-TTS) — no server needed. Pick the voice
            # language from the channel profile (English/Urdu/Hindi/Roman Urdu).
            tts_language = "english"
            try:
                profile = self.director.load_channel_profile(project.channel.slug)
                tts_language = (profile or {}).get("language", "english")
            except Exception:
                pass

            self._emit_progress(
                phase=PipelinePhase.TTS,
                project_id=project_id,
                message="Generating narration audio...",
                percent=0.0,
            )

            project_dir = self.config.paths.projects_dir / project_id
            combined_path, segments = self.tts.generate_full_narration(
                scenes=scene_dicts,
                output_dir=str(project_dir),
                language=tts_language,
            )

            self._emit_progress(
                phase=PipelinePhase.IDLE,
                message=f"Narration ready: {len(segments)} segments",
                percent=100.0,
            )

            return combined_path

        except Exception as e:
            logger.warning(f"[Pipeline] TTS generation failed (non-fatal): {e}")
            return None

        finally:
            session.close()

    # ── Phase 3: Asset Generation ──────────────────────────────────────

    def start_generation(self, project_id: str, scene_ids: Optional[list[str]] = None, width: Optional[int] = None, height: Optional[int] = None, batch: bool = False, upscale_inline: bool = False):
        with self._exclusive("generation"):
            return self._start_generation_impl(
                project_id, scene_ids=scene_ids, width=width, height=height,
                batch=batch, upscale_inline=upscale_inline)

    def _start_generation_impl(self, project_id: str, scene_ids: Optional[list[str]] = None, width: Optional[int] = None, height: Optional[int] = None, batch: bool = False, upscale_inline: bool = False):
        """
        Generate all scene assets one by one.
        If scene_ids is provided, only generate those specific scenes.
        `batch=True` keeps the video model resident across scenes (skips the
        per-scene local-model unload) so a same-model run (e.g. LTX-22B txt2vid)
        doesn't pay an avoidable reload between scenes.
        `upscale_inline=True` upscales each clip to the channel target right
        after it passes QA, so every finished scene is immediately final.
        """
        self._is_paused = False
        self._cancel_flag = False
        self._batch_mode = batch
        self._video_model_loaded = False
        # No explicit resolution -> use 832x480, NOT the model-family default
        # (1152x640): with the Gemma-12B text encoder resident, LTX-22B only
        # partially fits at 1152x640 and crawls at 15-72s/step (~10min/clip)
        # instead of ~5s/step. 832x480 is measured VRAM-safe on the 16GB card.
        if width is None:
            width = 832
        if height is None:
            height = 480
        if batch:
            logger.info("[Pipeline] Batch mode: keeping video model resident across scenes")
        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            query = session.query(Scene).filter(Scene.project_id == project_id)
            if scene_ids:
                query = query.filter(Scene.id.in_(scene_ids))
            else:
                query = query.filter(Scene.status.in_([SceneStatus.PENDING, SceneStatus.FAILED]))
            
            scenes = query.order_by(Scene.scene_number).all()

            # Generations orphaned at RUNNING by a dead run (server restart
            # mid-generation) stay RUNNING forever and make the UI show scenes
            # as busy/ungenerated. A new run means none of them are live —
            # mark them FAILED so the DB reflects reality.
            orphaned = session.query(Generation).join(Scene).filter(
                Scene.project_id == project_id,
                Generation.status == GenerationStatus.RUNNING,
            ).all()
            for og in orphaned:
                og.status = GenerationStatus.FAILED
            if orphaned:
                logger.info(f"[Pipeline] Marked {len(orphaned)} orphaned RUNNING generations as FAILED")

            total = len(scenes)
            project.status = ProjectStatus.GENERATING
            session.commit()

            # Create project output directories
            project_dir = self.config.paths.projects_dir / project_id
            clips_dir = project_dir / "clips"
            images_dir = project_dir / "images"
            clips_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)

            scene_times = []

            # Batch mode: free VRAM ONCE up front (e.g. evict the ACE-Step music
            # model) so the video model has the whole 16GB to stay resident across
            # scenes instead of being evicted/reloaded between them.
            if batch:
                try:
                    from app.services.comfyui_client import ComfyUIClient
                    ComfyUIClient().free_vram()
                except Exception as e:
                    logger.warning(f"[Pipeline] batch free_vram failed: {e}")

            # ── Pre-generate stills for IMG2VID scenes ──────────────────
            # Generate ALL stills first with one image model load, then free
            # it so the video model stays resident across scenes (batch mode).
            # Eliminates image↔video model thrashing (30-60s/scene on 16GB).
            self._pregenerated_stills = {}
            img2vid_scenes_to_prerender = [
                s for s in scenes if s.scene_type == SceneType.IMG2VID
            ]
            if img2vid_scenes_to_prerender:
                self._pre_generate_stills(
                    img2vid_scenes_to_prerender, images_dir, session,
                    width=width, height=height,
                )

            for i, scene in enumerate(scenes):
                self._check_cancel()
                self._check_pause()

                self._emit_progress(
                    phase=PipelinePhase.GENERATING,
                    project_id=project_id,
                    current_scene=scene.scene_number,
                    total_scenes=project.total_scenes,
                    scene_status=f"Generating scene {scene.scene_number}",
                    percent=(i / total) * 100,
                    message=f"Scene {scene.scene_number}/{project.total_scenes}: {scene.scene_type.value}",
                )

                t0 = time.time()

                try:
                    generation = self._generate_scene(
                        scene, clips_dir, images_dir, session, width=width, height=height
                    )

                    # QA gate: reject truncated/unreadable/static clips so the
                    # retry path kicks in instead of shipping dead footage.
                    self._qa_gate(scene, generation)

                    # Inline upscale: finish the clip completely before moving on
                    if upscale_inline and generation.output_path:
                        try:
                            res_map = {"1080p": (1920, 1080), "2k": (2560, 1440), "4k": (3840, 2160)}
                            tw, th = res_map.get(project.channel.target_resolution, (1920, 1080))
                            up = self.upscaler.upscale_video(
                                input_path=generation.output_path,
                                target_width=tw, target_height=th,
                                ffmpeg_bin=self.config.paths.ffmpeg_bin,
                            )
                            generation.upscaled_path = up.output_path
                            logger.info(f"[Pipeline] Scene {scene.scene_number} upscaled inline -> {tw}x{th}")
                            # The ESRGAN upscale calls ComfyUI /free, evicting the
                            # resident video model — force a clean free+reload for
                            # the next scene, otherwise LTX reloads under memory
                            # pressure ("loaded partially") and crawls.
                            self._video_model_loaded = False
                        except Exception as up_err:
                            logger.warning(f"[Pipeline] Inline upscale failed for scene {scene.scene_number} (keeping raw): {up_err}")

                    scene.status = SceneStatus.GENERATED
                    scene.active_generation_id = generation.id
                    project.completed_scenes = i + 1

                except Exception as e:
                    logger.error(f"[Pipeline] Scene {scene.scene_number} failed: {e}")
                    scene.status = SceneStatus.FAILED
                    err = str(e)

                    # Use the whole retry budget within this run (each attempt
                    # varies seed and, for static clips, img2vid strength).
                    while scene.retry_count < scene.max_retries:
                        self._check_cancel()
                        scene.retry_count += 1
                        try:
                            self._retry_scene_direct(scene, err, clips_dir, images_dir, session, width=width, height=height)
                            project.completed_scenes = i + 1
                            break
                        except Exception as retry_err:
                            logger.error(
                                f"[Pipeline] Retry {scene.retry_count}/{scene.max_retries} "
                                f"for scene {scene.scene_number} failed: {retry_err}")
                            err = str(retry_err)

                elapsed = time.time() - t0
                scene_times.append(elapsed)
                session.commit()

                # Self-heal VRAM squeeze: a healthy img2vid clip takes 60-90s
                # (832x480, 96 frames). A much slower scene means the resident
                # video model got squeezed into partial-load mode (other apps
                # grabbed VRAM, fragmentation) — measured mid-batch: 178s then
                # 727s/clip until reload. Force a clean free+reload for the
                # next scene rather than letting the crawl persist.
                if batch and elapsed > 150 and self._video_model_loaded:
                    logger.warning(
                        f"[Pipeline] Scene {scene.scene_number} took {elapsed:.0f}s "
                        f"(expected <90s) — forcing VRAM free before next scene")
                    self._video_model_loaded = False

                # ETA calculation
                if scene_times:
                    avg_time = sum(scene_times) / len(scene_times)
                    remaining = total - (i + 1)
                    self._emit_progress(
                        eta_seconds=avg_time * remaining,
                    )

            # Unload whatever model is loaded
            self.manager.unload()

            # Update project status
            failed_count = session.query(Scene).filter(
                Scene.project_id == project_id,
                Scene.status == SceneStatus.FAILED,
            ).count()

            if failed_count == 0:
                project.status = ProjectStatus.GENERATED
                self._emit_progress(
                    phase=PipelinePhase.IDLE,
                    message=f"Generation complete. Ready for upscaling.",
                    percent=100.0,
                )
            else:
                project.status = ProjectStatus.FAILED
                self._emit_progress(
                    phase=PipelinePhase.ERROR,
                    error="Failed scenes",
                    message=f"Generation stopped. {failed_count} scenes failed.",
                )

            session.commit()

        except PipelinePaused:
            logger.info("[Pipeline] Generation paused by user — finished clips kept")
            try:
                project = session.query(Project).get(project_id)
                if project:
                    # resumable state: Generate All / resume picks up the
                    # remaining PENDING/FAILED scenes
                    project.status = ProjectStatus.SCRIPTED
                    project.error_log = None
                    session.commit()
            except Exception as dbe:
                logger.error(f"Failed to update project status on pause: {dbe}")
            self._emit_progress(
                phase=PipelinePhase.IDLE,
                message="Paused — finished clips are saved; resume anytime",
                percent=0.0,
            )
            raise

        except PipelineCancelled as ce:
            logger.info(f"[Pipeline] Generation cancelled: {ce}")
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = "Generation cancelled by user"
                    session.commit()
            except Exception as dbe:
                logger.error(f"Failed to update project status on cancel: {dbe}")

            self._emit_progress(
                phase=PipelinePhase.ERROR,
                error="Cancelled",
                message="Generation cancelled by user",
            )
            raise

        except Exception as e:
            logger.error(f"[Pipeline] Generation failed: {e}", exc_info=True)
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = str(e)
                    session.commit()
            except Exception as dbe:
                logger.error(f"Failed to update project status on error: {dbe}")
            
            self._emit_progress(
                phase=PipelinePhase.ERROR,
                error=str(e),
                message=f"Generation failed: {e}",
            )
            raise

        finally:
            self.manager.unload()
            session.close()

    def _find_user_audio(self, project_id: str, kind: str) -> Optional[str]:
        """Find a user-supplied audio file (your own song / voiceover).

        Drop a file named `music.*` (background song) or `voice.*` (narration)
        into the project's `audio_in/` folder, OR set `music_file:` / `voice_file:`
        in the channel YAML to point at a file in `assets_generated/music/`.
        Checked BEFORE auto-generation, so your file always wins.
        Supported: .mp3 .wav .m4a .aac .flac .ogg
        """
        exts = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")
        # 1. per-project drop-in: projects/<id>/audio_in/{music|voice}.*
        audio_in = self.config.paths.projects_dir / project_id / "audio_in"
        if audio_in.exists():
            for f in sorted(audio_in.iterdir()):
                if f.is_file() and f.stem.lower() == kind and f.suffix.lower() in exts:
                    return str(f)
        # 2. channel-level default from the profile (music_file / voice_file)
        try:
            session = get_session()
            project = session.query(Project).get(project_id)
            slug = project.channel.slug if project and project.channel else None
            session.close()
            if slug:
                profile = self.director.load_channel_profile(slug)
                key = f"{kind}_file"
                ref = (profile or {}).get(key)
                if ref:
                    p = Path(ref)
                    if not p.is_absolute():
                        p = self.config.paths.assets_dir / "music" / ref
                    if p.exists():
                        return str(p)
        except Exception as e:
            logger.debug(f"[Pipeline] _find_user_audio channel lookup failed: {e}")
        return None

    def _get_project_loras(self, project) -> list[tuple[str, float]]:
        """Get LoRA list from project defaults as [(filename, weight), ...]."""
        ids = project.default_lora_ids or []
        weights = project.default_lora_weights or []
        result = []
        for i, lora_file in enumerate(ids):
            w = weights[i] if i < len(weights) else 0.8
            result.append((lora_file, w))
        return result

    def _pre_generate_stills(
        self, scenes: list, images_dir: Path, session,
        width=None, height=None,
    ):
        """Pre-generate all stills for IMG2VID scenes in one batch.

        Keeps the image model (ZImage/SDXL) loaded for ALL stills, then frees
        it once.  The subsequent video phase keeps LTX resident across scenes
        (batch mode) instead of thrashing image↔video models every scene.
        On a 16GB card this saves 30-60s per scene of model reload time.
        """
        import os
        import hashlib
        from app.services.image_gen import ImageResult

        ic = self.config.image
        image_engine = getattr(ic, "engine", "sdxl")
        sdxl_available = os.path.exists(str(ic.path))
        zimage_available = (
            self.config.paths.models_dir / "diffusion_models"
            / getattr(ic, "zimage_unet", "z_image_turbo_bf16.safetensors")
        ).exists()
        image_available = (
            (zimage_available or sdxl_available)
            if image_engine == "zimage" else sdxl_available
        )
        if not image_available:
            logger.info("[Pipeline] No image engine available, skipping still pre-generation")
            return

        logger.info(f"[Pipeline] ── Batch still phase: {len(scenes)} IMG2VID stills ──")
        self._emit_progress(
            phase=PipelinePhase.GENERATING,
            message=f"Generating {len(scenes)} stills (batch image phase)…",
        )

        for i, scene in enumerate(scenes):
            self._check_cancel()
            self._check_pause()

            project = scene.project
            scene_seed = int(hashlib.md5(project.id.encode()).hexdigest()[:7], 16)
            version = len(scene.generations) + 1
            still_path = str(images_dir / f"scene_{scene.scene_number:03d}_v{version}.png")

            # A user-pinned still (regenerated via the wizard) always wins
            pinned = (scene.director_notes or {}).get("pinned_still")
            if pinned and Path(pinned).exists():
                self._pregenerated_stills[scene.id] = ImageResult(
                    path=pinned, width=width or 0, height=height or 0,
                    seed=scene_seed, generation_time=0.0, prompt_used=scene.prompt,
                )
                logger.info(
                    f"[Pipeline] Still {i+1}/{len(scenes)} "
                    f"(scene {scene.scene_number}): using user-pinned still"
                )
                continue

            # Skip if still already exists from a previous run
            if Path(still_path).exists():
                self._pregenerated_stills[scene.id] = ImageResult(
                    path=still_path, width=width or 0, height=height or 0,
                    seed=scene_seed, generation_time=0.0, prompt_used=scene.prompt,
                )
                logger.info(
                    f"[Pipeline] Still {i+1}/{len(scenes)} "
                    f"(scene {scene.scene_number}): exists, reusing"
                )
                continue

            self._emit_progress(
                message=f"Generating still {i+1}/{len(scenes)} "
                        f"(scene {scene.scene_number})…",
                percent=(i / len(scenes)) * 30,  # stills = first 30% of progress
            )

            try:
                img_result = self.image_gen.generate(
                    prompt=scene.prompt,
                    negative_prompt=scene.negative_prompt,
                    output_path=still_path,
                    lora_paths=scene.lora_ids,
                    lora_weights=scene.lora_weights,
                    width=width,
                    height=height,
                    seed=scene_seed,
                )
                self._pregenerated_stills[scene.id] = img_result
                logger.info(
                    f"[Pipeline] Still {i+1}/{len(scenes)} "
                    f"(scene {scene.scene_number}): {img_result.generation_time:.1f}s"
                )
            except Exception as e:
                logger.warning(
                    f"[Pipeline] Still {i+1}/{len(scenes)} "
                    f"(scene {scene.scene_number}) failed: {e}  — will generate inline"
                )

        # Free the image model so the video phase has full VRAM
        self.manager.unload()
        try:
            from app.services.comfyui_client import ComfyUIClient
            client = ComfyUIClient()
            client.free_vram()
            time.sleep(2)
            client.free_vram()
        except Exception:
            pass

        n_ok = len(self._pregenerated_stills)
        logger.info(f"[Pipeline] ── Stills phase complete: {n_ok}/{len(scenes)} ──")

    def _generate_scene(
        self, scene: Scene, clips_dir: Path, images_dir: Path, session, width: Optional[int] = None, height: Optional[int] = None
    ) -> Generation:
        """Generate a single scene based on its type."""
        version = len(scene.generations) + 1
        gen = Generation(
            scene_id=scene.id,
            version=version,
            model_used="",
            prompt_used=scene.prompt,
            negative_prompt_used=scene.negative_prompt,
            status=GenerationStatus.RUNNING,
        )
        session.add(gen)
        session.flush()

        project = scene.project
        # Lock ONE seed per project so every clip shares the same visual "world"
        # (lighting, palette, character look). Random per-clip seeds were a major
        # cause of clips looking totally different scene to scene.
        import hashlib
        scene_seed = int(hashlib.md5(project.id.encode()).hexdigest()[:7], 16)
        # On retries, nudge the seed: a QA-rejected clip (e.g. "static, no
        # motion") regenerated with the identical seed produces the identical
        # rejected clip — each retry just burns another full generation.
        # For img2vid the still is already fixed, so the seed only varies the
        # motion sampling; the project's visual "world" is preserved.
        scene_seed += scene.retry_count
        # Prioritize scene's video model, then project's, then default
        video_model = scene.video_model or getattr(project, "video_model", None) or "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf"
        num_frames, fps, _ = self._plan_video_params(scene, video_model)
        project_loras = self._get_project_loras(project)

        # Per-scene LoRA overrides
        if scene.lora_ids:
            scene_loras = list(zip(
                scene.lora_ids,
                scene.lora_weights or [0.8] * len(scene.lora_ids),
            ))
        else:
            scene_loras = project_loras

        # Still-image engine availability: Z-Image-Turbo (primary) or SDXL fallback.
        ic = self.config.image
        image_engine = getattr(ic, "engine", "sdxl")
        sdxl_available = os.path.exists(str(ic.path))
        zimage_available = (
            self.config.paths.models_dir / "diffusion_models"
            / getattr(ic, "zimage_unet", "z_image_turbo_bf16.safetensors")
        ).exists()
        image_available = (zimage_available or sdxl_available) if image_engine == "zimage" else sdxl_available

        try:
            clear_vram = not (getattr(self, "_batch_mode", False) and getattr(self, "_video_model_loaded", False))

            if scene.scene_type == SceneType.TXT2VID:
                result = self.video_gen.txt2vid(
                    prompt=scene.prompt,
                    negative_prompt=scene.negative_prompt,
                    num_frames=num_frames,
                    fps=fps,
                    output_path=str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4"),
                    model_filename=video_model,
                    loras=scene_loras or None,
                    width=width,
                    height=height,
                    seed=scene_seed,
                    clear_vram_first=clear_vram,
                )
                self._video_model_loaded = True
                gen.model_used = result.model_used
                gen.output_path = result.path
                gen.seed = result.seed
                gen.generation_time_sec = result.generation_time
                gen.parameters = {
                    "width": result.width, "height": result.height,
                    "num_frames": result.num_frames, "fps": result.fps,
                }

            elif scene.scene_type == SceneType.IMG2VID:
                if not image_available:
                    logger.warning(f"[Pipeline] Scene {scene.scene_number}: no image engine available, falling back to txt2vid")
                    result = self.video_gen.txt2vid(
                        prompt=scene.prompt,
                        negative_prompt=scene.negative_prompt,
                        num_frames=num_frames,
                        fps=fps,
                        output_path=str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4"),
                        model_filename=video_model,
                        loras=scene_loras or None,
                        width=width,
                        height=height,
                        seed=scene_seed,
                        clear_vram_first=clear_vram,
                    )
                    self._video_model_loaded = True
                    gen.model_used = f"txt2vid-fallback:{result.model_used}"
                    gen.output_path = result.path
                    gen.seed = result.seed
                    gen.generation_time_sec = result.generation_time
                    gen.parameters = {
                        "width": result.width, "height": result.height,
                        "num_frames": result.num_frames, "fps": result.fps,
                    }
                else:
                    # A user-pinned still (regenerated via the wizard) always
                    # wins — it's the image the user approved for this scene.
                    pinned = (scene.director_notes or {}).get("pinned_still")
                    if pinned and os.path.exists(pinned):
                        from app.services.image_gen import ImageResult
                        pre_still = ImageResult(
                            path=pinned, width=0, height=0, seed=scene_seed,
                            generation_time=0.0, prompt_used=scene.prompt,
                        )
                        logger.info(f"[Pipeline] Scene {scene.scene_number}: animating user-pinned still")
                    else:
                        # Check for pre-generated still (batch image phase)
                        pre_still = getattr(self, '_pregenerated_stills', {}).get(scene.id)
                    if pre_still:
                        img_result = pre_still
                        logger.info(f"[Pipeline] Scene {scene.scene_number}: using pre-generated still")
                    else:
                        # Fallback: generate still inline (e.g. retry path,
                        # or pre-generation failed for this scene)
                        img_result = self.image_gen.generate(
                            prompt=scene.prompt,
                            negative_prompt=scene.negative_prompt,
                            output_path=str(images_dir / f"scene_{scene.scene_number:03d}_v{version}.png"),
                            lora_paths=scene.lora_ids,
                            lora_weights=scene.lora_weights,
                            width=width,
                            height=height,
                            seed=scene_seed,
                        )
                        self.manager.unload()

                    gen.thumbnail_path = img_result.path
                    session.commit()

                    # Only force VRAM free if we just generated a still inline
                    # (which loaded the image model into ComfyUI). If the still
                    # was pre-generated, the image model is already freed and the
                    # video model may be resident from a previous scene (batch).
                    need_vram_free = True if not pre_still else clear_vram
                    result = self.video_gen.img2vid(
                        prompt=scene.prompt,
                        image_path=img_result.path,
                        negative_prompt=scene.negative_prompt,
                        num_frames=num_frames,
                        fps=fps,
                        output_path=str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4"),
                        model_filename=video_model,
                        loras=scene_loras or None,
                        width=width,
                        height=height,
                        seed=scene_seed,
                        clear_vram_first=need_vram_free,
                    )
                    self._video_model_loaded = True
                    gen.model_used = f"{image_engine}+{result.model_used}"
                    gen.thumbnail_path = img_result.path
                    gen.generation_time_sec = (img_result.generation_time or 0) + result.generation_time

                gen.output_path = result.path
                gen.seed = result.seed
                if not gen.generation_time_sec:
                    gen.generation_time_sec = result.generation_time

            elif scene.scene_type == SceneType.STILL_PAN:
                if not image_available:
                    logger.warning(f"[Pipeline] Scene {scene.scene_number}: no image engine available, falling back to txt2vid for still_pan")
                    result = self.video_gen.txt2vid(
                        prompt=scene.prompt,
                        negative_prompt=scene.negative_prompt,
                        num_frames=num_frames,
                        fps=fps,
                        output_path=str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4"),
                        model_filename=video_model,
                        loras=scene_loras or None,
                        width=width,
                        height=height,
                        seed=scene_seed,
                        clear_vram_first=clear_vram,
                    )
                    self._video_model_loaded = True
                    gen.model_used = f"txt2vid-fallback:{result.model_used}"
                    gen.output_path = result.path
                    gen.seed = result.seed
                    gen.generation_time_sec = result.generation_time
                else:
                    img_result = self.image_gen.generate(
                        prompt=scene.prompt,
                        negative_prompt=scene.negative_prompt,
                        width=1280,
                        height=720,
                        steps=35,
                        output_path=str(images_dir / f"scene_{scene.scene_number:03d}_v{version}.png"),
                        lora_paths=scene.lora_ids,
                        lora_weights=scene.lora_weights,
                        seed=scene_seed,
                    )
                    self.manager.unload()

                    motion = scene.camera_motion or "zoom_in"
                    zoom_start, zoom_end = 1.0, 1.15
                    pan_x, pan_y = 0.0, 0.0
                    if motion == "zoom_out":
                        zoom_start, zoom_end = 1.15, 1.0
                    elif motion == "pan_left":
                        pan_x = -0.1
                    elif motion == "pan_right":
                        pan_x = 0.1
                    elif motion == "tilt_up":
                        pan_y = -0.08

                    result = self.video_gen.ken_burns(
                        image_path=img_result.path,
                        output_path=str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4"),
                        duration=scene.duration,
                        fps=self.config.video.default_fps,
                        zoom_start=zoom_start,
                        zoom_end=zoom_end,
                        pan_x=pan_x,
                        pan_y=pan_y,
                        ffmpeg_bin=self.config.paths.ffmpeg_bin,
                    )
                    gen.model_used = f"{image_engine}+kenburns"
                    gen.output_path = result.path
                    gen.seed = img_result.seed
                    gen.generation_time_sec = img_result.generation_time + result.generation_time
                    gen.thumbnail_path = img_result.path

            else:
                raise ValueError(f"Unknown scene type: {scene.scene_type}")

            gen.status = GenerationStatus.COMPLETED

        except Exception as e:
            import traceback
            gen.status = GenerationStatus.FAILED
            gen.error_log = traceback.format_exc()
            raise

        return gen

    def _plan_video_params(self, scene: Scene, video_model: str) -> tuple[int, int, float]:
        """Return (num_frames, fps, actual_clip_duration) for a scene.

        fps MUST match what the ComfyUI workflow encodes at (the model-family
        default), otherwise clips come out shorter than scene.duration:
        64 frames computed at config's 16fps but encoded at 24fps gave 2.67s
        clips for 4.0s scenes, which also corrupted crossfade offsets at
        assembly. Frames are capped at 97 — the measured VRAM-safe ceiling
        for LTX-22B at 832x480 on the 16GB card.
        """
        from app.services.comfyui_client import get_defaults_for_model
        fps = int(get_defaults_for_model(video_model).get("fps")
                  or self.config.video.default_fps)
        num_frames = min(int(float(scene.duration or 4.0) * fps), 97)
        # LTX only generates 8n+1 frame counts and silently rounds DOWN —
        # requesting 96 yielded 89 frames (3.71s clips for 4.0s scenes, a
        # 15s shortfall across a 50-clip video). Snap UP to the next 8n+1.
        num_frames = min(((num_frames - 1 + 7) // 8) * 8 + 1, 97)
        return num_frames, fps, num_frames / fps

    def _qa_gate(self, scene: Scene, generation) -> None:
        """Reject truncated/unreadable/static clips — raises RuntimeError so
        the retry path kicks in instead of shipping dead footage. Must run on
        EVERY generated clip, including retries."""
        from app.services import qa as qa_svc
        from dataclasses import asdict as _asdict
        if not (generation.output_path and str(generation.output_path).endswith(".mp4")):
            return
        video_model = (scene.video_model or getattr(scene.project, "video_model", None)
                       or "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf")
        _, _, planned_duration = self._plan_video_params(scene, video_model)
        # STILL_PAN clips are rendered by ffmpeg at the full scene duration;
        # txt2vid/img2vid clips at the (possibly frame-capped) planned duration.
        expected = (float(scene.duration or 0)
                    if scene.scene_type == SceneType.STILL_PAN else planned_duration)
        # LTX img2vid often yields very subtle motion (especially with strength≥0.6),
        # so the motion threshold is lowered from 3.0 to 1.0 for IMG2VID so QA stops
        # constantly rejecting valid clips.
        motion_thr = 0.0 if scene.scene_type == SceneType.STILL_PAN else (
            1.0 if scene.scene_type == SceneType.IMG2VID else 3.0)
        clip_qa = qa_svc.check_clip(
            self.config.paths.ffmpeg_bin,
            generation.output_path,
            expected,
            motion_threshold=motion_thr,
        )
        self._qa_notes.setdefault("scenes", []).append(
            {"scene": scene.scene_number, **_asdict(clip_qa)})
        if not clip_qa.ok:
            raise RuntimeError(
                f"QA rejected scene {scene.scene_number}: {', '.join(clip_qa.issues)}")

    def _retry_scene_direct(
        self, scene: Scene, error: str,
        clips_dir: Path, images_dir: Path, session,
        width: Optional[int] = None, height: Optional[int] = None
    ):
        """Retry generation directly without loading LLM for advice.
        Keeps the video/image model loaded instead of swapping to Qwen."""
        logger.info(
            f"[Pipeline] Retrying scene {scene.scene_number} directly "
            f"(attempt {scene.retry_count}, skipping LLM advice)"
        )

        # If the error is about a missing model (SDXL), switch scene type to txt2vid
        error_lower = error.lower()
        if "not found" in error_lower or "filenotfounderror" in error_lower:
            if scene.scene_type in (SceneType.STILL_PAN, SceneType.IMG2VID):
                logger.info(
                    f"[Pipeline] Scene {scene.scene_number}: model missing, "
                    f"switching {scene.scene_type.value} → txt2vid fallback"
                )
                scene.scene_type = SceneType.TXT2VID

        # QA rejected the clip as static? Retrying with identical settings
        # reproduces the same static clip. Lower img2vid strength for this
        # attempt — lower strength lets the clip depart further from the
        # still, i.e. more motion. (The seed is also nudged per retry in
        # _generate_scene.)
        vc = self.config.video
        orig_strength = getattr(vc, "img2vid_strength", 0.7)
        if "static clip" in error_lower and scene.scene_type == SceneType.IMG2VID:
            vc.img2vid_strength = max(0.5, round(orig_strength - 0.1 * scene.retry_count, 2))
            logger.info(
                f"[Pipeline] Scene {scene.scene_number}: static-clip retry, "
                f"img2vid_strength {orig_strength} -> {vc.img2vid_strength}"
            )
        try:
            generation = self._generate_scene(scene, clips_dir, images_dir, session, width=width, height=height)
            # Retried clips must pass the same QA gate as first attempts —
            # previously they were marked GENERATED unchecked.
            self._qa_gate(scene, generation)
        finally:
            vc.img2vid_strength = orig_strength
        scene.status = SceneStatus.GENERATED
        scene.active_generation_id = generation.id

    def _retry_scene(
        self, scene: Scene, error: str,
        clips_dir: Path, images_dir: Path, session
    ):
        """Use director LLM to adjust prompt and retry generation.
        Only called explicitly (e.g. from UI), not during automatic retries."""
        logger.info(f"[Pipeline] Retrying scene {scene.scene_number} with LLM advice (attempt {scene.retry_count})")

        self.manager.unload()
        advice = self.director.suggest_retry_strategy(
            scene_number=scene.scene_number,
            original_prompt=scene.prompt,
            error_log=error,
            retry_count=scene.retry_count,
        )
        self.manager.unload()

        scene.prompt = advice.get("new_prompt", scene.prompt)
        scene.negative_prompt = advice.get("negative_prompt", scene.negative_prompt)

        model_suggestion = advice.get("model_suggestion", "")
        if "kenburns" in model_suggestion.lower() and scene.scene_type != SceneType.STILL_PAN:
            scene.scene_type = SceneType.STILL_PAN

        generation = self._generate_scene(scene, clips_dir, images_dir, session)
        scene.status = SceneStatus.GENERATED
        scene.active_generation_id = generation.id

    # ── Phase 4: Upscale ───────────────────────────────────────────────

    def start_upscale(self, project_id: str):
        with self._exclusive("upscale"):
            return self._start_upscale_impl(project_id)

    def _start_upscale_impl(self, project_id: str):
        """Upscale all approved/generated clips."""
        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            scenes = session.query(Scene).filter(
                Scene.project_id == project_id,
                Scene.status.in_([SceneStatus.GENERATED, SceneStatus.APPROVED]),
            ).order_by(Scene.scene_number).all()

            project.status = ProjectStatus.UPSCALING
            session.commit()

            res_map = {"1080p": (1920, 1080), "2k": (2560, 1440)}
            channel = project.channel
            target_res = res_map.get(channel.target_resolution, (1920, 1080))

            for i, scene in enumerate(scenes):
                self._check_cancel()
                self._check_pause()

                gen = scene.active_generation
                if not gen or not gen.output_path:
                    continue

                # Resume-friendly: already upscaled (earlier run / inline) → skip
                if gen.upscaled_path and gen.upscaled_path != gen.output_path \
                        and Path(gen.upscaled_path).exists():
                    logger.info(f"[Pipeline] Scene {scene.scene_number} already upscaled, skipping")
                    continue

                self._emit_progress(
                    phase=PipelinePhase.UPSCALING,
                    current_scene=scene.scene_number,
                    total_scenes=len(scenes),
                    percent=(i / len(scenes)) * 100,
                    message=f"Upscaling scene {scene.scene_number}",
                )

                try:
                    result = self.upscaler.upscale_video(
                        input_path=gen.output_path,
                        target_width=target_res[0],
                        target_height=target_res[1],
                        ffmpeg_bin=self.config.paths.ffmpeg_bin,
                    )
                    gen.upscaled_path = result.output_path
                except Exception as e:
                    logger.error(f"[Pipeline] Upscale failed for scene {scene.scene_number}: {e}")
                    # Keep the raw clip — better than nothing
                    gen.upscaled_path = gen.output_path

                session.commit()

            self.manager.unload()

            # Mark project back to GENERATED when done so they can render
            project.status = ProjectStatus.GENERATED
            session.commit()

            self._emit_progress(
                phase=PipelinePhase.IDLE,
                message="Upscaling complete",
                percent=100.0,
            )

        except PipelinePaused:
            logger.info("[Pipeline] Upscale paused by user — finished upscales kept")
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.GENERATED  # resumable
                    project.error_log = None
                    session.commit()
            except Exception as dbe:
                logger.error(f"Failed to update project status on pause: {dbe}")
            self._emit_progress(
                phase=PipelinePhase.IDLE,
                message="Upscaling paused — finished upscales are saved; resume anytime",
                percent=0.0,
            )
            raise

        except PipelineCancelled as ce:
            logger.info(f"[Pipeline] Upscale cancelled: {ce}")
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = "Upscale cancelled by user"
                    session.commit()
            except Exception as dbe:
                logger.error(f"Failed to update project status on cancel: {dbe}")
            
            self._emit_progress(
                phase=PipelinePhase.ERROR,
                error="Cancelled",
                message="Upscale cancelled by user",
            )
            raise

        except Exception as e:
            logger.error(f"[Pipeline] Upscale failed: {e}", exc_info=True)
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = str(e)
                    session.commit()
            except Exception as dbe:
                logger.error(f"Failed to update project status on error: {dbe}")
            
            self._emit_progress(
                phase=PipelinePhase.ERROR,
                error=str(e),
                message=f"Upscale failed: {e}",
            )
            raise

        finally:
            self.manager.unload()
            session.close()

    def start_upscale_scene(self, scene_id: str):
        with self._exclusive("upscale (single scene)"):
            return self._start_upscale_scene_impl(scene_id)

    def _start_upscale_scene_impl(self, scene_id: str):
        """Upscale a single generated clip."""
        session = get_session()
        try:
            scene = session.query(Scene).get(scene_id)
            if not scene:
                raise ValueError(f"Scene {scene_id} not found")
            
            if scene.status not in [SceneStatus.GENERATED, SceneStatus.APPROVED]:
                raise ValueError(f"Scene {scene_id} is not generated yet")

            project = session.query(Project).get(scene.project_id)
            if not project:
                raise ValueError("Project not found")

            gen = scene.active_generation
            if not gen or not gen.output_path:
                raise ValueError(f"Scene {scene_id} has no output clip")

            # Skip if already upscaled
            if gen.upscaled_path and gen.upscaled_path != gen.output_path and Path(gen.upscaled_path).exists():
                logger.info(f"[Pipeline] Scene {scene.scene_number} already upscaled")
                return

            res_map = {"1080p": (1920, 1080), "2k": (2560, 1440)}
            channel = project.channel
            target_res = res_map.get(channel.target_resolution, (1920, 1080))

            self._emit_progress(
                phase=PipelinePhase.UPSCALING,
                current_scene=scene.scene_number,
                total_scenes=1,
                percent=0,
                message=f"Upscaling scene {scene.scene_number} (HD+)",
            )

            try:
                result = self.upscaler.upscale_video(
                    input_path=gen.output_path,
                    target_width=target_res[0],
                    target_height=target_res[1],
                    ffmpeg_bin=self.config.paths.ffmpeg_bin,
                )
                gen.upscaled_path = result.output_path
            except Exception as e:
                logger.error(f"[Pipeline] Upscale failed for scene {scene.scene_number}: {e}")
                gen.upscaled_path = gen.output_path
                raise

            session.commit()
            
            # Note: We do NOT unload self.manager here to avoid disrupting batch generation
            
            self._emit_progress(
                phase=PipelinePhase.IDLE,
                message=f"Scene {scene.scene_number} upscale complete",
                percent=100.0,
            )

        except Exception as e:
            logger.error(f"[Pipeline] Single upscale failed: {e}", exc_info=True)
            self._emit_progress(
                phase=PipelinePhase.ERROR,
                error=str(e),
                message=f"Upscale failed: {e}",
            )
            raise
        finally:
            session.close()

    # ── Phase 5: Music Generation ─────────────────────────────────

    def generate_music(
        self,
        project_id: str,
        style: Optional[str] = None,
        lyrics: Optional[str] = None,
        engine: Optional[str] = None,
        vocals: Optional[bool] = None,
    ) -> Optional[str]:
        """Generate the project's music track.

        Precedence for creative inputs: explicit args → project fields
        (lyrics/music_style/music_model from the wizard) → channel profile.
        A song with lyrics becomes a full vocal track that carries the video.
        """
        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            channel = project.channel

            self._emit_progress(
                phase=PipelinePhase.MUSIC,
                project_id=project_id,
                message="Generating background music...",
                percent=0.0,
            )

            # Load channel profile for music style
            profile = self.director.load_channel_profile(channel.slug)

            project_dir = self.config.paths.projects_dir / project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            music_path = str(project_dir / "music.wav")

            use_lyrics = lyrics if lyrics is not None else (project.lyrics or "")
            if vocals is False:
                use_lyrics = ""
            use_style = style or project.music_style or ""
            use_engine = engine or getattr(project, "music_model", None) or "auto"

            if use_lyrics or use_style:
                # Song mode / custom style — the music is the star
                music_cfg = (profile or {}).get("music", {})
                result = self.music_gen.generate(
                    style_prompt=use_style or music_cfg.get("style", "cheerful children's song"),
                    duration=int(project.duration_target) + 2,
                    lyrics=use_lyrics,
                    output_path=music_path,
                    instrumental=not bool(use_lyrics.strip()),
                    channel_profile=profile,
                    engine=use_engine,
                )
            elif profile:
                result = self.music_gen.generate_for_channel(
                    channel_profile=profile,
                    video_duration=project.duration_target,
                    output_path=music_path,
                )
            else:
                result = self.music_gen.generate(
                    style_prompt="gentle background music, instrumental",
                    duration=project.duration_target + 5,
                    output_path=music_path,
                    instrumental=True,
                )

            # Unload music model
            self.manager.unload()

            # Save to DB — new track supersedes previous ones
            session.query(MusicTrack).filter(
                MusicTrack.project_id == project_id,
                MusicTrack.is_active == True,
            ).update({"is_active": False})
            track = MusicTrack(
                project_id=project_id,
                style_prompt=result.style_prompt,
                output_path=result.path,
                duration=result.duration,
                is_active=True,
            )
            session.add(track)
            session.commit()

            self._emit_progress(
                phase=PipelinePhase.IDLE,
                message=f"Music ready: {result.duration:.0f}s track",
                percent=100.0,
            )

            return result.path

        finally:
            session.close()

    # Style tweaks applied on top of the base style for the song-audition
    # round — same lyrics every time, different musical treatment. v1 is
    # always the user's exact style, untouched.
    MUSIC_VARIANT_TWEAKS = [
        "",
        "upbeat tempo, bright ukulele and acoustic guitar, sunny feel",
        "playful bouncy rhythm, glockenspiel, hand claps, marimba",
        "sweet solo female vocal, warm and clear, soft pop arrangement",
        "warm male vocal, gentle folk acoustic, storyteller feel",
        "kids choir, sing-along clap-along energy, tambourine",
        "duet — lead vocal with children's choir answering each line",
        "gentle lullaby feel, soft piano, music box, dreamy and calm",
        "orchestral children's movie style, strings, flute, magical",
        "modern kids TV theme, catchy pop production, energetic chorus",
    ]

    def generate_music_variants(
        self,
        project_id: str,
        count: int = 10,
        style: Optional[str] = None,
        lyrics: Optional[str] = None,
        engine: Optional[str] = None,
        vocals: Optional[bool] = None,
        offset: int = 0,
    ) -> list[str]:
        """Song audition round: generate `count` versions of the song — same
        lyrics, tweaked styles — saved as inactive MusicTracks. None becomes
        the project's music until the user picks one (select-music endpoint).
        The music model stays loaded across the whole batch (one load, N songs).
        `offset` skips the first N style tweaks — used when resuming a paused
        batch so no style is generated twice. A pause request stops the batch
        gracefully after the current version.
        """
        session = get_session()
        generated: list[str] = []
        try:
            project = session.query(Project).get(project_id)
            channel = project.channel
            profile = self.director.load_channel_profile(channel.slug)
            music_cfg = (profile or {}).get("music", {})

            project_dir = self.config.paths.projects_dir / project_id
            project_dir.mkdir(parents=True, exist_ok=True)

            use_lyrics = lyrics if lyrics is not None else (project.lyrics or "")
            if vocals is False:
                use_lyrics = ""
            base_style = (style or project.music_style
                          or music_cfg.get("style", "cheerful children's song"))
            use_engine = engine or getattr(project, "music_model", None) or "auto"

            offset = max(0, min(offset, len(self.MUSIC_VARIANT_TWEAKS) - 1))
            end = min(offset + max(1, count), len(self.MUSIC_VARIANT_TWEAKS))
            count = end - offset
            batch_tag = int(time.time())
            for n, i in enumerate(range(offset, end)):
                if self._pause_requested:
                    # graceful pause between versions — everything done so far
                    # is kept; the wizard's Resume button continues from here
                    self._pause_requested = False
                    logger.info(f"[Pipeline] Music batch paused after {len(generated)}/{count} versions")
                    self._emit_progress(
                        phase=PipelinePhase.IDLE,
                        project_id=project_id,
                        message=f"Music paused — {len(generated)} version(s) saved, resume anytime",
                        percent=100.0,
                    )
                    return generated

                tweak = self.MUSIC_VARIANT_TWEAKS[i]
                v_style = f"{base_style}, {tweak}" if tweak else base_style
                out_path = str(project_dir / f"music_v{batch_tag}_{i + 1:02d}.wav")

                self._emit_progress(
                    phase=PipelinePhase.MUSIC,
                    project_id=project_id,
                    message=f"Song version {n + 1}/{count}…",
                    percent=100.0 * n / count,
                )
                try:
                    result = self.music_gen.generate(
                        style_prompt=v_style,
                        duration=int(project.duration_target) + 2,
                        lyrics=use_lyrics,
                        output_path=out_path,
                        instrumental=not bool(use_lyrics.strip()),
                        channel_profile=profile,
                        engine=use_engine,
                    )
                except Exception as e:
                    logger.warning(f"[Pipeline] Song version {i + 1}/{count} failed: {e}")
                    continue

                # inactive on purpose — the user auditions and selects one
                track = MusicTrack(
                    project_id=project_id,
                    style_prompt=v_style,
                    output_path=result.path,
                    duration=result.duration,
                    is_active=False,
                )
                session.add(track)
                session.commit()
                generated.append(result.path)

            self.manager.unload()
            self._emit_progress(
                phase=PipelinePhase.IDLE,
                project_id=project_id,
                message=f"{len(generated)} song versions ready — pick your favourite",
                percent=100.0,
            )
            return generated

        finally:
            session.close()

    # ── Phase 6: Final Render ──────────────────────────────────────────

    def render(
        self,
        project_id: str,
        narration_path: Optional[str] = None,
        music_path: Optional[str] = None,
        resolution: Optional[str] = None,
    ):
        with self._exclusive("render"):
            return self._render_impl(project_id, narration_path, music_path, resolution)

    def _render_impl(
        self,
        project_id: str,
        narration_path: Optional[str] = None,
        music_path: Optional[str] = None,
        resolution: Optional[str] = None,
    ):
        """Assemble all clips + audio into final video.

        `resolution` overrides the channel default (e.g. "4k" for 3840x2160).
        """
        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            scenes = session.query(Scene).filter(
                Scene.project_id == project_id,
                Scene.status.in_([SceneStatus.GENERATED, SceneStatus.APPROVED]),
            ).order_by(Scene.scene_number).all()

            project.status = ProjectStatus.ASSEMBLING
            session.commit()

            self._emit_progress(
                phase=PipelinePhase.ASSEMBLING,
                project_id=project_id,
                message="Assembling final video...",
            )

            # Build clip list — prefer upscaled, fall back to raw
            clips = []
            for scene in scenes:
                gen = scene.active_generation
                if not gen:
                    continue
                clip_path = gen.upscaled_path or gen.output_path
                if clip_path and Path(clip_path).exists():
                    notes = scene.director_notes or {}
                    clips.append(ClipEntry(
                        path=clip_path,
                        duration=scene.duration,
                        transition_in=notes.get("transition_in", "crossfade"),
                        transition_out=notes.get("transition_out", "crossfade"),
                    ))

            if not clips:
                raise ValueError("No clips available for assembly")

            # Check for music track
            if not music_path:
                active_music = session.query(MusicTrack).filter(
                    MusicTrack.project_id == project_id,
                    MusicTrack.is_active == True,
                ).first()
                if active_music:
                    music_path = active_music.output_path

            # Output path
            project_dir = self.config.paths.projects_dir / project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(project_dir / "final_render.mp4")

            channel = project.channel
            out_resolution = resolution or channel.target_resolution

            result = self.assembler.assemble(
                clips=clips,
                output_path=output_path,
                narration_path=narration_path,
                music_path=music_path,
                resolution=out_resolution,
                transition_duration=self.config.generation.transition_duration,
            )

            # Save render job
            render_job = RenderJob(
                project_id=project_id,
                resolution=out_resolution,
                output_path=result.output_path,
                status=RenderStatus.COMPLETED,
                progress_pct=100.0,
                render_settings={
                    "total_duration": result.total_duration,
                    "file_size_mb": result.file_size_mb,
                    "render_time": result.render_time,
                    "clip_count": len(clips),
                },
            )
            session.add(render_job)

            project.output_path = result.output_path
            project.status = ProjectStatus.RENDERED

            # Final QA: plays, ~target duration, HD, audio present — then persist
            # qa_report.json so overnight runs are diagnosable at a glance.
            try:
                from app.services import qa as qa_svc
                from dataclasses import asdict as _asdict
                final_qa = qa_svc.check_final(
                    self.config.paths.ffmpeg_bin,
                    result.output_path,
                    float(project.duration_target or 0),
                    expect_audio=bool(music_path or narration_path),
                )
                self._qa_notes["final"] = _asdict(final_qa)
                if not final_qa.ok:
                    logger.warning(f"[QA] Final render issues: {', '.join(final_qa.issues)}")
                    project.error_log = f"QA warnings: {', '.join(final_qa.issues)}"
                qa_svc.write_report(project_dir, self._qa_notes)
            except Exception as qa_err:
                logger.warning(f"[QA] final check failed (non-fatal): {qa_err}")

            session.commit()

            self._emit_progress(
                phase=PipelinePhase.DONE,
                message=f"Render complete: {result.total_duration:.0f}s, {result.file_size_mb:.1f}MB",
                percent=100.0,
            )

        except PipelineCancelled as ce:
            logger.info(f"[Pipeline] Render cancelled: {ce}")
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = "Render cancelled by user"
                    session.commit()
            except Exception as dbe:
                logger.error(f"Failed to update project status on cancel: {dbe}")
            
            self._emit_progress(
                phase=PipelinePhase.ERROR,
                error="Cancelled",
                message="Render cancelled by user",
            )
            raise

        except Exception as e:
            logger.error(f"[Pipeline] Render failed: {e}", exc_info=True)
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = str(e)
                    session.commit()
            except Exception as dbe:
                logger.error(f"Failed to update project status on error: {dbe}")
            
            self._emit_progress(
                phase=PipelinePhase.ERROR,
                error=str(e),
                message=f"Render failed: {e}",
            )
            raise

        finally:
            session.close()

    # ── Convenience: Full Auto Pipeline ────────────────────────────────

    def run_full_auto(
        self,
        project_id: str,
        narration_path: Optional[str] = None,
    ):
        with self._exclusive("full-auto"):
            return self._run_full_auto_impl(project_id, narration_path)

    def _run_full_auto_impl(
        self,
        project_id: str,
        narration_path: Optional[str] = None,
    ):
        """
        Run the entire pipeline end-to-end without user intervention.
        Skips already-completed phases based on current project status.
        """
        logger.info(f"[Pipeline] Starting full auto run for project {project_id}")

        try:
            # QA gate 0: preflight — verify engines/models/ffmpeg/disk BEFORE
            # burning GPU time. An overnight run must fail in 5 seconds, not 5 hours.
            from app.services import qa as qa_svc
            self._qa_notes = {"scenes": []}
            pf = qa_svc.preflight(self.config)
            self._qa_notes["preflight"] = pf.to_dict()
            if not pf.ok:
                session = get_session()
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = f"Preflight failed: {pf.summary()}"
                    session.commit()
                session.close()
                qa_svc.write_report(self.config.paths.projects_dir / project_id, self._qa_notes)
                self._emit_progress(
                    phase=PipelinePhase.ERROR,
                    error="preflight",
                    message=f"Preflight failed: {pf.summary()}",
                )
                return

            # Check current status to skip completed phases
            session = get_session()
            project = session.query(Project).get(project_id)
            current_status = project.status
            has_scenes = session.query(Scene).filter(
                Scene.project_id == project_id
            ).count() > 0
            session.close()

            logger.info(f"[Pipeline] Project status: {current_status}, has_scenes: {has_scenes}")

            # Phase 1: Script — only if no script exists yet.
            # Song projects (lyrics present) get instant beat-synced scenes;
            # story projects go through the director LLM.
            if current_status == ProjectStatus.DRAFT or not has_scenes:
                session = get_session()
                p = session.query(Project).get(project_id)
                has_lyrics = bool((p.lyrics or "").strip())
                session.close()
                if has_lyrics:
                    logger.info("[Pipeline] Phase 1: Building scenes from lyrics (song mode)...")
                    self.generate_scenes_from_lyrics(project_id)
                else:
                    logger.info("[Pipeline] Phase 1: Generating script...")
                    self.generate_script(project_id)
            else:
                logger.info(f"[Pipeline] Phase 1: Skipping script generation (status={current_status}, scenes exist)")

            # QA gate 1: creative lint — enforce channel framing/negatives/scene
            # types and normalize durations to the exact target, with auto-fixes.
            try:
                session = get_session()
                project = session.query(Project).get(project_id)
                scenes = session.query(Scene).filter(
                    Scene.project_id == project_id
                ).order_by(Scene.scene_number).all()
                profile = self.director.load_channel_profile(project.channel.slug) or {}
                lo, hi = self.config.generation.clip_duration_range
                if scenes:
                    lo = min(lo, float(project.duration_target) / len(scenes))
                self._qa_notes["lint"] = qa_svc.lint_script(
                    scenes, profile, float(project.duration_target), (lo, hi))
                session.commit()
                session.close()
            except Exception as lint_err:
                logger.warning(f"[Pipeline] Script lint failed (continuing): {lint_err}")

            # Auto-approve all scenes
            session = get_session()
            scenes = session.query(Scene).filter(
                Scene.project_id == project_id
            ).all()
            for s in scenes:
                if s.status in (SceneStatus.PENDING, SceneStatus.FAILED):
                    s.status = SceneStatus.PENDING
            project = session.query(Project).get(project_id)
            project.status = ProjectStatus.APPROVED
            session.commit()
            session.close()

            # Phase 2: Narration — prefer a user-supplied voiceover file, else TTS
            if not narration_path:
                narration_path = self._find_user_audio(project_id, "voice")
                if narration_path:
                    logger.info(f"[Pipeline] Using your custom voiceover: {narration_path}")
                else:
                    try:
                        narration_path = self.generate_tts(project_id)
                    except Exception as tts_err:
                        logger.warning(f"[Pipeline] TTS failed (skipping narration): {tts_err}")
                        narration_path = None

            # Phase 3: Generate assets (video/images).
            # 832x480 = 16:9 (YouTube-native) AND VRAM-safe for LTX-22B on 16GB
            # (1024x576 spills to system RAM). batch=True keeps the model resident.
            logger.info("[Pipeline] Phase 3: Starting asset generation...")
            session = get_session()
            p = session.query(Project).get(project_id)
            inline = bool(getattr(p, "upscale_inline", True))
            session.close()
            self.start_generation(project_id, width=832, height=480, batch=True,
                                  upscale_inline=inline)

            # Phase 4: Upscale (skips scenes already upscaled inline / previous run)
            try:
                self.start_upscale(project_id)
            except Exception as upscale_err:
                logger.warning(f"[Pipeline] Upscale failed (continuing): {upscale_err}")

            # Phase 5: Music — user file > existing generated track (resume) > generate
            music_path = self._find_user_audio(project_id, "music")
            if music_path:
                logger.info(f"[Pipeline] Using your custom song: {music_path}")
            else:
                session = get_session()
                existing = session.query(MusicTrack).filter(
                    MusicTrack.project_id == project_id,
                    MusicTrack.is_active == True,
                ).first()
                music_path = existing.output_path if existing else None
                session.close()
                if music_path and Path(music_path).exists():
                    logger.info(f"[Pipeline] Reusing existing music track: {music_path}")
                else:
                    try:
                        music_path = self.generate_music(project_id)
                    except Exception as music_err:
                        logger.warning(f"[Pipeline] Music generation failed (skipping): {music_err}")
                        music_path = None

            # Phase 6: Render
            self.render(project_id, narration_path=narration_path, music_path=music_path)

            logger.info(f"[Pipeline] Full auto complete for project {project_id}")

        except PipelinePaused:
            logger.info(f"[Pipeline] Paused for project {project_id} — resume anytime")
            self._emit_progress(
                phase=PipelinePhase.IDLE,
                message="Pipeline paused — progress saved, resume anytime",
            )
        except PipelineCancelled:
            logger.info(f"[Pipeline] Cancelled for project {project_id}")
            self._emit_progress(
                phase=PipelinePhase.IDLE,
                message="Pipeline cancelled",
            )
        except Exception as e:
            logger.error(f"[Pipeline] Fatal error: {e}", exc_info=True)
            self._emit_progress(
                phase=PipelinePhase.ERROR,
                error=str(e),
                message=f"Pipeline failed: {e}",
            )


class PipelineCancelled(Exception):
    pass


class PipelinePaused(Exception):
    """Graceful pause: stop after the current item, keep all finished work,
    leave the project in a resumable (non-failed) state."""
    pass


class PipelineBusy(Exception):
    """Another exclusive phase (generation/upscale/render) is already running.
    The caller should surface this to the user instead of queueing blindly."""
    pass
