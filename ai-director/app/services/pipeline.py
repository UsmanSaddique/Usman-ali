"""
AI Director — Pipeline Orchestrator
Coordinates the full video generation workflow across all services.
Manages state transitions, error handling, retries, and progress reporting.
"""
import os
import time
import logging
import asyncio
from enum import Enum
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

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

            # Quick check: is the TTS server reachable?
            try:
                import httpx
                httpx.get(self.config.tts.api_url, timeout=3.0)
            except Exception:
                logger.warning("[Pipeline] TTS server not reachable at %s, skipping narration", self.config.tts.api_url)
                return None

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

    def start_generation(self, project_id: str, scene_ids: Optional[list[str]] = None, width: Optional[int] = None, height: Optional[int] = None):
        """
        Generate all scene assets one by one.
        If scene_ids is provided, only generate those specific scenes.
        """
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

            for i, scene in enumerate(scenes):
                self._check_cancel()

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

                    scene.status = SceneStatus.GENERATED
                    scene.active_generation_id = generation.id
                    project.completed_scenes = i + 1

                except Exception as e:
                    logger.error(f"[Pipeline] Scene {scene.scene_number} failed: {e}")
                    scene.status = SceneStatus.FAILED
                    scene.retry_count += 1

                    self._check_cancel()
                    if scene.retry_count <= scene.max_retries:
                        try:
                            self._retry_scene_direct(scene, str(e), clips_dir, images_dir, session, width=width, height=height)
                        except Exception as retry_err:
                            logger.error(f"[Pipeline] Retry also failed: {retry_err}")

                elapsed = time.time() - t0
                scene_times.append(elapsed)
                session.commit()

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

    def _get_project_loras(self, project) -> list[tuple[str, float]]:
        """Get LoRA list from project defaults as [(filename, weight), ...]."""
        ids = project.default_lora_ids or []
        weights = project.default_lora_weights or []
        result = []
        for i, lora_file in enumerate(ids):
            w = weights[i] if i < len(weights) else 0.8
            result.append((lora_file, w))
        return result

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
        # Prioritize scene's video model, then project's, then default
        video_model = scene.video_model or getattr(project, "video_model", None) or "LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf"
        project_loras = self._get_project_loras(project)

        # Per-scene LoRA overrides
        if scene.lora_ids:
            scene_loras = list(zip(
                scene.lora_ids,
                scene.lora_weights or [0.8] * len(scene.lora_ids),
            ))
        else:
            scene_loras = project_loras

        sdxl_path = str(self.config.image.path)
        sdxl_available = sdxl_path and os.path.exists(sdxl_path)

        try:
            if scene.scene_type == SceneType.TXT2VID:
                result = self.video_gen.txt2vid(
                    prompt=scene.prompt,
                    negative_prompt=scene.negative_prompt,
                    num_frames=int(scene.duration * self.config.video.default_fps),
                    output_path=str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4"),
                    model_filename=video_model,
                    loras=scene_loras or None,
                    width=width,
                    height=height,
                )
                gen.model_used = result.model_used
                gen.output_path = result.path
                gen.seed = result.seed
                gen.generation_time_sec = result.generation_time
                gen.parameters = {
                    "width": result.width, "height": result.height,
                    "num_frames": result.num_frames, "fps": result.fps,
                }

            elif scene.scene_type == SceneType.IMG2VID:
                if not sdxl_available:
                    logger.warning(f"[Pipeline] Scene {scene.scene_number}: SDXL not available, falling back to txt2vid")
                    result = self.video_gen.txt2vid(
                        prompt=scene.prompt,
                        negative_prompt=scene.negative_prompt,
                        num_frames=int(scene.duration * self.config.video.default_fps),
                        output_path=str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4"),
                        model_filename=video_model,
                        loras=scene_loras or None,
                        width=width,
                        height=height,
                    )
                    gen.model_used = f"txt2vid-fallback:{result.model_used}"
                    gen.output_path = result.path
                    gen.seed = result.seed
                    gen.generation_time_sec = result.generation_time
                    gen.parameters = {
                        "width": result.width, "height": result.height,
                        "num_frames": result.num_frames, "fps": result.fps,
                    }
                else:
                    # 1. Generate image first
                    img_result = self.image_gen.generate(
                        prompt=scene.prompt,
                        negative_prompt=scene.negative_prompt,
                        output_path=str(images_dir / f"scene_{scene.scene_number:03d}_v{version}.png"),
                        lora_paths=scene.lora_ids,
                        lora_weights=scene.lora_weights,
                        width=width,
                        height=height,
                    )
                    self.manager.unload()

                    # 2. Img2Vid
                    result = self.video_gen.img2vid(
                        prompt=scene.prompt,
                        image_path=img_result.path,
                        negative_prompt=scene.negative_prompt,
                        num_frames=int(scene.duration * self.config.video.default_fps),
                        output_path=str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4"),
                        model_filename=video_model,
                        loras=scene_loras or None,
                        width=width,
                        height=height,
                    )
                    gen.model_used = f"sdxl+{result.model_used}"
                    gen.thumbnail_path = img_result.path
                    gen.generation_time_sec = img_result.generation_time + result.generation_time

                gen.output_path = result.path
                gen.seed = result.seed
                if not gen.generation_time_sec:
                    gen.generation_time_sec = result.generation_time

            elif scene.scene_type == SceneType.STILL_PAN:
                if not sdxl_available:
                    logger.warning(f"[Pipeline] Scene {scene.scene_number}: SDXL not available, falling back to txt2vid for still_pan")
                    result = self.video_gen.txt2vid(
                        prompt=scene.prompt,
                        negative_prompt=scene.negative_prompt,
                        num_frames=int(scene.duration * self.config.video.default_fps),
                        output_path=str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4"),
                        model_filename=video_model,
                        loras=scene_loras or None,
                        width=width,
                        height=height,
                    )
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
                    gen.model_used = "sdxl+kenburns"
                    gen.output_path = result.path
                    gen.seed = img_result.seed
                    gen.generation_time_sec = img_result.generation_time + result.generation_time
                    gen.thumbnail_path = img_result.path

            else:
                raise ValueError(f"Unknown scene type: {scene.scene_type}")

            gen.status = GenerationStatus.COMPLETED

        except Exception as e:
            gen.status = GenerationStatus.FAILED
            gen.error_log = str(e)
            raise

        return gen

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

        self._generate_scene(scene, clips_dir, images_dir, session, width=width, height=height)
        scene.status = SceneStatus.GENERATED

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

        self._generate_scene(scene, clips_dir, images_dir, session)
        scene.status = SceneStatus.GENERATED

    # ── Phase 4: Upscale ───────────────────────────────────────────────

    def start_upscale(self, project_id: str):
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

                gen = scene.active_generation
                if not gen or not gen.output_path:
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

    # ── Phase 5: Music Generation ─────────────────────────────────

    def generate_music(self, project_id: str) -> Optional[str]:
        """Generate background music track using ACE-Step."""
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
            music_path = str(project_dir / "music.wav")

            if profile:
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

            # Save to DB
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

    # ── Phase 6: Final Render ──────────────────────────────────────────

    def render(
        self,
        project_id: str,
        narration_path: Optional[str] = None,
        music_path: Optional[str] = None,
    ):
        """Assemble all clips + audio into final video."""
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

            result = self.assembler.assemble(
                clips=clips,
                output_path=output_path,
                narration_path=narration_path,
                music_path=music_path,
                resolution=channel.target_resolution,
                transition_duration=self.config.generation.transition_duration,
            )

            # Save render job
            render_job = RenderJob(
                project_id=project_id,
                resolution=channel.target_resolution,
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
        """
        Run the entire pipeline end-to-end without user intervention.
        Skips already-completed phases based on current project status.
        """
        logger.info(f"[Pipeline] Starting full auto run for project {project_id}")

        try:
            # Check current status to skip completed phases
            session = get_session()
            project = session.query(Project).get(project_id)
            current_status = project.status
            has_scenes = session.query(Scene).filter(
                Scene.project_id == project_id
            ).count() > 0
            session.close()

            logger.info(f"[Pipeline] Project status: {current_status}, has_scenes: {has_scenes}")

            # Phase 1: Script — only if no script exists yet
            if current_status == ProjectStatus.DRAFT or not has_scenes:
                logger.info("[Pipeline] Phase 1: Generating script...")
                self.generate_script(project_id)
            else:
                logger.info(f"[Pipeline] Phase 1: Skipping script generation (status={current_status}, scenes exist)")

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

            # Phase 2: TTS (if narration exists) — non-fatal
            if not narration_path:
                try:
                    narration_path = self.generate_tts(project_id)
                except Exception as tts_err:
                    logger.warning(f"[Pipeline] TTS failed (skipping narration): {tts_err}")
                    narration_path = None

            # Phase 3: Generate assets (video/images)
            logger.info("[Pipeline] Phase 3: Starting asset generation...")
            self.start_generation(project_id)

            # Phase 4: Upscale
            try:
                self.start_upscale(project_id)
            except Exception as upscale_err:
                logger.warning(f"[Pipeline] Upscale failed (continuing): {upscale_err}")

            # Phase 5: Music — non-fatal
            music_path = None
            try:
                music_path = self.generate_music(project_id)
            except Exception as music_err:
                logger.warning(f"[Pipeline] Music generation failed (skipping): {music_err}")

            # Phase 6: Render
            self.render(project_id, narration_path=narration_path, music_path=music_path)

            logger.info(f"[Pipeline] Full auto complete for project {project_id}")

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
