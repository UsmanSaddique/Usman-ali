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
    SAFETY = "safety"
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
        from app.services.yt_safety import SafetyGateService
        self.safety = SafetyGateService(model_manager, config)
        from app.services.narration_writer import NarrationWriterService
        self.narration_writer = NarrationWriterService(model_manager, config)
        from app.services.template_renderer import TemplateRenderer
        self.template_renderer = TemplateRenderer(model_manager, config)

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
        self._write_run_state()

    def _write_run_state(self):
        """Journal the live pipeline state to projects/<id>/run_state.json.
        Written atomically on every progress tick, so after ANY crash / kill /
        power loss the file shows exactly which phase+scene the run died in.
        Best-effort: journaling must never break the pipeline itself."""
        p = self._progress
        if not p.project_id:
            return
        try:
            import json
            from datetime import datetime, timezone
            pdir = self.config.paths.projects_dir / p.project_id
            pdir.mkdir(parents=True, exist_ok=True)
            state = {
                "phase": getattr(p.phase, "value", str(p.phase)),
                "current_scene": p.current_scene,
                "total_scenes": p.total_scenes,
                "percent": round(float(p.percent or 0), 1),
                "message": p.message,
                "error": p.error,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
            }
            tmp = pdir / "run_state.json.tmp"
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp, pdir / "run_state.json")
        except Exception:
            pass

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
                # SEO block — consumed by metadata.json at render time
                "description": script.description,
                "tags": script.tags,
                "hashtags": script.hashtags,
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

    # ── Phase 1n: Narration Script (narration mode) ────────────────────

    def generate_narration_script(self, project_id: str) -> dict:
        """Narration mode phase 1: two-pass script write (outline → beats),
        then the universal safety gate WHILE the LLM is still resident (the
        critic reuses the loaded model — no second 27B load). Saves to
        Project.narration_script."""
        import json as _json
        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            channel_slug = project.channel.slug
            title, duration = project.title, project.duration_target
            context = project.context or ""
        finally:
            session.close()

        self._emit_progress(
            phase=PipelinePhase.SCRIPTING, project_id=project_id,
            message="Writing narration script (outline → chapters → beats)...")

        try:
            script = self.narration_writer.write(
                topic=title, duration_sec=duration,
                context=context, channel_slug=channel_slug,
                unload_after=False,   # safety critic reuses the resident LLM
            )

            session = get_session()
            try:
                project = session.query(Project).get(project_id)
                project.project_type = "narration"
                project.narration_script = _json.dumps(script, ensure_ascii=False)
                # mirror the SEO block into script_raw so the existing
                # release-assets writer (metadata.json) works unchanged
                seo = script.get("seo", {})
                project.script_raw = _json.dumps({
                    "description": seo.get("description", ""),
                    "tags": seo.get("tags", []),
                    "hashtags": seo.get("hashtags", []),
                    "thumbnail_prompt": seo.get("thumbnail_prompt", ""),
                })
                if script.get("title"):
                    project.title = script["title"]
                project.status = ProjectStatus.SCRIPTED
                session.commit()
            finally:
                session.close()

            # Universal safety gate on the fresh script (LLM already loaded)
            self._emit_progress(
                phase=PipelinePhase.SAFETY, project_id=project_id,
                message="YT safety gate: reviewing narration + metadata...")
            gate = self.safety.run_gate(project_id, use_llm=True,
                                        auto_revise=True, unload_after=True)

            n_beats = sum(len(c.get("beats", [])) for c in script.get("chapters", []))
            self._emit_progress(
                phase=PipelinePhase.IDLE, project_id=project_id, percent=100.0,
                message=f"Narration script ready: {len(script.get('chapters', []))} "
                        f"chapters, {n_beats} beats — safety: {gate.verdict}")
            return script
        except Exception as e:
            logger.error(f"[Pipeline] Narration scripting failed: {e}", exc_info=True)
            session = get_session()
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = str(e)
                    session.commit()
            finally:
                session.close()
            self._emit_progress(phase=PipelinePhase.ERROR, error=str(e),
                                message=f"Narration scripting failed: {e}")
            raise
        finally:
            try:
                self.manager.unload()
            except Exception:
                pass

    # ── Phase 2n: Narration Audio + Beat Planning (narration mode) ─────

    def generate_narration_audio(self, project_id: str) -> str:
        """Narration mode phase 2: Kokoro master WAV (sample-accurate beat
        offsets + transcribe-back QA) → beat planner → Scene rows with exact
        (narration_start, narration_end). Returns the master WAV path."""
        import json as _json
        from app.services import narration_scenes

        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            if not project.narration_script:
                raise ValueError("No narration script — run generate-narration-script first")
            script = _json.loads(project.narration_script)
            voice = project.narration_voice or self.config.tts.kokoro_voice
            channel_slug = project.channel.slug
        finally:
            session.close()

        beats = narration_scenes.flatten_beats(script)
        beats_for_tts = [{"text": b["text"],
                          "chapter_index": b["chapter_index"],
                          "chapter_break": b["chapter_break"]} for b in beats]

        self._emit_progress(
            phase=PipelinePhase.TTS, project_id=project_id,
            message=f"Voicing {len(beats_for_tts)} beats with Kokoro/Chatterbox ({voice})...")

        # Determine if we have a custom voice reference for Chatterbox cloning
        voice_ref = None
        try:
            profile = self.director.load_channel_profile(channel_slug) or {}
            voice_ref = profile.get("narration", {}).get("voice_ref")
            if voice_ref and not Path(voice_ref).is_absolute():
                voice_ref = str(self.config.paths.assets_dir / "voices" / voice_ref)
        except Exception:
            pass

        project_dir = self.config.paths.projects_dir / project_id
        try:
            master = self.tts.generate_narration_master(
                beats_for_tts, str(project_dir), voice=voice, voice_ref=voice_ref)
        finally:
            # Kokoro + whisper are small but the video phase wants every MB
            try:
                self.tts.unload_engines()
            except Exception:
                pass

        # Plan scenes against the REAL narration timeline
        timing = _json.loads(Path(master.timing_json).read_text(encoding="utf-8"))
        profile = {}
        try:
            profile = self.director.load_channel_profile(channel_slug) or {}
        except Exception:
            pass

        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            cap = self._max_clip_seconds(project.video_model)
            assets_dir = self.config.paths.projects_dir / project_id / "assets_in"
            planned = narration_scenes.plan_scenes(
                script, timing["beats"], master.duration, profile, cap, assets_dir=assets_dir)

            session.query(Scene).filter(Scene.project_id == project_id).delete()
            for p in planned:
                session.add(Scene(
                    project_id=project_id,
                    scene_number=p["scene_number"],
                    scene_type=SceneType(p["scene_type"]),
                    prompt=p["prompt"],
                    negative_prompt=p["negative_prompt"],
                    duration=p["duration"],
                    camera_motion=p["camera_motion"],
                    narration_text=p["narration_text"],
                    narration_start=p["narration_start"],
                    narration_end=p["narration_end"],
                    visual_type=p["visual_type"],
                    sfx_prompt=p["sfx_prompt"],
                    director_notes=p["director_notes"],
                    status=SceneStatus.PENDING,
                ))
            project.narration_audio_path = master.audio_path
            project.total_scenes = len(planned)
            project.completed_scenes = 0
            project.status = ProjectStatus.APPROVED
            session.commit()
        finally:
            session.close()

        bad = [b for b in master.beats if b.wer > self.config.tts.wer_flag_threshold]
        self._emit_progress(
            phase=PipelinePhase.IDLE, project_id=project_id, percent=100.0,
            message=f"Narration ready: {master.duration:.0f}s, "
                    f"{len(planned)} scenes planned"
                    + (f" — {len(bad)} beat(s) flagged by WER QA" if bad else ""))
        return master.audio_path

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

            # The video must cover the SONG, not the nominal target — when a
            # music track is already selected, plan against its real length.
            active_track = session.query(MusicTrack).filter(
                MusicTrack.project_id == project_id,
                MusicTrack.is_active == True,
            ).first()
            track_path = active_track.output_path if active_track else None
            if active_track and (active_track.duration or 0) > 0 \
                    and Path(track_path or "").exists():
                duration = float(active_track.duration)
                logger.info(f"[Pipeline] Planning scenes against the selected "
                            f"song: {duration:.0f}s (target was {project.duration_target}s)")

            n = num_clips or project.num_scenes_target or max(1, int(duration // 5))
            # Physical ceiling: a clip can't exceed max_num_frames/fps (5.04s
            # on LTX), and the assembler overlaps each cut by
            # transition_duration (0.5s crossfade) — so each extra clip only
            # ADDS (cap - transition) seconds. Too few clips would silently
            # render a shorter video; raise the count so the song is covered.
            import math
            cap = self._max_clip_seconds(project.video_model)
            td = float(self.config.generation.transition_duration or 0)
            min_n = max(1, math.ceil((duration - td) / max(cap - td, 1.0)))
            if n < min_n:
                logger.info(f"[Pipeline] Raising clip count {n} -> {min_n}: "
                            f"{duration:.0f}s song / {cap:.2f}s clip ceiling "
                            f"/ {td:.1f}s crossfade overlap per cut")
                n = min_n
            # Segments must SUM to duration + total crossfade overlap so the
            # rendered (overlapped) video still spans the whole song.
            plan_total = duration + max(0, n - 1) * td
            segments = parse_lyrics(
                lyrics, plan_total,
                max_segments=n,
                target_segment_sec=plan_total / n,
            )

            profile = self.director.load_channel_profile(project.channel.slug) or {}

            # Lyric sync: snap scene boundaries to real vocal phrase onsets
            # (faster-whisper, CPU). Best-effort — estimates kept on failure.
            if track_path and Path(track_path).exists():
                try:
                    from app.services import lyric_sync
                    synced = lyric_sync.sync_segments_to_audio(
                        segments, track_path, max_clip_sec=cap,
                        language=lyric_sync.whisper_language(
                            profile.get("language", "")))
                    if synced:
                        segments = synced
                except Exception as sync_err:
                    logger.warning(f"[Pipeline] Lyric sync failed (using estimates): {sync_err}")
            # Primary: LLM visual director — reads the project CONTEXT + each
            # lyric line and writes a concrete scene that shows the locked
            # character doing exactly what the line sings (a brushing song
            # actually brushes). Falls back to the deterministic template builder
            # only if the LLM storyboard is unusable.
            prompts = None
            try:
                prompts = self.director.storyboard_from_lyrics(
                    (project.context or "").strip(),
                    (project.title or "").strip(),
                    segments, project.channel.slug, project_id)
                if not prompts or len(prompts) < len(segments):
                    raise ValueError(
                        f"storyboard returned {len(prompts) if prompts else 0}/"
                        f"{len(segments)} scenes")
                logger.info("[Pipeline] Scenes built via LLM visual director "
                            "(context + lyric matched)")
            except Exception as sb_err:
                logger.warning(f"[Pipeline] LLM storyboard failed ({sb_err}); "
                               f"falling back to template builder")
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
                    # keep the lyric line off narration (no TTS) but on record —
                    # the render uses it to write the final .srt subtitle file
                    director_notes={
                        "lyric_text": p["narration_text"],
                        "director_guidance": p.get("director_guidance", {}),
                    },
                    status=SceneStatus.PENDING,
                ))
            project.total_scenes = len(prompts)
            project.completed_scenes = 0
            project.status = ProjectStatus.SCRIPTED
            session.commit()
            # persist the master-director storyboard next to the renders
            try:
                from app.services import master_director
                master_director.save_plan(
                    self.config.paths.projects_dir / project_id,
                    [p.get("director_guidance", {}) for p in prompts])
            except Exception as g_err:
                logger.warning(f"[Pipeline] director guidance save failed: {g_err}")
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

    def ensure_safety(self, project_id: str, use_llm: bool = True):
        """UNIVERSAL YT-safety gate — every project type, before ANY GPU spend.

        Honors the latest stored SafetyReport (pass/override lets generation
        run); with no report on file it runs the gate now. Raises SafetyBlocked
        on block/revise. Kill switch: AIDIR_SAFETY_ENFORCE=0."""
        if os.environ.get("AIDIR_SAFETY_ENFORCE", "1") == "0":
            return
        from app.services import yt_safety
        session = get_session()
        try:
            verdict = yt_safety.latest_verdict(session, project_id)
        finally:
            session.close()

        if verdict is None:
            self._emit_progress(
                phase=PipelinePhase.SAFETY, project_id=project_id,
                message="Running YouTube safety gate on the script...")
            result = self.safety.run_gate(project_id, use_llm=use_llm)
            verdict = result.verdict

        if verdict in ("pass", "override"):
            return
        self._emit_progress(
            phase=PipelinePhase.ERROR, project_id=project_id, error="safety",
            message=f"Safety gate verdict '{verdict}' — generation blocked. "
                    f"Review the safety report, fix or override, then retry.")
        raise SafetyBlocked(
            f"YT-safety gate verdict is '{verdict}' — fix the flagged issues "
            f"(GET /api/projects/{project_id}/safety-report), re-run the check, "
            f"or record a manual override before generating.")

    def start_generation(self, project_id: str, scene_ids: Optional[list[str]] = None, width: Optional[int] = None, height: Optional[int] = None, batch: bool = False, upscale_inline: bool = False):
        with self._exclusive("generation"):
            self._progress.project_id = project_id  # run_state.json journal target
            self.ensure_safety(project_id)
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

            # Premium opening: scenes starting inside the first N seconds of
            # the video get higher resolution + more steps (see config.video).
            premium_ids = self._premium_scene_ids(session, project_id)

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

                is_premium = scene.id in premium_ids
                vc = self.config.video
                s_width = vc.premium_width if is_premium else width
                s_height = vc.premium_height if is_premium else height
                s_steps = vc.premium_steps if is_premium else None
                if is_premium:
                    logger.info(
                        f"[Pipeline] Scene {scene.scene_number}: PREMIUM OPENING "
                        f"quality — {s_width}x{s_height} @ {s_steps} steps "
                        f"(slower on purpose; the hook must look best)")

                try:
                    generation = self._generate_scene(
                        scene, clips_dir, images_dir, session,
                        width=s_width, height=s_height, steps=s_steps,
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
                            self._retry_scene_direct(scene, err, clips_dir, images_dir, session,
                                                     width=s_width, height=s_height)
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
        self, scene: Scene, clips_dir: Path, images_dir: Path, session,
        width: Optional[int] = None, height: Optional[int] = None,
        steps: Optional[int] = None,
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
                    steps=steps,
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
                    "steps": steps,
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
                        steps=steps,
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
                        "steps": steps,
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
                        steps=steps,
                        seed=scene_seed,
                        clear_vram_first=need_vram_free,
                    )
                    self._video_model_loaded = True
                    gen.model_used = f"{image_engine}+{result.model_used}"
                    gen.thumbnail_path = img_result.path
                    gen.generation_time_sec = (img_result.generation_time or 0) + result.generation_time

                gen.output_path = result.path
                gen.seed = result.seed
                if not gen.parameters:
                    gen.parameters = {
                        "width": result.width, "height": result.height,
                        "num_frames": result.num_frames, "fps": result.fps,
                        "steps": steps,
                    }
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
                        steps=steps,
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

            elif scene.scene_type == SceneType.TEMPLATE:
                logger.info(f"[Pipeline] Scene {scene.scene_number}: generating motion-graphics template for {scene.visual_type}")
                profile = self.director.load_channel_profile(project.channel.slug) if project.channel else None
                result_path = self.template_renderer.render_clip(
                    visual_type=scene.visual_type,
                    visual_prompt=scene.prompt,
                    output_path=str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4"),
                    duration=scene.duration,
                    fps=self.config.video.default_fps,
                    width=width or 1920,
                    height=height or 1080,
                    profile=profile,
                )
                gen.model_used = f"template:{scene.visual_type}"
                gen.output_path = result_path
                gen.seed = 0
                gen.generation_time_sec = 0.0

            elif scene.scene_type == SceneType.USER_ASSET:
                asset_path = scene.director_notes.get("user_asset_path")
                logger.info(f"[Pipeline] Scene {scene.scene_number}: processing user asset: {asset_path}")
                output_path = str(clips_dir / f"scene_{scene.scene_number:03d}_v{version}.mp4")
                
                # Use ffmpeg to scale/pad the user's asset to the exact dimensions and duration
                ffmpeg = str(self.config.paths.ffmpeg_bin)
                w, h = width or 1920, height or 1080
                fps = self.config.video.default_fps
                cmd = [
                    ffmpeg, "-y", "-i", str(asset_path),
                    "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}",
                    "-t", f"{scene.duration:.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-an",
                    output_path,
                ]
                import subprocess, time
                t0 = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"User asset processing failed: {r.stderr[-300:]}")
                
                gen.model_used = "user_asset"
                gen.output_path = output_path
                gen.seed = 0
                gen.generation_time_sec = time.time() - t0

            else:
                raise ValueError(f"Unknown scene type: {scene.scene_type}")

            gen.status = GenerationStatus.COMPLETED

        except Exception as e:
            import traceback
            gen.status = GenerationStatus.FAILED
            gen.error_log = traceback.format_exc()
            raise

        return gen

    # ── Release assets: metadata.json + thumbnail ──────────────────────

    def _write_lyrics_srt(self, project, project_dir: Path,
                          cues: list[tuple[float, str]],
                          total_duration: float) -> Optional[Path]:
        """Write final_render.srt: the lyric line each scene was cut on, timed
        on the RENDERED timeline (each cut overlaps by the crossfade, so clip
        starts shift earlier the deeper into the video they are). Scenes that
        predate lyric_text director notes fall back to re-parsing the project
        lyrics onto the clip plan. Returns the .srt path, or None (no lyrics)."""
        if not cues or not (project.lyrics or "").strip():
            return None

        if not any(text for _, text in cues):
            cues = self._cues_from_lyrics(project.lyrics, [d for d, _ in cues])
            if not cues:
                return None

        td = float(self.config.generation.transition_duration or 0)
        starts = []
        t = 0.0
        for i, (dur, _) in enumerate(cues):
            starts.append(max(0.0, t - i * td))
            t += dur
        video_end = float(total_duration or 0) or \
            max(0.0, t - max(0, len(cues) - 1) * td)

        # A long lyric line is split across several clips (same text) — merge
        # those runs into one cue so the subtitle doesn't blink.
        entries: list[tuple[float, float, str]] = []
        for i, (dur, text) in enumerate(cues):
            text = text.strip()
            end = starts[i + 1] if i + 1 < len(starts) else video_end
            if entries and entries[-1][2] == text:
                entries[-1] = (entries[-1][0], end, text)
            else:
                entries.append((starts[i], end, text))

        def ts(sec: float) -> str:
            ms = int(round(max(0.0, min(sec, video_end)) * 1000))
            h, rem = divmod(ms, 3600000)
            m, rem = divmod(rem, 60000)
            s, ms = divmod(rem, 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        blocks = []
        for start, end, text in entries:
            if not text or text == "(instrumental)" or end <= start:
                continue
            blocks.append(f"{len(blocks) + 1}\n{ts(start)} --> {ts(end)}\n{text}\n")
        if not blocks:
            return None
        srt_path = project_dir / "final_render.srt"
        srt_path.write_text("\n".join(blocks), encoding="utf-8")
        logger.info(f"[Release] lyrics subtitles: {srt_path} ({len(blocks)} cues)")
        return srt_path

    def _cues_from_lyrics(self, lyrics: str,
                          durations: list[float]) -> list[tuple[float, str]]:
        """Rebuild (duration, lyric line) for scene lists created before
        lyric_text was recorded: parse the lyrics across the clip plan and give
        each clip the segment under its midpoint."""
        from app.services.lyrics_parser import parse_lyrics
        total = sum(durations)
        if total <= 0:
            return []
        segments = parse_lyrics(lyrics, total, max_segments=len(durations),
                                target_segment_sec=total / len(durations))
        if not segments:
            return []
        cues = []
        t = 0.0
        for dur in durations:
            mid = t + dur / 2
            seg = next((s for s in segments
                        if s.start_sec <= mid < s.end_sec), segments[-1])
            cues.append((dur, "" if seg.is_instrumental else seg.text))
            t += dur
        return cues

    def _write_release_assets(self, project, project_dir: Path, video_path: str,
                              duration: float, resolution: str):
        """Make every render upload-ready: metadata.json (title/description/
        tags/hashtags/made_for_kids) + an auto-picked thumbnail.jpg next to
        the final video. Best-effort — never fails the render."""
        import json

        # ── metadata.json ────────────────────────────────────────────────
        try:
            seo = {}
            try:
                seo = json.loads(project.script_raw or "{}") or {}
            except Exception:
                pass
            profile = self.director.load_channel_profile(project.channel.slug) or {}

            tags = list(seo.get("tags") or [])
            if not tags:
                themes = profile.get("seo_themes") or []
                if isinstance(themes, str):
                    themes = [themes]
                tags = [t.strip() for th in themes for t in str(th).split(",") if t.strip()]
            hashtags = list(seo.get("hashtags") or [])

            description = (seo.get("description") or "").strip()
            if not description:
                # Song / no-LLM projects: synthesize a serviceable description
                hook = (project.context or "").strip()
                first_lines = "\n".join(
                    l.strip() for l in (project.lyrics or "").splitlines()
                    if l.strip() and not l.strip().startswith("[")
                )[:200]
                parts = [project.title, hook or None, first_lines or None,
                         " ".join(hashtags) or None]
                description = "\n\n".join(p for p in parts if p)

            metadata = {
                "title": project.title,
                "description": description,
                "tags": tags[:30],
                "hashtags": hashtags,
                "made_for_kids": bool(getattr(project.channel, "made_for_kids", True)),
                "channel": project.channel.slug,
                "language": profile.get("language", ""),
                "duration_sec": round(float(duration), 1),
                "resolution": resolution,
                "video_file": str(video_path),
                "thumbnail_file": str(project_dir / "thumbnail.jpg"),
            }
            srt_file = project_dir / "final_render.srt"
            if srt_file.exists():
                metadata["subtitles_file"] = str(srt_file)
            meta_path = project_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
            logger.info(f"[Release] metadata.json written: {meta_path}")
        except Exception as e:
            logger.warning(f"[Release] metadata.json failed (non-fatal): {e}")

        # ── thumbnail.jpg — sharpest, most colorful frame wins ──────────
        try:
            thumb = self._pick_thumbnail(video_path, project_dir, duration)
            if thumb:
                project.thumbnail_path = str(thumb)
                logger.info(f"[Release] thumbnail: {thumb}")
        except Exception as e:
            logger.warning(f"[Release] thumbnail failed (non-fatal): {e}")

    def _pick_thumbnail(self, video_path: str, project_dir: Path,
                        duration: float) -> Optional[Path]:
        """Sample frames across the video, score sharpness x colorfulness,
        save the winner as a 1280x720 thumbnail.jpg (CTR beats randomness)."""
        import subprocess
        import tempfile
        from PIL import Image, ImageFilter, ImageStat

        dur = max(float(duration or 0), 2.0)
        # skip the first/last 5%: fade-ins and end cards make bad thumbnails
        points = [dur * f for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)]
        best, best_score = None, -1.0
        with tempfile.TemporaryDirectory() as td:
            for i, t in enumerate(points):
                frame = Path(td) / f"cand_{i}.png"
                cmd = [self.config.paths.ffmpeg_bin, "-y", "-ss", f"{t:.2f}",
                       "-i", str(video_path), "-frames:v", "1",
                       "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
                       str(frame)]
                try:
                    subprocess.run(cmd, capture_output=True, timeout=60)
                except Exception:
                    continue
                if not frame.exists():
                    continue
                try:
                    img = Image.open(frame).convert("RGB")
                    # sharpness: edge energy; colorfulness: channel stddev
                    edges = ImageStat.Stat(img.convert("L").filter(ImageFilter.FIND_EDGES))
                    sharp = edges.stddev[0]
                    color = sum(ImageStat.Stat(img).stddev) / 3.0
                    score = sharp * 0.7 + color * 0.3
                    if score > best_score:
                        best_score = score
                        best = img.copy()
                except Exception:
                    continue
        if best is None:
            return None
        out = project_dir / "thumbnail.jpg"
        best.save(out, "JPEG", quality=92)
        return out

    def write_production_report(self, session, project, project_dir: Path,
                                render_info: Optional[dict] = None) -> Optional[dict]:
        """Emit production_report.json + report.md: what was generated, at what
        resolution/steps, how long every phase took. Best-effort, never fatal.
        Callable standalone for backfilling old projects."""
        import json
        from datetime import datetime
        try:
            scenes = session.query(Scene).filter(
                Scene.project_id == project.id).order_by(Scene.scene_number).all()
            gens = [s.active_generation for s in scenes if s.active_generation]

            # Clip generation stats grouped by resolution+steps profile
            profiles: dict = {}
            gpu_clip_total = 0.0
            for g in gens:
                p = g.parameters or {}
                key = f"{p.get('width', '?')}x{p.get('height', '?')} @ {p.get('steps') or 'default'} steps"
                t = float(g.generation_time_sec or 0)
                gpu_clip_total += t
                b = profiles.setdefault(key, {"clips": 0, "total_sec": 0.0})
                b["clips"] += 1
                b["total_sec"] += t
            for b in profiles.values():
                b["avg_sec_per_clip"] = round(b["total_sec"] / max(b["clips"], 1), 1)
                b["total_sec"] = round(b["total_sec"], 1)

            # Upscale phase span from output file mtimes (not tracked in DB)
            upscale = {}
            up_files = [g.upscaled_path for g in gens
                        if g.upscaled_path and g.upscaled_path != g.output_path
                        and Path(g.upscaled_path).exists()]
            if up_files:
                mtimes = [Path(f).stat().st_mtime for f in up_files]
                upscale = {
                    "clips_upscaled": len(up_files),
                    "wall_clock_sec": round(max(mtimes) - min(mtimes), 0),
                }

            # Music
            tracks = session.query(MusicTrack).filter(
                MusicTrack.project_id == project.id).all()
            active = next((t for t in tracks if t.is_active), None)

            # Render info (passed live, or recovered from the last render job)
            if not render_info:
                rj = session.query(RenderJob).filter(
                    RenderJob.project_id == project.id,
                    RenderJob.status == RenderStatus.COMPLETED,
                ).order_by(RenderJob.created_at.desc()).first()
                render_info = dict(rj.render_settings or {}) if rj else {}
                if rj:
                    render_info["resolution"] = rj.resolution

            wall_total = None
            try:
                wall_total = round(
                    (project.updated_at - project.created_at).total_seconds(), 0)
            except Exception:
                pass

            report = {
                "title": project.title,
                "project_id": project.id,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "final_video": {
                    "path": project.output_path,
                    "duration_sec": render_info.get("total_duration"),
                    "file_size_mb": render_info.get("file_size_mb"),
                    "resolution": render_info.get("resolution", "1080p"),
                    "render_time_sec": render_info.get("render_time"),
                },
                "totals": {
                    "wall_clock_create_to_done_sec": wall_total,
                    "gpu_clip_generation_sec": round(gpu_clip_total, 0),
                    "scenes": len(scenes),
                    "clips_generated": len(gens),
                    "avg_sec_per_clip": round(gpu_clip_total / max(len(gens), 1), 1),
                },
                "generation_profiles": profiles,
                "video_model": getattr(project, "video_model", None),
                "upscale": upscale,
                "music": {
                    "variants_generated": len(tracks),
                    "selected_duration_sec": active.duration if active else None,
                },
            }
            (project_dir / "production_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

            def hm(sec):
                if sec is None:
                    return "n/a"
                sec = int(sec)
                return f"{sec // 3600}h {sec % 3600 // 60:02d}m {sec % 60:02d}s"

            lines = [
                f"# Production Report — {project.title}", "",
                f"**Final video:** {render_info.get('total_duration', 0):.0f}s, "
                f"{render_info.get('resolution', '1080p')}, "
                f"{render_info.get('file_size_mb', 0):.0f}MB "
                f"(render {hm(render_info.get('render_time'))})", "",
                f"- Total wall clock (create → done): **{hm(wall_total)}**",
                f"- GPU clip generation: **{hm(gpu_clip_total)}** over "
                f"{len(gens)} clips (avg {report['totals']['avg_sec_per_clip']}s/clip)",
                f"- Video model: `{report['video_model']}`", "",
                "## Clip profiles (resolution @ steps)", "",
            ]
            for key, b in profiles.items():
                lines.append(f"- **{key}** — {b['clips']} clips, "
                             f"avg {b['avg_sec_per_clip']}s/clip ({hm(b['total_sec'])} total)")
            if upscale:
                lines += ["", "## Upscale",
                          f"- {upscale['clips_upscaled']} clips upscaled in "
                          f"~{hm(upscale['wall_clock_sec'])} (wall clock)"]
            lines += ["", "## Music",
                      f"- {len(tracks)} variants generated; selected track "
                      f"{active.duration if active else '?'}s"]
            (project_dir / "report.md").write_text(
                "\n".join(lines), encoding="utf-8")
            logger.info(f"[Release] production report written: {project_dir / 'report.md'}")
            return report
        except Exception as e:
            logger.warning(f"[Release] production report failed (non-fatal): {e}")
            return None

    def _premium_scene_ids(self, session, project_id: str) -> set[str]:
        """Scene ids whose START falls inside the premium-opening window
        (config.video.premium_open_seconds). Computed over the FULL project
        timeline so a resume run classifies scenes the same way."""
        pw = float(getattr(self.config.video, "premium_open_seconds", 0) or 0)
        if pw <= 0:
            return set()
        all_scenes = session.query(Scene).filter(
            Scene.project_id == project_id
        ).order_by(Scene.scene_number).all()
        ids: set[str] = set()
        t = 0.0
        for s in all_scenes:
            if t < pw:
                ids.add(s.id)
            t += float(s.duration or 4.0)
        return ids

    def _max_clip_seconds(self, video_model: Optional[str]) -> float:
        """Longest clip the video model can physically render — the largest
        8n+1 frame count within max_num_frames, at the model family's fps
        (121f @ 24fps = 5.04s for LTX). Scene durations above this silently
        render at the cap, so planners must respect it."""
        from app.services.comfyui_client import get_defaults_for_model
        try:
            fps = int(get_defaults_for_model(video_model or "").get("fps")
                      or self.config.video.default_fps)
        except Exception:
            fps = int(self.config.video.default_fps)
        max_frames = int(getattr(self.config.video, "max_num_frames", 97))
        frames = ((max_frames - 1) // 8) * 8 + 1
        return frames / max(fps, 1)

    def _plan_video_params(self, scene: Scene, video_model: str) -> tuple[int, int, float]:
        """Return (num_frames, fps, actual_clip_duration) for a scene.

        fps MUST match what the ComfyUI workflow encodes at (the model-family
        default), otherwise clips come out shorter than scene.duration:
        64 frames computed at config's 16fps but encoded at 24fps gave 2.67s
        clips for 4.0s scenes, which also corrupted crossfade offsets at
        assembly. Frames are capped at config.video.max_num_frames — the
        bench-measured VRAM-safe ceiling for LTX-22B at 832x480 on 16GB.
        """
        from app.services.comfyui_client import get_defaults_for_model
        fps = int(get_defaults_for_model(video_model).get("fps")
                  or self.config.video.default_fps)
        max_frames = int(getattr(self.config.video, "max_num_frames", 97))
        num_frames = min(int(float(scene.duration or 4.0) * fps), max_frames)
        # LTX only generates 8n+1 frame counts and silently rounds DOWN —
        # requesting 96 yielded 89 frames (3.71s clips for 4.0s scenes, a
        # 15s shortfall across a 50-clip video). Snap UP to the next 8n+1.
        num_frames = min(((num_frames - 1 + 7) // 8) * 8 + 1, max_frames)
        planned = num_frames / fps
        want = float(scene.duration or 4.0)
        if want - planned > 0.25:
            logger.warning(
                f"[Pipeline] Scene {scene.scene_number}: {want:.1f}s requested but "
                f"frame cap renders {planned:.2f}s — plan more/shorter scenes "
                f"(ceiling {self._max_clip_seconds(video_model):.2f}s/clip)")
        return num_frames, fps, planned

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
            self._progress.project_id = project_id  # run_state.json journal target
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
                    style_prompt="upbeat background music, strong rhythm, highly enjoyable, instrumental",
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
        with self._exclusive("music"):
            return self._generate_music_variants_impl(
                project_id, count=count, style=style, lyrics=lyrics,
                engine=engine, vocals=vocals, offset=offset)

    def _generate_music_variants_impl(
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
            self._progress.project_id = project_id  # run_state.json journal target
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
            lyric_cues = []  # (planned duration, lyric line) per assembled clip
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
                    lyric_cues.append((float(scene.duration or 0),
                                       str(notes.get("lyric_text") or "")))

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

            # Lyrics subtitle file (.srt) next to the final video — song
            # projects only; best-effort, never fails the render.
            try:
                self._write_lyrics_srt(project, project_dir, lyric_cues,
                                       result.total_duration)
            except Exception as srt_err:
                logger.warning(f"[Release] lyrics .srt failed (non-fatal): {srt_err}")

            # CocoMelon-style karaoke captions: each lyric line on screen with
            # a word-by-word color fill as it is sung (whisper word timings),
            # burned into final_render.mp4; the caption-less master is kept as
            # final_render_nocaptions.mp4. Best-effort, never fails the render.
            try:
                if music_path and (project.lyrics or "").strip():
                    from app.services import karaoke, lyric_sync
                    self._emit_progress(
                        phase=PipelinePhase.ASSEMBLING, project_id=project_id,
                        message="Karaoke captions: syncing lyrics to the vocals...")
                    kcues = lyric_cues
                    if not any(text for _, text in kcues):
                        kcues = self._cues_from_lyrics(
                            project.lyrics, [d for d, _ in lyric_cues])
                    profile = self.director.load_channel_profile(
                        project.channel.slug) or {}
                    karaoke.add_to_render(
                        self.config.paths.ffmpeg_bin, result.output_path,
                        music_path, kcues,
                        self.config.generation.transition_duration,
                        result.total_duration,
                        language=lyric_sync.whisper_language(
                            profile.get("language", "")))
                    # plain lyrics next to the render, for CapCut-style edits
                    (project_dir / "lyrics.txt").write_text(
                        project.lyrics, encoding="utf-8")
            except Exception as k_err:
                logger.warning(f"[Release] karaoke captions failed (non-fatal): {k_err}")

            # Upload-ready assets: metadata.json + auto-picked thumbnail.jpg
            self._write_release_assets(
                project, project_dir, result.output_path,
                result.total_duration, out_resolution)

            # Production report: what was made, at what quality, in how long
            self.write_production_report(session, project, project_dir, {
                "total_duration": result.total_duration,
                "file_size_mb": result.file_size_mb,
                "render_time": result.render_time,
                "resolution": out_resolution,
            })

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

    # ── Narration mode: music bed helper ─────────────────────────────

    # Mood → instrumental ACE-Step style hints (keeps things tasteful
    # under a narration — never competes with the voice).
    _MOOD_MUSIC = {
        "curious":  "light plucked acoustic guitar, gentle piano chords, "
                    "inquisitive feeling, subtle celesta",
        "tense":    "low cello drone, sparse dark synth pads, suspenseful "
                    "atmosphere, slow heartbeat pulse",
        "warm":     "soft fingerpicked acoustic guitar, gentle strings swell, "
                    "cozy warm folk feel, muted percussion",
        "epic":     "slow orchestral build, deep brass, cinematic strings, "
                    "timpani rumble, epic documentary score",
        "neutral":  "minimal ambient pad, soft piano, clean and transparent, "
                    "corporate background music",
    }

    def _narration_music_prompt(self, project_id: str) -> str:
        """Build an instrumental ACE-Step prompt from the narration script's
        dominant mood.  Falls back to a generic documentary bed."""
        import json as _json
        from collections import Counter
        session = get_session()
        try:
            p = session.query(Project).get(project_id)
            script = _json.loads(p.narration_script or "{}") if p else {}
        finally:
            session.close()

        moods = Counter()
        style = script.get("style", "explainer")
        for ch in script.get("chapters", []):
            for b in ch.get("beats", []):
                moods[b.get("mood", "neutral")] += 1

        dominant = moods.most_common(1)[0][0] if moods else "neutral"
        base = self._MOOD_MUSIC.get(dominant, self._MOOD_MUSIC["neutral"])

        # Style-level flavour
        if style == "documentary":
            base += ", documentary score, real instruments, cinematic"
        elif style == "tutorial":
            base += ", clean lo-fi study beat, soft keyboard"
        else:
            base += ", modern explainer background music"

        return (f"{base}, instrumental only, no vocals, no singing, "
                f"no lyrics, background underscore, mix-friendly, "
                f"no drums overpowering, low energy")

    def _ensure_narration_music_bed(self, project_id: str):
        """Generate an instrumental music bed for a narration project if none
        exists. Follows the same precedence as song mode:
        user-supplied file → existing MusicTrack → ACE-Step generate."""
        # 1. user-supplied music wins
        user_music = self._find_user_audio(project_id, "music")
        if user_music:
            logger.info(f"[Pipeline] Narration using user-supplied music: "
                        f"{user_music}")
            return

        # 2. existing active track (resume case)
        session = get_session()
        try:
            existing = session.query(MusicTrack).filter(
                MusicTrack.project_id == project_id,
                MusicTrack.is_active == True,
            ).first()
            if existing and existing.output_path and \
                    Path(existing.output_path).exists():
                logger.info(f"[Pipeline] Narration reusing existing music: "
                            f"{existing.output_path}")
                return
            # grab duration for the prompt
            project = session.query(Project).get(project_id)
            dur = int(project.duration_target) + 5 if project else 120
        finally:
            session.close()

        # 3. generate via ACE-Step (instrumental, mood-matched)
        style_prompt = self._narration_music_prompt(project_id)
        logger.info(f"[Pipeline] Generating narration music bed: "
                    f"'{style_prompt[:80]}…'")
        self.generate_music(
            project_id,
            style=style_prompt,
            lyrics="",
            vocals=False,
        )

    # ── Narration mode: duration-exact render ──────────────────────────

    def render_narration(self, project_id: str, resolution: Optional[str] = None):
        with self._exclusive("render"):
            self._progress.project_id = project_id
            return self._render_narration_impl(project_id, resolution)

    def _render_narration_impl(self, project_id: str, resolution: Optional[str] = None):
        """Narration-mode assembly: every scene trimmed to its exact
        narration window (straight cuts), 4-bus mix (VO / ducked music /
        per-beat SFX), −14 LUFS master, word-level SRT + chapter markers."""
        import json as _json
        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            if not project.narration_audio_path or \
                    not Path(project.narration_audio_path).exists():
                raise ValueError("No narration master WAV — run narration audio first")

            scenes = session.query(Scene).filter(
                Scene.project_id == project_id,
                Scene.status.in_([SceneStatus.GENERATED, SceneStatus.APPROVED]),
            ).order_by(Scene.scene_number).all()

            project.status = ProjectStatus.ASSEMBLING
            session.commit()
            self._emit_progress(
                phase=PipelinePhase.ASSEMBLING, project_id=project_id,
                message="Assembling narration video (duration-exact)...")

            blocks, sfx_tracks = [], []
            for scene in scenes:
                gen = scene.active_generation
                if not gen:
                    continue
                clip = gen.upscaled_path or gen.output_path
                if not (clip and Path(clip).exists()):
                    continue
                blocks.append({
                    "path": clip,
                    "duration": float(scene.duration or 5.0),
                    "offset": float(scene.narration_start or 0.0),
                    "is_template": scene.scene_type == SceneType.TEMPLATE,
                })
                notes = scene.director_notes or {}
                sfx_path = notes.get("sfx_path")
                if sfx_path and Path(sfx_path).exists():
                    sfx_tracks.append({
                        "path": sfx_path,
                        "offset": float(scene.narration_start or 0.0),
                        "gain_db": float(notes.get("sfx_gain_db", -14.0)),
                    })
            if not blocks:
                raise ValueError("No clips available for narration assembly")

            # optional music bed (ambience under the voice)
            music_path = None
            active_music = session.query(MusicTrack).filter(
                MusicTrack.project_id == project_id,
                MusicTrack.is_active == True,
            ).first()
            if active_music and active_music.output_path and \
                    Path(active_music.output_path).exists():
                music_path = active_music.output_path

            project_dir = self.config.paths.projects_dir / project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(project_dir / "final_render.mp4")
            out_resolution = resolution or project.channel.target_resolution

            result = self.assembler.assemble_narration(
                blocks=blocks,
                output_path=output_path,
                narration_path=project.narration_audio_path,
                music_path=music_path,
                sfx_tracks=sfx_tracks,
                resolution=out_resolution,
            )

            session.add(RenderJob(
                project_id=project_id, resolution=out_resolution,
                output_path=result.output_path, status=RenderStatus.COMPLETED,
                progress_pct=100.0,
                render_settings={
                    "mode": "narration",
                    "total_duration": result.total_duration,
                    "file_size_mb": result.file_size_mb,
                    "render_time": result.render_time,
                    "clip_count": len(blocks),
                    "sfx_count": len(sfx_tracks),
                    "music_bed": bool(music_path),
                }))
            project.output_path = result.output_path
            project.status = ProjectStatus.RENDERED

            # QA: duration must track the narration master (<0.5s drift)
            try:
                from app.services import qa as qa_svc
                from dataclasses import asdict as _asdict
                narr_dur = self.tts._get_audio_duration(project.narration_audio_path)
                drift = abs(result.total_duration - narr_dur)
                self._qa_notes["narration_sync"] = {
                    "narration_sec": round(narr_dur, 2),
                    "video_sec": round(result.total_duration, 2),
                    "drift_sec": round(drift, 3),
                    "ok": drift < 0.5,
                }
                if drift >= 0.5:
                    logger.warning(f"[QA] Narration/video drift {drift:.2f}s")
                final_qa = qa_svc.check_final(
                    self.config.paths.ffmpeg_bin, result.output_path,
                    narr_dur, expect_audio=True)
                self._qa_notes["final"] = _asdict(final_qa)
                qa_svc.write_report(project_dir, self._qa_notes)
            except Exception as qa_err:
                logger.warning(f"[QA] narration final check failed: {qa_err}")

            # word-level SRT + chapter markers from the timing sidecar
            try:
                self._write_narration_srt_and_chapters(project, project_dir)
            except Exception as srt_err:
                logger.warning(f"[Release] narration srt/chapters failed: {srt_err}")

            self._write_release_assets(
                project, project_dir, result.output_path,
                result.total_duration, out_resolution)
            self.write_production_report(session, project, project_dir, {
                "total_duration": result.total_duration,
                "file_size_mb": result.file_size_mb,
                "render_time": result.render_time,
                "resolution": out_resolution,
            })
            session.commit()

            self._emit_progress(
                phase=PipelinePhase.DONE, project_id=project_id, percent=100.0,
                message=f"Narration render complete: {result.total_duration:.0f}s, "
                        f"{result.file_size_mb:.1f}MB")
        except Exception as e:
            logger.error(f"[Pipeline] Narration render failed: {e}", exc_info=True)
            try:
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = str(e)
                    session.commit()
            except Exception:
                pass
            self._emit_progress(phase=PipelinePhase.ERROR, error=str(e),
                                message=f"Narration render failed: {e}")
            raise
        finally:
            session.close()

    def _write_narration_srt_and_chapters(self, project, project_dir: Path):
        """final_render.srt from word timestamps (caption-sized lines) +
        chapters.txt ("00:00 Title" per line, ready for the description)."""
        import json as _json
        timing_path = project_dir / "narration" / "narration_timing.json"
        if not timing_path.exists():
            return
        timing = _json.loads(timing_path.read_text(encoding="utf-8"))

        def ts(sec: float) -> str:
            ms = int(round(sec * 1000))
            h, rem = divmod(ms, 3600000)
            m, rem = divmod(rem, 60000)
            s, ms = divmod(rem, 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        # group words into caption lines (≤42 chars or ≥3.5s per cue)
        words = timing.get("words", [])
        cues, cur, cur_start = [], [], None
        for w in words:
            if cur_start is None:
                cur_start = w["start"]
            cur.append(w)
            line = " ".join(x["word"] for x in cur)
            if len(line) >= 42 or (w["end"] - cur_start) >= 3.5:
                cues.append((cur_start, w["end"], line))
                cur, cur_start = [], None
        if cur:
            cues.append((cur_start, cur[-1]["end"],
                         " ".join(x["word"] for x in cur)))

        if cues:
            srt_lines = []
            for i, (a, b, text) in enumerate(cues, 1):
                srt_lines += [str(i), f"{ts(a)} --> {ts(b)}", text, ""]
            (project_dir / "final_render.srt").write_text(
                "\n".join(srt_lines), encoding="utf-8")
            logger.info(f"[Release] narration SRT: {len(cues)} cues")

        # chapter markers
        try:
            from app.services import narration_scenes
            script = _json.loads(project.narration_script or "{}")
            markers = narration_scenes.chapter_markers(script, timing.get("beats", []))
            if markers:
                # YouTube needs the first chapter at 00:00
                if markers[0][0] > 0.01:
                    markers.insert(0, (0.0, "Intro"))
                lines = []
                for start, title in markers:
                    m, s = divmod(int(start), 60)
                    lines.append(f"{m:02d}:{s:02d} {title}")
                (project_dir / "chapters.txt").write_text(
                    "\n".join(lines), encoding="utf-8")
                logger.info(f"[Release] chapters.txt: {len(lines)} chapters")
        except Exception as ch_err:
            logger.warning(f"[Release] chapters failed: {ch_err}")

    # ── Narration mode: full auto ──────────────────────────────────────

    def run_full_auto_narration(self, project_id: str):
        """Narration mode end-to-end: script+safety → narration audio+scenes
        → visuals (batch) → upscale → SFX (if available) → render. Resumable:
        each phase is skipped when its artifact already exists."""
        with self._exclusive("full-auto"):
            self._progress.project_id = project_id
            return self._run_full_auto_narration_impl(project_id)

    def _run_full_auto_narration_impl(self, project_id: str):
        logger.info(f"[Pipeline] Full-auto NARRATION run for {project_id}")
        try:
            from app.services import qa as qa_svc
            self._qa_notes = {"scenes": []}
            pf = qa_svc.preflight(self.config)
            self._qa_notes["preflight"] = pf.to_dict()
            if not pf.ok:
                self._emit_progress(phase=PipelinePhase.ERROR, error="preflight",
                                    message=f"Preflight failed: {pf.summary()}")
                return

            session = get_session()
            project = session.query(Project).get(project_id)
            has_script = bool(project.narration_script)
            has_audio = bool(project.narration_audio_path and
                             Path(project.narration_audio_path).exists())
            has_scenes = session.query(Scene).filter(
                Scene.project_id == project_id).count() > 0
            session.close()

            # Phase 1: script (includes the universal safety gate)
            if not has_script:
                self.generate_narration_script(project_id)
            # Safety enforcement even when the script pre-exists
            self.ensure_safety(project_id)

            # Phase 2: narration master + beat-planned scenes
            if not (has_audio and has_scenes):
                self.generate_narration_audio(project_id)

            # Engine routing: LTX Director renders the whole video natively
            # (narration lines become spoken dialogue per segment).
            if self._project_video_engine(project_id) == "ltx_director":
                logger.info("[Pipeline] video_engine=ltx_director (narration) → "
                            "stills batch + one-shot multi-director render")
                self._ensure_scene_stills(project_id)
                self.generate_ltx_director(project_id)
                return

            # Phase 3: visuals — batch mode, VRAM-safe resolution
            session = get_session()
            p = session.query(Project).get(project_id)
            inline = bool(getattr(p, "upscale_inline", True))
            session.close()
            self.start_generation(project_id, width=832, height=480,
                                  batch=True, upscale_inline=inline)

            # Phase 4: upscale stragglers
            try:
                self.start_upscale(project_id)
            except Exception as up_err:
                logger.warning(f"[Pipeline] Upscale failed (continuing): {up_err}")

            # Phase 5: per-clip SFX (best-effort; needs MMAudio installed)
            try:
                self.generate_sfx(project_id)
            except Exception as sfx_err:
                logger.warning(f"[Pipeline] SFX pass skipped: {sfx_err}")

            # Phase 5b: instrumental music bed (ACE-Step)
            try:
                self._ensure_narration_music_bed(project_id)
            except Exception as mus_err:
                logger.warning(f"[Pipeline] Music bed skipped: {mus_err}")

            # Phase 6: duration-exact render + captions + release assets
            self.render_narration(project_id)
            logger.info(f"[Pipeline] Narration full-auto complete: {project_id}")

        except PipelinePaused:
            logger.info(f"[Pipeline] Narration run paused for {project_id}")
        except SafetyBlocked as sb:
            logger.warning(f"[Pipeline] {sb}")
        except Exception as e:
            logger.error(f"[Pipeline] Narration full-auto failed: {e}", exc_info=True)
            self._emit_progress(phase=PipelinePhase.ERROR, error=str(e),
                                message=f"Pipeline failed: {e}")

    def _project_video_engine(self, project_id: str) -> str:
        """Per-project generation engine: 'clips' (default, one 5-6s ComfyUI
        job per scene) or 'ltx_director' (multi-director one-shot workflow)."""
        session = get_session()
        try:
            p = session.query(Project).get(project_id)
            return (getattr(p, "video_engine", None) or "clips") if p else "clips"
        finally:
            session.close()

    def _ensure_scene_stills(self, project_id: str):
        """Guarantee every scene has a reference still (batch image phase) —
        the LTX Director engine attaches one still per segment, so all scenes
        need an image before the workflow is built."""
        session = get_session()
        try:
            scenes = session.query(Scene).filter(
                Scene.project_id == project_id,
            ).order_by(Scene.scene_number).all()
            images_dir = self.config.paths.projects_dir / project_id / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self._pregenerated_stills = {}
            # 16:9 stills — the director crops references to 1280x720, so
            # matching aspect avoids losing composition
            self._pre_generate_stills(scenes, images_dir, session,
                                      width=1280, height=720)
        finally:
            session.close()

    def _mux_song_over_ltx(self, project_id: str):
        """Song projects on the LTX Director engine: the pipeline's normal
        music phase (user file > existing active track > generate), then the
        song replaces the director video's native audio in final_render.mp4."""
        import subprocess
        session = get_session()
        project = session.query(Project).get(project_id)
        ptype = getattr(project, "project_type", "song") if project else "song"
        video_path = project.output_path if project else None
        session.close()
        if ptype != "song":
            return
        if not video_path or not Path(video_path).exists():
            raise RuntimeError("LTX Director video missing — cannot mux song")

        # Music: user file > existing generated track (resume) > generate
        music_path = self._find_user_audio(project_id, "music")
        if music_path:
            logger.info(f"[Pipeline] Using your custom song: {music_path}")
        else:
            session = get_session()
            existing = session.query(MusicTrack).filter(
                MusicTrack.project_id == project_id,
                MusicTrack.is_active == True,
            ).order_by(MusicTrack.created_at.desc()).first()
            music_path = existing.output_path if existing else None
            session.close()
            if music_path and Path(music_path).exists():
                logger.info(f"[Pipeline] Reusing existing music track: {music_path}")
            else:
                music_path = self.generate_music(project_id)

        self._emit_progress(
            phase=PipelinePhase.ASSEMBLING, project_id=project_id,
            message="Muxing song over the LTX Director video…")
        tmp = str(Path(video_path).with_suffix(".muxtmp.mp4"))
        subprocess.run(
            [str(self.config.paths.ffmpeg_bin), "-y",
             "-i", video_path, "-i", music_path,
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-af", "apad", "-shortest", tmp],
            check=True, capture_output=True)
        Path(tmp).replace(video_path)

        session = get_session()
        project = session.query(Project).get(project_id)
        if project:
            project.status = ProjectStatus.RENDERED
            session.commit()
        session.close()
        self._emit_progress(
            phase=PipelinePhase.DONE, project_id=project_id, percent=100.0,
            message=f"Song video complete: {video_path}")

    def make_urdu_version(self, project_id: str) -> Optional[str]:
        """Second-language release: when the project has lyrics_urdu, generate
        a Hindi/Urdu vocal track (same style brief) and mux it over the SAME
        final video → final_render_urdu.mp4. English final_render.mp4 stays."""
        import subprocess
        session = get_session()
        project = session.query(Project).get(project_id)
        lyr = (getattr(project, "lyrics_urdu", None) or "").strip() if project else ""
        video_path = project.output_path if project else None
        style = (project.music_style or "") if project else ""
        engine = (getattr(project, "music_model", None) or "auto") if project else "auto"
        duration = int(project.duration_target) if project else 60
        session.close()
        if not lyr:
            return None
        if not video_path or not Path(video_path).exists():
            logger.warning("[Pipeline] Urdu version skipped — no rendered video yet")
            return None

        project_dir = self.config.paths.projects_dir / project_id
        urdu_wav = project_dir / "music_urdu.wav"
        if not urdu_wav.exists():
            self._emit_progress(
                phase=PipelinePhase.MUSIC, project_id=project_id,
                message="Generating Hindi/Urdu song version…")
            urdu_style = ("female vocal singing in Hindi Urdu language with clear "
                          "desi pronunciation, " + style) if style else \
                         "female vocal singing in Hindi Urdu language, cheerful children's nursery rhyme"
            self.music_gen.generate(
                style_prompt=urdu_style,
                duration=duration + 2,
                lyrics=lyr,
                output_path=str(urdu_wav),
                instrumental=False,
                engine=engine,
            )
            self.manager.unload()
            session = get_session()
            session.add(MusicTrack(
                project_id=project_id, style_prompt=urdu_style,
                output_path=str(urdu_wav), duration=float(duration),
                is_active=False))
            session.commit()
            session.close()

        out = str(Path(video_path).with_name(
            Path(video_path).stem + "_urdu.mp4"))
        subprocess.run(
            [str(self.config.paths.ffmpeg_bin), "-y",
             "-i", video_path, "-i", str(urdu_wav),
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-af", "apad", "-shortest", out],
            check=True, capture_output=True)
        logger.info(f"[Pipeline] Urdu version ready: {out}")
        self._emit_progress(
            phase=PipelinePhase.DONE, project_id=project_id, percent=100.0,
            message=f"Urdu/Hindi version ready: {out}")
        return out

    def generate_ltx_director(self, project_id: str) -> str:
        """LTX Director multi-segment engine: one continuous long video with
        NATIVE audio/dialogue from the project's stills + prompts. Runs as an
        exclusive phase (it is the heaviest job in the app)."""
        from app.database import get_session, Project, ProjectStatus
        import subprocess
        
        with self._exclusive("ltx-director"):
            self._progress.project_id = project_id
            self.ensure_safety(project_id)
            from app.services.ltx_director import LTXDirectorService
            self._emit_progress(
                phase=PipelinePhase.GENERATING, project_id=project_id,
                message="LTX Director: generating the full 720p video natively (this runs for a long time)...")
            svc = LTXDirectorService(self.manager, self.config)
            session = get_session()
            try:
                # 1. Native 720p generation
                raw_out = svc.generate_for_project(project_id)
                
                # 2. Upscale to 1080p via FFmpeg Lanczos
                self._emit_progress(
                    phase=PipelinePhase.UPSCALING, project_id=project_id,
                    message="LTX Director: upscaling final video to 1080p...")
                
                project_dir = self.config.paths.projects_dir / project_id
                final_out = str(project_dir / "final_render.mp4")
                
                ffmpeg_cmd = [
                    str(self.config.paths.ffmpeg_bin), "-y",
                    "-i", raw_out,
                    "-vf", "scale=1920:1080:flags=lanczos",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "copy",
                    final_out
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                
                # 3. QA Check
                self._emit_progress(
                    phase=PipelinePhase.ASSEMBLING, project_id=project_id,
                    message="LTX Director: running final QA checks...")
                    
                project = session.query(Project).get(project_id)
                if project:
                    project.output_path = final_out
                    project.status = ProjectStatus.RENDERED
                    session.commit()
                
                try:
                    from app.services import qa as qa_svc
                    from dataclasses import asdict as _asdict
                    # Pass expect_audio=False since LTX Director might not have generated an audio stream
                    final_qa = qa_svc.check_final(
                        self.config.paths.ffmpeg_bin, final_out,
                        expected_dur=0.0, expect_audio=False)
                    self._qa_notes = getattr(self, "_qa_notes", {})
                    self._qa_notes["final"] = _asdict(final_qa)
                    qa_svc.write_report(project_dir, self._qa_notes)
                except Exception as qa_err:
                    logger.warning(f"[QA] ltx-director final check failed: {qa_err}")

                self._emit_progress(
                    phase=PipelinePhase.DONE, project_id=project_id,
                    percent=100.0,
                    message=f"LTX Director render complete (1080p): {final_out}")
                return final_out
            except Exception as e:
                self._emit_progress(phase=PipelinePhase.ERROR, error=str(e),
                                    message=f"LTX Director failed: {e}")
                raise
            finally:
                session.close()

    def generate_sfx(self, project_id: str):
        """Per-clip synced SFX via MMAudio (own VRAM phase). Soft dependency:
        raises with a clear message when MMAudio isn't installed yet."""
        from app.services.sfx_gen import SFXService
        sfx = SFXService(self.manager, self.config)
        return sfx.generate_for_project(project_id,
                                        progress_cb=self._emit_progress)

    # ── Convenience: Full Auto Pipeline ────────────────────────────────

    def run_full_auto(
        self,
        project_id: str,
        narration_path: Optional[str] = None,
    ):
        # Narration projects have their own phase chain (script→voice→beats→
        # visuals→sfx→exact render). One routing point here means resume,
        # auto-resume, /full-auto and /produce all pick the right pipeline.
        session = get_session()
        try:
            p = session.query(Project).get(project_id)
            ptype = getattr(p, "project_type", "song") if p else "song"
        finally:
            session.close()
        if ptype == "narration":
            return self.run_full_auto_narration(project_id)
        with self._exclusive("full-auto"):
            self._progress.project_id = project_id  # run_state.json journal target
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
                # Crossfades overlap clips by transition_duration per cut, so
                # scene durations must sum PAST the target by that overlap for
                # the rendered video to actually hit the target length.
                td = float(self.config.generation.transition_duration or 0)
                lint_target = float(project.duration_target) \
                    + max(0, len(scenes) - 1) * td
                if scenes:
                    lo = min(lo, lint_target / len(scenes))
                self._qa_notes["lint"] = qa_svc.lint_script(
                    scenes, profile, lint_target, (lo, hi),
                    video_clip_cap=self._max_clip_seconds(project.video_model))
                session.commit()
                session.close()
            except Exception as lint_err:
                logger.warning(f"[Pipeline] Script lint failed (continuing): {lint_err}")

            # UNIVERSAL SAFETY GATE — fresh verdict on the just-written script
            # (every project type), before any GPU generation begins. LLM critic
            # included: overnight runs must not ship policy-risky videos.
            self._emit_progress(
                phase=PipelinePhase.SAFETY, project_id=project_id,
                message="YT safety gate: reviewing script/lyrics/metadata...")
            gate = self.safety.run_gate(project_id, use_llm=True)
            if gate.verdict not in ("pass", "override"):
                session = get_session()
                project = session.query(Project).get(project_id)
                if project:
                    project.status = ProjectStatus.FAILED
                    project.error_log = (
                        f"Safety gate: {gate.verdict} — "
                        + "; ".join(i.get("detail", "") for i in gate.issues[:5]))
                    session.commit()
                session.close()
                self._emit_progress(
                    phase=PipelinePhase.ERROR, error="safety",
                    message=f"Safety gate verdict '{gate.verdict}' — "
                            f"{len(gate.issues)} issue(s); see safety report.")
                return

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

            # Engine routing: LTX Director replaces the clip/upscale/music/
            # render phases with ONE multi-director ComfyUI workflow (native
            # audio + dialogue). Stills are generated first — one reference
            # image is attached to every segment of every director node.
            if self._project_video_engine(project_id) == "ltx_director":
                logger.info("[Pipeline] video_engine=ltx_director → "
                            "stills batch + one-shot multi-director render")
                self._ensure_scene_stills(project_id)
                self.generate_ltx_director(project_id)
                self._mux_song_over_ltx(project_id)  # song projects only
                try:
                    self.make_urdu_version(project_id)
                except Exception as ur_err:
                    logger.warning(f"[Pipeline] Urdu version failed: {ur_err}")
                logger.info(f"[Pipeline] Full auto (LTX Director) complete for {project_id}")
                return

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

            # Phase 7: Hindi/Urdu second-language version (same video, new song)
            try:
                self.make_urdu_version(project_id)
            except Exception as ur_err:
                logger.warning(f"[Pipeline] Urdu version failed: {ur_err}")

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


class SafetyBlocked(Exception):
    """The universal YT-safety gate did not pass for this project. GPU
    generation is refused until the script/lyrics are revised (or a human
    records an override via /safety-override)."""
    pass
