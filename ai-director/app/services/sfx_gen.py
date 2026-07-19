"""
AI Director — SFX Service (MMAudio)
Per-clip SYNCHRONIZED sound effects: MMAudio watches each rendered clip and
generates foley/ambience that lands on the visible action (keyboard sounds on
typing shots, footsteps on footfalls). Runs via ComfyUI (kijai/ComfyUI-MMAudio)
as its own VRAM phase after video generation.

Models (ComfyUI/models/mmaudio):
  mmaudio_large_44k_v2_fp16.safetensors   — main net (best quality, 44kHz)
  mmaudio_vae_44k_fp16.safetensors
  mmaudio_synchformer_fp16.safetensors
  apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors
BigVGAN v2 vocoder auto-downloads on first run (nvidia/bigvgan_v2_44khz...).
"""
import json
import random
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

MMAUDIO_MODEL = "mmaudio_large_44k_v2_fp16.safetensors"
MMAUDIO_VAE = "mmaudio_vae_44k_fp16.safetensors"
MMAUDIO_SYNC = "mmaudio_synchformer_fp16.safetensors"
MMAUDIO_CLIP = "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"

# SFX must never fight the voice or carry its own music/speech
SFX_NEGATIVE = "music, melody, song, speech, voice, talking, singing, narration"


def build_sfx_workflow(video_path: str, prompt: str, duration: float,
                       seed: int = -1, steps: int = 25, cfg: float = 4.5) -> dict:
    """API-format ComfyUI workflow: video → MMAudio → SaveAudio (flac)."""
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    return {
        "1": {"class_type": "MMAudioModelLoader",
              "inputs": {"mmaudio_model": MMAUDIO_MODEL,
                         "base_precision": "fp16"}},
        "2": {"class_type": "MMAudioFeatureUtilsLoader",
              "inputs": {"vae_model": MMAUDIO_VAE,
                         "synchformer_model": MMAUDIO_SYNC,
                         "clip_model": MMAUDIO_CLIP,
                         "mode": "44k", "precision": "fp16"}},
        "3": {"class_type": "VHS_LoadVideoPath",
              "inputs": {"video": str(video_path), "force_rate": 0,
                         "force_size": "Disabled", "custom_width": 512,
                         "custom_height": 512, "frame_load_cap": 0,
                         "skip_first_frames": 0, "select_every_nth": 1}},
        "4": {"class_type": "MMAudioSampler",
              "inputs": {"mmaudio_model": ["1", 0],
                         "feature_utils": ["2", 0],
                         "images": ["3", 0],
                         "duration": round(float(duration), 2),
                         "steps": steps, "cfg": cfg, "seed": seed,
                         "prompt": prompt or "subtle ambient room tone",
                         "negative_prompt": SFX_NEGATIVE,
                         "mask_away_clip": False,
                         "force_offload": True}},
        "5": {"class_type": "SaveAudio",
              "inputs": {"audio": ["4", 0],
                         "filename_prefix": "aidir_sfx"}},
    }


class SFXService:
    """Clip-synced SFX via MMAudio (ComfyUI custom node)."""

    def __init__(self, model_manager, config):
        self.manager = model_manager
        self.config = config

    def _check_available(self, client) -> Optional[str]:
        """Return an error string when MMAudio can't run, else None."""
        models_dir = self.config.paths.models_dir / "mmaudio"
        missing = [f for f in (MMAUDIO_MODEL, MMAUDIO_VAE, MMAUDIO_SYNC,
                               MMAUDIO_CLIP)
                   if not (models_dir / f).exists()]
        if missing:
            return (f"MMAudio weights missing in {models_dir}: {missing} — "
                    f"download from huggingface.co/Kijai/MMAudio_safetensors")
        try:
            req = urllib.request.Request(
                f"{client.base_url}/object_info/MMAudioSampler")
            with urllib.request.urlopen(req, timeout=10) as resp:
                info = json.loads(resp.read())
            if "MMAudioSampler" not in info:
                raise KeyError
        except Exception:
            return ("MMAudioSampler node not registered — the ComfyUI-MMAudio "
                    "custom node is installed but ComfyUI needs a RESTART to "
                    "load it")
        return None

    def generate_for_scene(self, client, clip_path: str, sfx_prompt: str,
                           duration: float, out_path: str) -> str:
        """One clip → one synced SFX track. Returns the saved audio path."""
        wf = build_sfx_workflow(clip_path, sfx_prompt, duration)
        prompt_id = client.submit(wf)
        history = client.wait_for_completion(prompt_id, timeout=600)
        return client.collect_output(history, out_path)

    def generate_for_project(self, project_id: str,
                             progress_cb: Optional[Callable] = None) -> int:
        """SFX for every generated scene that has an sfx_prompt. Stores the
        track path in scene.director_notes['sfx_path'] (picked up by the
        narration assembler at its narration_start offset). Returns count."""
        from app.database import get_session, Scene, SceneStatus
        from app.services.comfyui_client import ComfyUIClient
        from app.services.pipeline import PipelinePhase

        client = ComfyUIClient()
        if not client.ensure_running():
            raise RuntimeError("ComfyUI is not running — SFX needs it")
        err = self._check_available(client)
        if err:
            raise RuntimeError(err)

        # Own VRAM phase: drop any app-side model, clear ComfyUI's cache
        try:
            self.manager.unload()
        except Exception:
            pass
        client.free_vram()

        session = get_session()
        try:
            scenes = session.query(Scene).filter(
                Scene.project_id == project_id,
                Scene.status.in_([SceneStatus.GENERATED, SceneStatus.APPROVED]),
            ).order_by(Scene.scene_number).all()

            todo = []
            for s in scenes:
                notes = s.director_notes or {}
                if notes.get("sfx_path") and Path(notes["sfx_path"]).exists():
                    continue  # resume: already done
                if not (s.sfx_prompt or "").strip():
                    continue  # beat explicitly silent
                gen = s.active_generation
                clip = (gen.upscaled_path or gen.output_path) if gen else None
                # SFX watches the RAW clip (smaller = faster CLIP/sync encode);
                # timing is identical to the upscaled version
                if gen and gen.output_path and Path(gen.output_path).exists():
                    clip = gen.output_path
                if clip and Path(clip).exists():
                    todo.append((s, clip))

            if not todo:
                logger.info("[SFX] Nothing to do (no scenes with sfx_prompt)")
                return 0

            sfx_dir = self.config.paths.projects_dir / project_id / "sfx"
            sfx_dir.mkdir(parents=True, exist_ok=True)

            done = 0
            for i, (scene, clip) in enumerate(todo):
                if progress_cb:
                    progress_cb(phase=PipelinePhase.GENERATING,
                                project_id=project_id,
                                message=f"SFX {i + 1}/{len(todo)}: "
                                        f"{(scene.sfx_prompt or '')[:50]}",
                                percent=100.0 * i / len(todo))
                out = str(sfx_dir / f"sfx_{scene.scene_number:03d}.flac")
                try:
                    self.generate_for_scene(
                        client, clip, scene.sfx_prompt,
                        float(scene.duration or 5.0), out)
                    notes = dict(scene.director_notes or {})
                    notes["sfx_path"] = out
                    scene.director_notes = notes
                    session.commit()
                    done += 1
                except Exception as e:
                    logger.warning(f"[SFX] Scene {scene.scene_number} failed "
                                   f"(non-fatal): {e}")
                    session.rollback()

            logger.info(f"[SFX] Generated {done}/{len(todo)} SFX tracks")
            if progress_cb:
                progress_cb(phase=PipelinePhase.IDLE, project_id=project_id,
                            percent=100.0,
                            message=f"SFX done: {done}/{len(todo)} tracks")
            return done
        finally:
            session.close()
            client.free_vram()
