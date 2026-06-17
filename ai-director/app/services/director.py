"""
AI Director — Director Service
Uses Qwen LLM to plan video scenes, write prompts, and evaluate quality.
The "brain" of the entire pipeline.
"""
import json
import logging
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from app.services.model_manager import ModelManager, ModelType

logger = logging.getLogger(__name__)


# ── Scene Plan Data Structures ─────────────────────────────────────────────

@dataclass
class ScenePlan:
    scene_number: int
    scene_type: str            # txt2vid, img2vid, still_pan
    prompt: str
    negative_prompt: str
    duration: float
    camera_motion: str         # static, pan_left, pan_right, zoom_in, zoom_out, tilt_up
    loras: list[str]
    lora_weights: list[float]
    narration_text: str
    director_notes: dict       # transition hints, style cues


@dataclass
class VideoScript:
    title: str
    total_duration: float
    scene_count: int
    scenes: list[ScenePlan]
    music_style: str
    music_mood: str
    thumbnail_prompt: str


# ── System Prompts ─────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are an expert AI Video Director. Your job is to plan YouTube videos that maximize viewer retention and monetization.

You will receive a video title, target duration, and channel profile. You must produce a detailed scene-by-scene breakdown.

## Your Capabilities
You control these generation models:
- **SDXL**: High-quality still images (1024x1024 or landscape). Best for detailed scenes, backgrounds, establishing shots.
- **LTX-Video 2.3**: Motion video clips, 5-7 seconds. Good for: gentle movement, particle effects, camera pans, nature scenes, abstract motion. Bad for: human faces, text, complex interactions, hands.
- **Ken Burns**: Apply zoom/pan animation to a still image. Best for: establishing shots, landscapes, detailed scenes where motion quality matters less.

## Rules for AI-Generated Content
1. NEVER include human faces, hands, or detailed human figures — AI generates these poorly
2. NEVER include readable text in frames — AI cannot render text reliably
3. NEVER include copyrighted characters or branded content
4. Keep each scene visually simple — one main subject, clear composition
5. Use consistent color palette and style across all scenes
6. Plan gentle transitions between scenes (crossfade works best)
7. Each clip should be self-contained (no dependencies on temporal continuity between clips)
8. Describe scenes in vivid detail: colors, lighting, textures, atmosphere, camera angle

## Scene Types
- `txt2vid`: AI generates a video clip from text prompt. Use for scenes needing gentle motion.
- `img2vid`: AI first generates a still image, then animates it. Use for complex scenes that need a strong base image.
- `still_pan`: AI generates a high-quality still, then Ken Burns zoom/pan is applied. Use for establishing shots, detailed backgrounds, or when video quality matters most.

## Output Format
You MUST respond with ONLY valid JSON (no markdown, no explanation, no preamble). Structure:

```json
{
  "title": "Video Title",
  "total_duration": 300,
  "scene_count": 60,
  "scenes": [
    {
      "scene_number": 1,
      "scene_type": "still_pan",
      "prompt": "Detailed visual description...",
      "negative_prompt": "blurry, low quality, text, watermark, human face, hands...",
      "duration": 5.0,
      "camera_motion": "zoom_in",
      "loras": [],
      "lora_weights": [],
      "narration_text": "Optional narration for this scene",
      "director_notes": {
        "transition_in": "fade_from_black",
        "transition_out": "crossfade",
        "style_cue": "warm golden hour lighting",
        "importance": "high"
      }
    }
  ],
  "music_style": "gentle piano lullaby with wind chimes",
  "music_mood": "peaceful, dreamy, magical",
  "thumbnail_prompt": "Best single frame description for YouTube thumbnail"
}
```
"""

YT_KNOWLEDGE_PROMPT = """
## YouTube Monetization Knowledge
- Kids content (Made for Kids = YES): no personalized ads, lower but consistent CPM ($2-5 USD)
- Kids content relies on AUTOPLAY VOLUME — design for binge-watching, gentle pacing, no jarring transitions
- First 30 seconds are critical for retention — start with the most visually appealing scene
- Aim for consistent visual style so YouTube's algorithm recognizes the channel
- 5-8 minute videos are optimal for kids: long enough for ad placement, short enough for attention spans
- Use repetitive but varied patterns — kids love familiar structures with new details
- Background music should be continuous and gentle — sudden silence or loud changes cause drop-offs
- Avoid anything that could trigger restricted mode: no violence, no scary imagery, no dark themes
- For adult content: hook in first 3 seconds, promise value in first 15, deliver throughout
- For motivational/horror channels: emotional peaks every 60-90 seconds maintain retention
- CPM varies by season: Q4 (Oct-Dec) is highest, Q1 (Jan-Mar) is lowest
- Upload consistency matters more than production quality for algorithm favor
- Descriptions and tags should target long-tail keywords kids might accidentally type
"""


# ── Director Service ───────────────────────────────────────────────────────

class DirectorService:
    """
    The AI brain that plans videos.
    Loads Qwen into VRAM, generates scene plans, then unloads.
    """

    def __init__(self, model_manager: ModelManager, config):
        self.manager = model_manager
        self.config = config
        self._channel_profiles: dict[str, dict] = {}

    def load_channel_profile(self, channel_slug: str) -> dict:
        """Load a channel's YAML profile."""
        if channel_slug in self._channel_profiles:
            return self._channel_profiles[channel_slug]

        profile_path = self.config.paths.channels_dir / f"{channel_slug}.yaml"
        if profile_path.exists():
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile = yaml.safe_load(f)
                self._channel_profiles[channel_slug] = profile
                return profile

        logger.warning(f"No profile found for channel: {channel_slug}")
        return {}

    def build_system_prompt(self, channel_slug: str) -> str:
        """Build the full system prompt with channel-specific context."""
        profile = self.load_channel_profile(channel_slug)

        parts = [BASE_SYSTEM_PROMPT, YT_KNOWLEDGE_PROMPT]

        if profile:
            parts.append(f"\n## Channel Profile\n```yaml\n{yaml.dump(profile, default_flow_style=False)}```")

        return "\n\n".join(parts)

    def generate_script(
        self,
        title: str,
        duration: int,
        context: str,
        channel_slug: str,
        available_loras: Optional[list[dict]] = None,
    ) -> VideoScript:
        """
        Generate a complete video scene breakdown.
        Loads LLM → generates → returns structured script.
        Does NOT unload (caller decides when to free VRAM).
        """
        # Load LLM into VRAM
        loaded = self.manager.load(ModelType.LLM)
        llm = loaded.model

        system_prompt = self.build_system_prompt(channel_slug)

        # Build user message
        user_msg = f"""Plan a YouTube video with these specifications:

**Title:** {title}
**Target Duration:** {duration} seconds ({duration // 60} minutes {duration % 60} seconds)
**Context/Notes:** {context}

Calculate the number of scenes needed:
- Average clip duration: 5 seconds
- Total clips needed: approximately {duration // 5}
- Mix of scene types based on channel profile's still_ratio

Generate the complete scene list as JSON."""

        if available_loras:
            lora_list = "\n".join(
                f"- {l['name']}: {l.get('description', '')} (trigger: {l.get('trigger_words', [])})"
                for l in available_loras
            )
            user_msg += f"\n\n**Available LoRAs:**\n{lora_list}"

        # Generate
        logger.info(f"[Director] Generating script for '{title}' ({duration}s)")

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            response_format={"type": "json_object"},
        )

        raw_text = response["choices"][0]["message"]["content"]
        logger.info(f"[Director] Raw response length: {len(raw_text)} chars")

        # Parse JSON
        script_data = self._parse_script_response(raw_text)

        return script_data

    def refine_prompt(
        self,
        original_prompt: str,
        scene_type: str,
        feedback: str,
        channel_slug: str,
    ) -> dict:
        """
        Ask the director to refine a specific scene prompt.
        Used when a clip fails quality check or user requests changes.
        """
        loaded = self.manager.load(ModelType.LLM)
        llm = loaded.model

        system_prompt = self.build_system_prompt(channel_slug)

        user_msg = f"""A scene needs to be regenerated. Please provide an improved prompt.

**Scene Type:** {scene_type}
**Original Prompt:** {original_prompt}
**Feedback/Issue:** {feedback}

Respond with ONLY JSON:
{{
  "prompt": "improved detailed prompt",
  "negative_prompt": "things to avoid",
  "camera_motion": "suggested motion",
  "notes": "what changed and why"
}}"""

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw = response["choices"][0]["message"]["content"]
        return json.loads(self._clean_json(raw))

    def suggest_retry_strategy(
        self,
        scene_number: int,
        original_prompt: str,
        error_log: str,
        retry_count: int,
    ) -> dict:
        """
        When a generation fails, ask the director what to try differently.
        Returns: new prompt, model suggestion, parameter changes.
        """
        loaded = self.manager.load(ModelType.LLM)
        llm = loaded.model

        user_msg = f"""A video generation failed. Suggest a retry strategy.

**Scene #{scene_number}**
**Original Prompt:** {original_prompt}
**Error:** {error_log}
**Retry Attempt:** {retry_count + 1}

Respond with ONLY JSON:
{{
  "new_prompt": "simplified/modified prompt for retry",
  "negative_prompt": "updated negatives",
  "model_suggestion": "ltx-2.3 or sdxl+kenburns",
  "parameter_changes": {{"steps": 10, "cfg": 1.0, "resolution_scale": 0.8}},
  "reasoning": "why this change might fix the issue"
}}"""

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a technical AI video generation expert. Diagnose failures and suggest fixes."},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        raw = response["choices"][0]["message"]["content"]
        return json.loads(self._clean_json(raw))

    # ── Parsing Helpers ────────────────────────────────────────────────

    def _parse_script_response(self, raw_text: str) -> VideoScript:
        """Parse LLM JSON response into VideoScript."""
        clean = self._clean_json(raw_text)

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(f"[Director] JSON parse error: {e}")
            logger.error(f"[Director] Raw text: {raw_text[:500]}")
            raise ValueError(f"Director produced invalid JSON: {e}")

        scenes = []
        for s in data.get("scenes", []):
            scenes.append(ScenePlan(
                scene_number=s.get("scene_number", len(scenes) + 1),
                scene_type=s.get("scene_type", "txt2vid"),
                prompt=s.get("prompt", ""),
                negative_prompt=s.get("negative_prompt", ""),
                duration=float(s.get("duration", 5.0)),
                camera_motion=s.get("camera_motion", "static"),
                loras=s.get("loras", []),
                lora_weights=s.get("lora_weights", []),
                narration_text=s.get("narration_text", ""),
                director_notes=s.get("director_notes", {}),
            ))

        return VideoScript(
            title=data.get("title", ""),
            total_duration=float(data.get("total_duration", 0)),
            scene_count=int(data.get("scene_count", len(scenes))),
            scenes=scenes,
            music_style=data.get("music_style", ""),
            music_mood=data.get("music_mood", ""),
            thumbnail_prompt=data.get("thumbnail_prompt", ""),
        )

    @staticmethod
    def _clean_json(text: str) -> str:
        """Strip markdown fences and whitespace from LLM JSON output."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
