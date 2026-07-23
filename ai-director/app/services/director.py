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
    description: str = ""        # SEO YouTube description
    tags: list[str] = None       # SEO tags / keywords
    hashtags: list[str] = None   # #hashtags for title/description

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.hashtags is None:
            self.hashtags = []


# ── System Prompts ─────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are an ELITE AI Video Director — operating in the top 0.1% of three crafts at once:
  1. A YouTube growth strategist who reverse-engineers what makes faceless/animated channels go viral and retain audiences.
  2. A master children's story writer who can craft a tight, emotionally satisfying moral story with a clear arc.
  3. An AI art director who knows EXACTLY what a local image/video model on a 16GB GPU can render beautifully — and what it cannot.

You will receive a video title, target duration, and a Channel Profile. Produce a complete, production-ready scene-by-scene plan that another automated pipeline will execute with NO human cleanup. Every prompt must be directly usable.

## The Local Generation Stack You Are Directing (16GB VRAM — plan within its limits)
- **SDXL** → high-quality STILL images (illustration / storybook style). This is your strongest, most reliable tool. Use it heavily.
- **LTX-Video 2.3 distilled** → short 2–6s motion clips. EXCELLENT at: ambient/environmental motion (drifting clouds, falling leaves, sparkles, flowing water, swaying grass, gentle camera moves, flickering light). POOR at: faces, hands, fingers, lip-sync, fast or complex action, multiple interacting characters, anything with text.
- **Ken Burns** → slow zoom/pan applied to an SDXL still. Your go-to for character close-ups, emotional beats, and detailed scenes where motion quality matters more than literal movement.

## Character System (CRITICAL for story channels)
Local AI cannot keep a character's face consistent across clips. Therefore:
- Prefer **stylized talking ANIMALS or friendly CREATURES** as protagonists (a wise owl, a little blue parrot, a brave rabbit, a kind elephant). These render far more consistently than humans and fit the kids-story niche perfectly.
- Maintain a **Character Bible**: pick 2–4 exact physical descriptors per character (species, color, one signature item) and REPEAT THOSE EXACT WORDS in every scene prompt the character appears in. Consistency comes from repetition, not from referencing "the same character".
- Keep the SAME art style phrase in every prompt (e.g. "soft 3D storybook render, pastel palette, rounded shapes").
- One clear main subject per scene. Never stage complex multi-character interactions in a single clip — cut between single-subject shots instead.

## Visual Prompt Rules
1. Write ALL visual prompts (`prompt` + `negative_prompt`) in ENGLISH — the models are English-trained and produce much better images. (Narration is a separate field and follows the channel's language.)
2. NO readable text in frames, NO copyrighted/branded characters, NO human faces in close-up.
3. Describe RICHLY and cinematically (45–85 words): exact character descriptors + a subtle ACTION, then layered depth (foreground/midground/background), specific textures (fluffy fur, dewy leaves, velvety petals), volumetric/golden-hour lighting with rim light and soft bokeh, atmosphere (floating pollen, drifting sparkles, soft mist), clear composition/camera angle, the art style, and SDXL quality boosters ("highly detailed, cinematic, soft volumetric lighting, depth of field, beautifully rendered"). Thin, generic prompts are NOT acceptable — make each frame look like a polished storybook illustration.
4. Keep a consistent palette and art style across ALL scenes so the channel reads as one cohesive brand.
5. Each clip is self-contained — assume clips are cut together with crossfades; no shot may depend on motion continuity from the previous shot.

## Scene-Type Strategy (choose deliberately per beat)
- `img2vid` → **USE THIS FOR ALMOST EVERY SCENE.** A consistent SDXL still is generated, then LTX ANIMATES it into a real moving video clip — the character actually moves/acts while keeping a consistent look. This is the primary tool: real motion + consistency. Write the prompt so the character is DOING something (reaching, hopping, smiling, looking around, wind blowing) so there is motion to animate.
- `txt2vid` → LTX motion clip with no base image. Use for pure ambient/environment beats (weather, water, particles, wide nature shots) where no consistent character is needed.
- `still_pan` → SDXL still + Ken Burns (static image, camera pan only — NO real motion). Use rarely, only for a deliberate quiet "title card" style beat.
Default to `img2vid` unless the channel profile says otherwise. Every character scene MUST describe an action/movement.

## Retention & Story Engineering
- Scene 1 is the HOOK: the single most beautiful/intriguing frame + a narration line that opens a curiosity gap or promises the payoff.
- Structure a real arc: setup → rising tension → turning point → resolution → clear moral. For kids, keep beats short and the moral explicit and warm.
- Think like a REAL film director — choose framing based on the STORY BEAT, not a formula:
  • WIDE / ESTABLISHING SHOT → scene openers, location reveals, showing scale/environment. Shows full body + surroundings.
  • MEDIUM WIDE SHOT → action beats, two characters interacting, walking. Shows full body with context.
  • MEDIUM SHOT → conversation, gentle emotion, a character doing something with their hands. Waist-up framing.
  • MEDIUM CLOSE-UP → emotional reaction, tender moment. Chest-up. Use sparingly.
  • CLOSE-UP → rare. Only for a single powerful emotional beat (a tear, a smile, holding something precious). NEVER the default.
- Every prompt MUST begin with the shot type (e.g. "wide establishing shot," or "medium shot,"). This is critical — without it the model defaults to zoomed-in face crops which look bad.
- NEVER use the same shot type for consecutive scenes. Alternate deliberately. A typical 5-scene sequence might be: wide → medium → medium wide → medium close-up → wide.
- Bias toward wider shots overall (~60% wide/medium-wide, ~30% medium, ~10% close-up). Environment context makes AI renders look dramatically better.
- End on a gentle, satisfying, slightly loopable note (helps autoplay/binge metrics).

## Output Format
Respond with ONLY valid JSON — no markdown, no explanation, no preamble. The pipeline parses these exact fields:

```json
{
  "title": "Click-worthy title in the channel's NARRATION language (curiosity gap, emotional, age-appropriate)",
  "total_duration": 300,
  "scene_count": 60,
  "scenes": [
    {
      "scene_number": 1,
      "scene_type": "still_pan",
      "prompt": "ENGLISH visual description with exact character descriptors, setting, lighting, palette, art style...",
      "negative_prompt": "blurry, low quality, text, watermark, human face, hands, deformed, extra limbs, harsh lighting...",
      "duration": 5.0,
      "camera_motion": "zoom_in",
      "loras": [],
      "lora_weights": [],
      "narration_text": "Narration for this scene IN THE CHANNEL'S NARRATION LANGUAGE (empty string if a music/ambient-only beat)",
      "director_notes": {
        "transition_in": "fade_from_black",
        "transition_out": "crossfade",
        "style_cue": "warm golden hour lighting",
        "importance": "high"
      }
    }
  ],
  "music_style": "concise music-generation prompt emphasizing strong rhythm, clear beat, and highly enjoyable musicality (instruments like drums/bass, genre, upbeat tempo)",
  "music_mood": "energetic, upbeat, highly enjoyable, catchy",
  "thumbnail_prompt": "ENGLISH description of the single most clickable frame for the thumbnail",
  "description": "SEO-optimized YouTube description in the narration language: a hooky first line, a 2-3 sentence summary, the moral/value, then relevant keywords woven in naturally. End with 3-5 hashtags.",
  "tags": ["15-25 SEO tags mixing the narration language AND English: topic, character, moral, niche, long-tail search phrases viewers actually type"],
  "hashtags": ["#3to5", "#relevant", "#hashtags"]
}
```

## SEO REQUIREMENTS (do NOT skip — this drives discovery)
- `title`: front-load the hook + the most-searched keyword; keep it punchy and clickable, not generic.
- `description`: first line is a hook (it shows in search). Include the main keyword in the first sentence. Weave in long-tail keywords naturally. Add a clear moral/value line. Finish with hashtags.
- `tags`: think like the audience's search bar — include the character, the moral/lesson, the niche ("moral stories for kids", "bedtime story"), and BOTH the narration language and English variants. 15-25 tags.
- Never keyword-stuff awkwardly; it must read naturally to a human.
"""

YT_KNOWLEDGE_PROMPT = """
## YouTube Strategy Knowledge (apply like a top-0.1% creator)
RETENTION (the metric that drives everything):
- The first 3–5 seconds decide the video. Open on your strongest visual + a narration hook (a question, a promise, or a tiny mystery).
- Re-hook every 30–60s with a new visual or a small story turn so the watch-time curve stays flat.
- Smooth, continuous experience: gentle crossfades, continuous background music, no jarring cuts or sudden silence (silence = drop-off).
- Satisfying, loopable endings lift autoplay and session time.

KIDS / FAMILY CONTENT (Made for Kids = YES):
- No personalized ads → CPM is lower but volume + watch time + autoplay carry revenue. Design for BINGE.
- Gentle pacing, warm tone, zero scary/violent/dark imagery (avoids restricted mode and protects the channel).
- Familiar, repeatable structures with fresh details each episode — kids love predictability; the algorithm loves a recognizable format.
- 4–8 minutes is the sweet spot: room for an ad break, short enough to finish.

TITLES, THUMBNAIL & SEO:
- Title = curiosity gap + emotional word + clarity. Write it in the channel's narration language; keep it skimmable.
- Thumbnail frame: one clear subject, bright high-contrast colors, big emotion, minimal clutter, no tiny text.
- Think in searchable, long-tail topics (moral, lesson, character, theme) so the description/tags can target them later.

CONSISTENCY > PERFECTION:
- A recognizable, repeatable visual identity and a steady upload cadence beat one-off high production. Keep the channel's look identical across videos.
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

        # Language directive — visual prompts stay English (model quality),
        # narration + title + SEO follow the channel's language.
        language = (profile or {}).get("language", "English")
        lang_lower = str(language).lower()
        if "roman" in lang_lower:
            lang_rule = (
                "- This channel uses **Roman Urdu** — Urdu written in ENGLISH/LATIN letters only. "
                "Do NOT use Urdu/Arabic script anywhere. "
                "Example narration: \"Aik dafa ka zikr hai, aik neela tota tha jo bohat lalchi tha.\"\n"
                "- Write `narration_text`, `title`, `description`, and `tags` in Roman Urdu "
                "(plus English keywords in tags for search reach)."
            )
        elif "urdu" in lang_lower:
            lang_rule = (
                "- Write `narration_text`, `title`, `description` in Urdu (اردو script).\n"
                "- `tags` should mix Urdu and English keywords."
            )
        else:
            lang_rule = (
                f"- Write `narration_text`, `title`, `description`, and `tags` in {language}."
            )
        parts.append(
            f"\n## Language Directive\n"
            f"- This channel's narration language is: **{language}**.\n"
            f"{lang_rule}\n"
            f"- Write every `prompt` and `negative_prompt` in ENGLISH regardless of narration language.\n"
            f"- `music_style` and `thumbnail_prompt` always in English."
        )

        if profile:
            parts.append(f"\n## Channel Profile\n```yaml\n{yaml.dump(profile, default_flow_style=False, allow_unicode=True)}```")

        # Channel-mandated visual direction beats every generic rule above
        # (e.g. a "minimal story" channel must not get plot-heavy scene plans).
        visual_direction = (profile or {}).get("visual_direction")
        if visual_direction:
            if isinstance(visual_direction, list):
                visual_direction = "\n".join(f"- {v}" for v in visual_direction)
            parts.append(
                f"\n## Visual Direction (CHANNEL MANDATE — overrides any "
                f"conflicting rule in this prompt, including episode_structure "
                f"and Retention & Story Engineering)\n{visual_direction}"
            )

        # Load cinematography guide for shot selection training
        cine_path = Path(__file__).parent.parent / "knowledge" / "cinematography_guide.yaml"
        if cine_path.exists():
            with open(cine_path, encoding="utf-8") as f:
                parts.append(f"\n## Cinematography & Framing Guide (FOLLOW THIS)\n```yaml\n{f.read()}```")

        return "\n\n".join(parts)

    def generate_script(
        self,
        title: str,
        duration: int,
        context: str,
        channel_slug: str,
        available_loras: Optional[list[dict]] = None,
        num_scenes: Optional[int] = None,
    ) -> VideoScript:
        """
        Generate a complete video scene breakdown.
        Loads LLM → generates → returns structured script.
        Does NOT unload (caller decides when to free VRAM).
        """
        # Free ComfyUI's VRAM first so the 27B LLM gets the whole 16GB GPU
        # (otherwise it spills layers to CPU → slow, ~30% GPU utilization).
        try:
            import time as _t
            from app.services.comfyui_client import ComfyUIClient
            ComfyUIClient().free_vram()
            _t.sleep(5)  # /free is async — let ComfyUI actually release VRAM before
                         # the 13GB LLM loads, otherwise they contend and the LLM
                         # spills to CPU (script gen crawls to several minutes).
        except Exception:
            pass

        # Load LLM into VRAM
        loaded = self.manager.load(ModelType.LLM)
        llm = loaded.model

        system_prompt = self.build_system_prompt(channel_slug)

        # Calculate the number of scenes needed
        if not num_scenes or num_scenes <= 0:
            num_scenes = duration // 5
            avg_duration_str = "approximately 5 seconds"
        else:
            avg_duration_str = f"exactly {duration / num_scenes:.1f} seconds"

        # Build user message
        user_msg = f"""Plan a YouTube video with these specifications:

**Title/Subject:** {title}
**Target Duration:** {duration} seconds ({duration // 60} minutes {duration % 60} seconds)
**Context/Notes:** {context}

CRITICAL RULES:
1. The entire video MUST be about the "Title/Subject" provided above. Do not deviate from this topic!
2. Ensure every single scene follows the narrative and includes the subjects mentioned in the title.
3. Strictly follow all guidelines in the Channel Profile.

Calculate the number of scenes needed:
- Average clip duration: {avg_duration_str}
- Total clips needed: exactly {num_scenes}
- Mix of scene types based on channel profile's still_ratio

OUTPUT SHAPE — return ONE JSON object with this EXACT top level. Do NOT return a single scene object; the "scenes" array MUST contain all {num_scenes} scene objects:
{{"title": "...", "total_duration": {duration}, "scene_count": {num_scenes}, "scenes": [ <<{num_scenes} scene objects>> ], "music_style": "...", "music_mood": "...", "thumbnail_prompt": "...", "description": "...", "tags": ["..."], "hashtags": ["..."]}}

Generate the full object now with all {num_scenes} scenes inside the "scenes" array."""

        if available_loras:
            lora_list = "\n".join(
                f"- {l['name']}: {l.get('description', '')} (trigger: {l.get('trigger_words', [])})"
                for l in available_loras
            )
            user_msg += f"\n\n**Available LoRAs:**\n{lora_list}"

        # Generate
        logger.info(f"[Director] Generating script for '{title}' ({duration}s)")

        # NOTE: non-streaming. The streaming path did not reliably accumulate
        # this (hybrid SSM/reasoning) model's JSON chunks and produced empty
        # output; a single non-streaming call with the json_object grammar is
        # robust and fast on a fully-GPU-resident model.
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            response_format={"type": "json_object"},
            stream=False,
        )

        raw_text = response["choices"][0]["message"].get("content", "") or ""
        logger.info(f"[Director] Raw response length: {len(raw_text)} chars")

        # Parse JSON
        script_data = self._parse_script_response(raw_text)

        return script_data

    def storyboard_from_lyrics(
        self,
        context: str,
        title: str,
        segments: list,
        channel_slug: str,
        seed_key: str,
    ) -> list[dict]:
        """LLM visual director for SONG projects: turn each lyric line into ONE
        concrete scene that (a) shows the exact locked character(s) from the
        project CONTEXT, (b) sits in the CONTEXT's setting, and (c) DEPICTS THE
        LITERAL ACTION of that lyric line (so a brushing song actually shows
        brushing). Replaces the old canned-template builder that ignored both the
        context and the lyrics. Returns the same shape as
        lyric_scenes.build_prompts so the caller is a drop-in swap.

        Loads the LLM and does NOT unload (caller/safety gate frees it) — within
        one run the model stays cached so no second llama_cpp re-init happens."""
        from app.services import master_director

        profile = self.load_channel_profile(channel_slug) or {}
        art_style = profile.get("art_style_phrase", "soft 3D Pixar-style cartoon render")
        palette = profile.get("color_palette", "bright cheerful pastels")
        if isinstance(palette, dict):
            palette = palette.get("playtime") or \
                next(iter(palette.values()), "bright cheerful pastels")
        elif isinstance(palette, list):
            palette = ", ".join(str(p) for p in palette)
        neg = profile.get("negative_prompt_additions",
                          "photorealistic, realistic skin, text, watermark, scary, "
                          "deformed, extra limbs, blurry")
        if isinstance(neg, list):
            neg = ", ".join(neg)

        # numbered lyric lines with their song section
        line_rows = []
        for seg in segments:
            txt = "(instrumental / no words)" if getattr(seg, "is_instrumental", False) \
                else (getattr(seg, "text", "") or "").strip()
            line_rows.append(f'{seg.index + 1}. [{seg.section_type}] "{txt}"')
        lines_block = "\n".join(line_rows)
        n = len(segments)

        # Free ComfyUI VRAM so the LLM gets the full GPU, then load (cached if
        # already resident — no re-init crash within a run).
        try:
            import time as _t
            from app.services.comfyui_client import ComfyUIClient
            ComfyUIClient().free_vram()
            _t.sleep(5)
        except Exception:
            pass
        llm = self.manager.load(ModelType.LLM).model

        sys_prompt = (
            "You are an ELITE visual director and storyboard artist for a children's "
            "animated MUSIC VIDEO. You are given (A) a CONTEXT that LOCKS the exact "
            "character(s), setting and art style, and (B) the song's lyric lines in order.\n\n"
            "For EVERY lyric line, write ONE concrete visual scene description that:\n"
            "- Shows the EXACT locked character(s) from the CONTEXT — same species, colours "
            "and signature props. NEVER a human child, NEVER a different animal, NEVER a "
            "generic character.\n"
            "- Takes place in the SETTING named in the CONTEXT (do not invent unrelated "
            "locations like libraries, mosques, meadows, riverbanks).\n"
            "- DEPICTS THE LITERAL ACTION / MEANING OF THAT LYRIC LINE so the sung moment is "
            "clearly visible. If the line is 'brush your teeth', the character is actively "
            "brushing its teeth with its toothbrush; 'show your smile' = it opens wide showing "
            "teeth; 'rinse and spit' = it sips from the cup at the sink. Translate poetic lines "
            "into a concrete visible action.\n"
            "- Is visually DISTINCT from every other scene — vary the pose, the exact beat of "
            "the action, and the framing. No two descriptions may repeat.\n"
            "- Is 25-55 words, concrete and physical (who + exact action + a couple of setting/"
            "light details). ENGLISH only.\n"
            "- Has NO readable text/letters/numbers in the image, no watermark, no human faces; "
            "toddler-safe, warm and cute. Do NOT include camera or shot-size jargon — only "
            "describe what happens in the frame.\n\n"
            f'Respond with ONLY JSON: {{"scenes":[{{"n":1,"desc":"..."}}, ... exactly {n} '
            'objects, one per lyric line, same order]}.'
        )
        user_msg = (
            f"CONTEXT (locked characters / setting / style — obey exactly):\n{context}\n\n"
            f"SONG TITLE: {title}\n\n"
            f"LYRIC LINES — write one matching scene for each, in order ({n} total):\n"
            f"{lines_block}\n"
        )

        logger.info(f"[Director] Storyboarding {n} lyric scenes from context+lyrics")
        resp = llm.create_chat_completion(
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_msg}],
            temperature=0.7,
            max_tokens=self.config.llm.max_tokens,
            response_format={"type": "json_object"},
            stream=False,
        )
        raw = resp["choices"][0]["message"].get("content", "") or ""
        try:
            data = json.loads(self._clean_json(raw))
        except json.JSONDecodeError:
            data = self._salvage_truncated(self._clean_json(raw))
        items = data.get("scenes") or data.get("enhanced") or []
        by_n = {}
        for it in items:
            try:
                by_n[int(it.get("n"))] = str(it.get("desc", "")).strip()
            except (TypeError, ValueError):
                continue

        prompts = []
        for seg in segments:
            desc = by_n.get(seg.index + 1, "").strip()
            if not desc:
                # context-aware fallback for any line the LLM skipped — still
                # anchored to the real lyric, never the old template pool.
                lyric = (getattr(seg, "text", "") or "").strip()
                desc = (f"the locked main character from the context, in the context's "
                        f"setting, clearly performing the action of this moment: \"{lyric}\"")
            guidance = master_director.guidance_for(
                seg.index, len(segments), seg.section_type, seed_key)
            full = (f"{guidance['shot']}, {desc}, {art_style}, {palette}, "
                    f"highly detailed, cinematic, soft volumetric lighting, depth of field, "
                    f"beautifully rendered, warm rim light, gentle bokeh background, "
                    f"clean frame without any text, captions, titles or watermarks, "
                    f"camera: {guidance['camera']}, {guidance['lighting']}, "
                    f"{guidance['mood']} mood, {guidance['composition']}")
            prompts.append({
                "segment_index": seg.index,
                "prompt": full,
                "negative_prompt": neg,
                "camera_motion": ["static", "zoom_in", "pan_left", "pan_right",
                                  "static", "zoom_in"][seg.index % 6],
                "narration_text": "" if getattr(seg, "is_instrumental", False) else seg.text,
                "duration": seg.duration,
                "director_guidance": guidance,
            })
        logger.info(f"[Director] Storyboard built {len(prompts)} context-locked scenes "
                    f"({len(by_n)} from LLM, {len(prompts) - len(by_n)} fallback)")
        return prompts

    def enhance_prompts(self, scenes: list[dict], channel_slug: str) -> list[dict]:
        """Second-pass 'prompt engineer': rewrite each scene's visual prompt to be
        far richer and more cinematic (texture, layered depth, volumetric lighting,
        atmosphere, composition, SDXL quality boosters) while KEEPING the exact
        character descriptors + art style so consistency is preserved.

        `scenes`: [{scene_number, prompt, negative_prompt}]  ->  enhanced same shape.
        One batched LLM call for all scenes.
        """
        profile = self.load_channel_profile(channel_slug)
        art_style = (profile or {}).get("art_style_phrase", "")
        palette = (profile or {}).get("color_palette", "")
        char_bible = (profile or {}).get("character_bible", [])

        try:
            from app.services.comfyui_client import ComfyUIClient
            ComfyUIClient().free_vram()
            import time as _t; _t.sleep(5)
        except Exception:
            pass
        llm = self.manager.load(ModelType.LLM).model

        sys_prompt = f"""You are a WORLD-CLASS AI image prompt engineer for SDXL, specializing in gorgeous children's storybook art.
Rewrite each scene's visual prompt to be DRAMATICALLY richer and more cinematic, while preserving consistency.

KEEP EXACTLY (do not change):
- The character's exact physical descriptors (species, colors, signature items).
- The art style: {art_style or 'soft 3D storybook render, pastel'}
- The palette: {palette or 'warm pastels'}

ADD to every prompt (this is the enhancement):
- Rich TEXTURE detail (fluffy fur with soft strands, dewy leaves, velvety petals, soft fabric)
- LAYERED DEPTH: foreground + midground + background elements, soft depth-of-field / bokeh
- VOLUMETRIC, CINEMATIC LIGHTING (warm rim light, god rays, soft golden-hour glow, gentle bounce light)
- ATMOSPHERE (floating pollen, drifting sparkles, soft mist, dust motes in light)
- COMPOSITION (clear focal point, rule-of-thirds, appealing camera angle)
- A subtle ACTION the character is doing (so it animates well)
- SDXL quality boosters: "highly detailed, cinematic, beautifully rendered, soft volumetric lighting, depth of field, masterpiece"
- 45-85 words each. ENGLISH only. NEVER add readable text, human faces, or watermarks.

Character bible (use exact descriptors when a character appears):
{chr(10).join('- ' + c for c in char_bible) if char_bible else '(none)'}

Respond with ONLY JSON:
{{"enhanced": [{{"scene_number": 1, "prompt": "<rich enhanced prompt>", "negative_prompt": "<negatives>"}}]}}"""

        user_msg = "Enhance these scene prompts (keep character + style identical, make them cinematic and detailed):\n\n"
        for s in scenes:
            user_msg += f'Scene {s["scene_number"]}: {s.get("prompt","")}\n'

        resp = llm.create_chat_completion(
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_msg}],
            temperature=0.8, max_tokens=self.config.llm.max_tokens,
            response_format={"type": "json_object"}, stream=False,
        )
        raw = resp["choices"][0]["message"].get("content", "") or ""
        try:
            data = json.loads(self._clean_json(raw))
        except json.JSONDecodeError:
            data = self._salvage_truncated(self._clean_json(raw))
            data = {"enhanced": data.get("scenes", [])}
        out = data.get("enhanced") or data.get("scenes") or []
        logger.info(f"[Director] Enhanced {len(out)} scene prompts")
        return out

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
            logger.warning(f"[Director] JSON parse failed ({e}); attempting salvage of complete scenes")
            data = self._salvage_truncated(clean)
            if not data.get("scenes"):
                logger.error(f"[Director] Salvage found no scenes. Raw head: {raw_text[:400]}")
                raise ValueError(f"Director produced invalid JSON: {e}")
            logger.info(f"[Director] Salvaged {len(data['scenes'])} complete scenes from truncated output")

        # Resilience: the model occasionally returns the scenes in a different
        # shape — a bare list, or a single scene object instead of the wrapper.
        if isinstance(data, list):
            data = {"scenes": data}
        elif "scenes" not in data and ("scene_number" in data or "prompt" in data):
            logger.warning("[Director] Model returned a single scene object; wrapping it.")
            data = {"scenes": [data]}

        scene_list = data.get("scenes", [])
        if not scene_list:
            logger.error(f"[Director] No scenes in response. Keys: {list(data.keys())}. Raw head: {raw_text[:300]}")

        scenes = []
        for s in scene_list:
            scenes.append(ScenePlan(
                scene_number=s.get("scene_number", len(scenes) + 1),
                # img2vid is the app default: Z-Image still + LTX animation gives
                # far better character consistency than pure txt2vid.
                scene_type=s.get("scene_type", "img2vid"),
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
            description=data.get("description", ""),
            tags=data.get("tags", []) or [],
            hashtags=data.get("hashtags", []) or [],
        )

    @staticmethod
    def _salvage_truncated(text: str) -> dict:
        """Recover a usable script from truncated/invalid JSON by extracting every
        COMPLETE balanced {...} scene object inside the "scenes" array, plus the
        top-level string fields via regex. Lets a cut-off LLM response still produce
        a video instead of hard-failing."""
        import re
        out = {}
        for key in ("title", "music_style", "music_mood", "thumbnail_prompt", "description"):
            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if m:
                out[key] = m.group(1)
        scenes = []
        si = text.find('"scenes"')
        if si != -1:
            start = text.find('[', si)
            if start != -1:
                depth = 0
                obj_start = None
                for idx in range(start, len(text)):
                    c = text[idx]
                    if c == '{':
                        if depth == 0:
                            obj_start = idx
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0 and obj_start is not None:
                            try:
                                scenes.append(json.loads(text[obj_start:idx + 1]))
                            except Exception:
                                pass
                            obj_start = None
                    elif c == ']' and depth == 0:
                        break
        out["scenes"] = scenes
        return out

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
