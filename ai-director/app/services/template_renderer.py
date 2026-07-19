"""
AI Director — Template Renderer
Headless-browser-based motion-graphics renderer for text-heavy visual types
(diagram, code, map, title_card) that AI video models render terribly.

Pipeline: visual_prompt (LLM) → structured data → HTML template → Playwright
screenshot sequence → ffmpeg MP4.

Runs on CPU only (no VRAM), at native 1920×1080 — no upscale needed.
Can run IN PARALLEL with GPU video gen (no contention).
"""
import json
import logging
import subprocess
import time
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent / "knowledge" / "templates"

# visual_type → template filename
TEMPLATE_MAP = {
    "diagram":    "diagram.html",
    "code":       "code.html",
    "map":        "map.html",
    "title_card": "title_card.html",
}

# ── LLM prompt to convert visual_prompt → structured JSON for a template ──

_DATA_PROMPTS = {
    "diagram": """Convert this visual description into a JSON diagram layout.
The diagram has a title, rows of connected nodes (boxes with arrows between them),
and an optional bottom note.

Output ONLY valid JSON:
{"title": "...", "note": "optional bottom note",
 "rows": [
   [{"label": "Step 1", "desc": "short detail", "icon": "emoji or blank", "highlight": false},
    {"label": "Step 2", "desc": "...", "icon": "", "highlight": true}],
   [{"label": "Step 3", "desc": "..."}]
 ]}
Each row is rendered left-to-right with arrows. Multiple rows stack vertically.
Keep labels SHORT (2-4 words). Max 3 rows, max 4 nodes per row.
Use *text* around a word in the title to accent it.

Visual description: """,

    "code": """Convert this visual description into a code/terminal display.
Output the code that should be shown on screen, with syntax highlighting tokens.

Output ONLY valid JSON:
{"filename": "example.py", "label": "optional bottom caption",
 "lines": [
   {"num": 1, "html": "<span class=\\"kw\\">def</span> <span class=\\"fn\\">hello</span>():", "active": false},
   {"num": 2, "html": "    <span class=\\"kw\\">return</span> <span class=\\"str\\">\\"world\\"</span>", "active": true}
 ]}
CSS classes: kw (keyword/purple), fn (function/blue), str (string/green),
num (number/orange), cmt (comment/grey), op (operator/cyan), type (type/teal), var (variable/white).
Max 12 lines. Mark the most important line with "active": true.

Visual description: """,

    "title_card": """Convert this visual description into a title card.

Output ONLY valid JSON:
{"label": "CHAPTER 1", "title": "The key *concept*", "subtitle": "A one-line explanation"}
Use *text* around keywords in the title to apply an accent gradient.
Keep the title under 8 words. Label is a small uppercase tag above the title.

Visual description: """,

    "map": """Convert this visual description into a map with location markers.

Output ONLY valid JSON:
{"title": "Map Title", "description": "What this map shows",
 "markers": [
   {"x": 30, "y": 40, "label": "Location A", "desc": "detail", "secondary": false},
   {"x": 65, "y": 55, "label": "Location B", "desc": "", "secondary": true}
 ]}
x and y are percentages (0-100) positioning markers on the viewport.
Space markers apart so labels don't overlap. Max 5 markers.

Visual description: """,
}


class TemplateRenderer:
    """Render text-heavy visual types to MP4 via headless browser + ffmpeg."""

    def __init__(self, model_manager, config):
        self.manager = model_manager
        self.config = config
        self._playwright = None
        self._browser = None

    # ── LLM: visual_prompt → template data ─────────────────────────────

    def _prompt_to_data(self, visual_type: str, visual_prompt: str) -> dict:
        """Use the LLM to convert a free-text visual_prompt into the
        structured JSON that the template expects."""
        data_prompt = _DATA_PROMPTS.get(visual_type)
        if not data_prompt:
            return {"title": visual_prompt}

        try:
            from app.services.model_manager import ModelType
            llm = self.manager.load(ModelType.LLM).model
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system",
                     "content": "You convert visual descriptions into structured JSON for "
                                "HTML template rendering. Output ONLY valid JSON, nothing else."},
                    {"role": "user", "content": data_prompt + visual_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"},
                stream=False,
            )
            raw = resp["choices"][0]["message"].get("content", "") or ""
            raw = raw.strip()
            if raw.startswith("```"):
                import re
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                raw = raw[start:end + 1]
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"[TemplateRenderer] LLM data gen failed: {e}")
            # fallback: use the visual prompt as a title card
            return {"title": visual_prompt, "label": visual_type.upper()}

    # ── Channel style injection ────────────────────────────────────────

    def _channel_css(self, profile: dict) -> str:
        """Build CSS custom property overrides from the channel profile."""
        palette = profile.get("color_palette", {})
        css = ":root {\n"
        if palette.get("bg"):     css += f"  --bg: {palette['bg']};\n"
        if palette.get("accent"): css += f"  --accent: {palette['accent']};\n"
        if palette.get("text"):   css += f"  --text: {palette['text']};\n"
        css += "}\n"
        return css

    # ── Playwright rendering ───────────────────────────────────────────

    def _ensure_browser(self):
        """Lazy-start Playwright + Chromium."""
        if self._browser is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=True,
                args=["--disable-gpu", "--no-sandbox",
                      "--disable-dev-shm-usage"],
            )
            logger.info("[TemplateRenderer] Playwright Chromium launched")
        except ImportError:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && "
                "playwright install chromium")

    def close(self):
        """Release browser resources."""
        try:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._playwright = None

    def render_clip(
        self,
        visual_type: str,
        visual_prompt: str,
        output_path: str,
        duration: float = 5.0,
        fps: int = 24,
        width: int = 1920,
        height: int = 1080,
        profile: Optional[dict] = None,
    ) -> str:
        """Render one template clip to MP4.

        1. LLM converts visual_prompt → structured data
        2. Open template in headless Chromium with injected data
        3. Capture PNG screenshots at `fps` for `duration` seconds
        4. ffmpeg encodes the sequence to H.264 MP4

        Returns the output MP4 path.
        """
        t0 = time.time()
        template_file = TEMPLATE_MAP.get(visual_type)
        if not template_file:
            raise ValueError(f"No template for visual_type '{visual_type}'")

        template_path = TEMPLATES_DIR / template_file
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        # 1. Get structured data
        data = self._prompt_to_data(visual_type, visual_prompt)
        logger.info(f"[TemplateRenderer] {visual_type}: "
                    f"{json.dumps(data)[:120]}…")

        # 2. Open in headless browser
        self._ensure_browser()
        page = self._browser.new_page(
            viewport={"width": width, "height": height})

        # Inject channel styles + data BEFORE navigating
        channel_css = self._channel_css(profile or {})
        data_script = f"window.__DATA__ = {json.dumps(data)};"

        # Navigate to the template (file:// URL)
        page.goto(f"file:///{template_path.as_posix()}")
        page.add_style_tag(content=channel_css)
        page.evaluate(data_script)
        page.evaluate("if (typeof render === 'function') render(window.__DATA__)")

        # 3. Capture frames
        frames_dir = Path(output_path).parent / "_template_frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True)

        n_frames = max(1, int(duration * fps))
        interval_ms = 1000.0 / fps

        for i in range(n_frames):
            # Advance CSS animations by stepping time
            # (Playwright doesn't auto-advance animations for screenshots,
            #  so we wait real-time for the animation to play)
            if i == 0:
                # Give the first frame 100ms for fonts to load
                page.wait_for_timeout(100)
            frame_path = frames_dir / f"frame_{i:05d}.png"
            page.screenshot(path=str(frame_path), type="png")
            if i < n_frames - 1:
                page.wait_for_timeout(int(interval_ms))

        page.close()

        # 4. Encode to MP4
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = str(self.config.paths.ffmpeg_bin)
        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-t", f"{duration:.3f}",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg template encode failed: "
                               f"{r.stderr[-300:]}")

        # Cleanup frames
        try:
            shutil.rmtree(frames_dir)
        except Exception:
            pass

        elapsed = time.time() - t0
        logger.info(f"[TemplateRenderer] {visual_type} → {output_path} "
                    f"({n_frames} frames, {duration:.1f}s) in {elapsed:.1f}s")
        return output_path
