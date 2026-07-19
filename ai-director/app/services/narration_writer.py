"""
AI Director — Narration Writer
Two-pass long-form script engine for narration-mode projects (faceless
explainer / tutorial / documentary channels).

Pass 1 (outline): hook, chapter structure, retention curve, word budget.
Pass 2 (draft):   full narration per chapter, split into visual BEATS, each
                  with a visual intent + prompt + sound-design intent.

Output (stored in Project.narration_script as JSON):
{
  "title": "...", "hook": "...", "style": "explainer|tutorial|documentary",
  "chapters": [
    {"title": "...", "summary": "...",
     "beats": [
       {"narration_text": "...",           # what the voice says in this beat
        "visual_type": "broll|still|diagram|code|map|title_card",
        "visual_prompt": "...",            # English, for the image/video model
        "sfx_prompt": "...",               # sound-design intent ("keyboard clatter")
        "mood": "curious|tense|warm|epic|neutral"}
     ]}
  ],
  "seo": {"description": "...", "tags": [...], "hashtags": [...],
          "thumbnail_prompt": "..."}
}
"""
import re
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

WORDS_PER_MINUTE = 150   # measured natural narration pace; drives word budget

VISUAL_TYPES = ("broll", "still", "diagram", "code", "map", "title_card")


OUTLINE_SYSTEM = """You are a top-1% YouTube scriptwriter and retention strategist for faceless narration channels (explainers, tutorials, documentaries).
Design the OUTLINE of a video that keeps viewers watching:
- The first 15 seconds open a curiosity gap or promise a concrete payoff (the HOOK).
- Chapters escalate: each ends with a mini-open-loop into the next.
- The payoff promised in the hook lands near the end (never in the middle).
- No filler, no "in this video I will..." throat-clearing.

Respond ONLY with JSON:
{"title": "<final video title, curiosity + clarity>",
 "hook": "<the exact first 1-2 narration sentences>",
 "style": "explainer|tutorial|documentary",
 "chapters": [{"title": "...", "summary": "<2-3 sentences: what this chapter covers and its open loop>", "target_words": <int>}],
 "thumbnail_prompt": "<one striking visual concept, English>"}"""


DRAFT_SYSTEM = """You are a top-1% YouTube narration writer AND an AI art director who knows exactly what local image/video models on a 16GB GPU render beautifully.
Write the FULL narration for the given chapter, split into visual BEATS.

NARRATION RULES:
- Spoken language: short sentences, concrete nouns, active voice. Write for the EAR, not the page.
- No headings, no stage directions, no "[pause]" markers — pure speakable text.
- Each beat is ONE idea, 1-3 sentences (roughly 4-12 seconds of speech).
- ORIGINAL and transformative writing: analysis, commentary, vivid specifics — never generic filler a template could produce (YouTube demonetizes reused/repetitious content).
- Advertiser-safe: no profanity, graphic violence detail, dangerous instructions, or medical/election misinformation.

VISUAL RULES (per beat):
- visual_type: "broll" (cinematic AI video), "still" (AI still + slow pan), "diagram" (animated diagram/motion graphic), "code" (code/terminal on screen), "map" (animated map), "title_card" (big text card).
- Explainer/tutorial content should LEAN on diagram/code/title_card — AI models cannot render readable text, but the diagram renderer can.
- visual_prompt: ENGLISH. For broll/still: a rich cinematic prompt starting with the shot type ("wide establishing shot, ..."), concrete subject, lighting, atmosphere — NEVER readable text, NEVER real people's faces. For diagram/code/map/title_card: describe exactly what should be on screen (labels, boxes, arrows, code lines, regions).
- sfx_prompt: the sound this beat should carry ("low server-room hum", "distant city traffic", "soft keyboard typing"), or "" for none.
- mood: curious|tense|warm|epic|neutral — drives voice emotion + music.

Respond ONLY with JSON:
{"beats": [{"narration_text": "...", "visual_type": "...", "visual_prompt": "...", "sfx_prompt": "...", "mood": "..."}]}"""


class NarrationWriterService:
    """Two-pass narration script writer on the shared Qwen director brain."""

    def __init__(self, model_manager, config):
        self.manager = model_manager
        self.config = config

    # ── LLM plumbing ───────────────────────────────────────────────────

    def _llm(self):
        from app.services.model_manager import ModelType
        try:
            from app.services.comfyui_client import ComfyUIClient
            ComfyUIClient().free_vram()
            import time as _t; _t.sleep(5)
        except Exception:
            pass
        return self.manager.load(ModelType.LLM).model

    def _chat_json(self, llm, system: str, user: str,
                   temperature: float = 0.7, max_tokens: Optional[int] = None) -> dict:
        resp = llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens or self.config.llm.max_tokens,
            response_format={"type": "json_object"},
            stream=False,
        )
        raw = resp["choices"][0]["message"].get("content", "") or ""
        return json.loads(self._clean_json(raw))

    @staticmethod
    def _clean_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        return text

    # ── Channel context ────────────────────────────────────────────────

    def _channel_context(self, channel_slug: str) -> str:
        try:
            from app.services.director import DirectorService
            profile = DirectorService(self.manager, self.config) \
                .load_channel_profile(channel_slug) or {}
        except Exception:
            profile = {}
        parts = []
        for key, label in [("niche", "Niche"), ("audience", "Audience"),
                           ("tone", "Tone"), ("art_style_phrase", "Visual style"),
                           ("color_palette", "Palette"), ("language", "Language")]:
            if profile.get(key):
                parts.append(f"- {label}: {profile[key]}")
        return ("Channel profile:\n" + "\n".join(parts)) if parts else ""

    # ── Public API ─────────────────────────────────────────────────────

    def write(self, topic: str, duration_sec: int, context: str,
              channel_slug: str, unload_after: bool = False) -> dict:
        """Full two-pass write. Returns the narration script dict.
        Caller controls model unload economics via unload_after."""
        total_words = max(120, int(duration_sec / 60.0 * WORDS_PER_MINUTE))
        n_chapters = max(2, min(8, duration_sec // 75))
        chan_ctx = self._channel_context(channel_slug)

        llm = self._llm()
        try:
            # ── Pass 1: outline ────────────────────────────────────────
            logger.info(f"[NarrationWriter] Outline pass: '{topic}' "
                        f"({duration_sec}s, ~{total_words} words, {n_chapters} chapters)")
            outline = self._chat_json(llm, OUTLINE_SYSTEM, (
                f"Topic: {topic}\n"
                f"Target length: {duration_sec} seconds (~{total_words} words total "
                f"at {WORDS_PER_MINUTE} wpm)\n"
                f"Chapters: exactly {n_chapters} (chapter 1 IS the hook chapter)\n"
                f"{chan_ctx}\n"
                f"Notes from the creator: {context or '(none)'}\n\n"
                f"Distribute target_words across chapters so they sum to ~{total_words}. "
                f"Design the outline now."
            ), temperature=0.8)

            chapters_out = []
            outline_chapters = outline.get("chapters") or []
            if not outline_chapters:
                raise ValueError("Outline pass returned no chapters")

            # ── Pass 2: draft each chapter ─────────────────────────────
            for i, ch in enumerate(outline_chapters):
                logger.info(f"[NarrationWriter] Drafting chapter {i+1}/"
                            f"{len(outline_chapters)}: {ch.get('title')}")
                prev = outline_chapters[i - 1].get("summary", "") if i > 0 else ""
                nxt = outline_chapters[i + 1].get("summary", "") \
                    if i + 1 < len(outline_chapters) else "(this is the final chapter — land the payoff, then a single-sentence outro)"
                draft = self._chat_json(llm, DRAFT_SYSTEM, (
                    f"Video title: {outline.get('title', topic)}\n"
                    f"Video hook (already promised to the viewer): {outline.get('hook','')}\n"
                    f"Style: {outline.get('style', 'explainer')}\n"
                    f"{chan_ctx}\n\n"
                    f"CHAPTER {i+1} of {len(outline_chapters)}: {ch.get('title')}\n"
                    f"Chapter brief: {ch.get('summary')}\n"
                    f"Word budget for this chapter: ~{int(ch.get('target_words') or (total_words // len(outline_chapters)))} words\n"
                    f"Previous chapter covered: {prev or '(none — this chapter OPENS with the hook sentences, verbatim)'}\n"
                    f"Next chapter will cover: {nxt}\n\n"
                    f"Write this chapter's beats now."
                ), temperature=0.75)

                beats = []
                for b in draft.get("beats", []):
                    vt = str(b.get("visual_type", "broll")).lower().strip()
                    if vt not in VISUAL_TYPES:
                        vt = "broll"
                    text = str(b.get("narration_text", "")).strip()
                    if not text:
                        continue
                    beats.append({
                        "narration_text": text,
                        "visual_type": vt,
                        "visual_prompt": str(b.get("visual_prompt", "")).strip(),
                        "sfx_prompt": str(b.get("sfx_prompt", "")).strip(),
                        "mood": str(b.get("mood", "neutral")).strip() or "neutral",
                    })
                if beats:
                    chapters_out.append({
                        "title": str(ch.get("title", f"Chapter {i+1}")),
                        "summary": str(ch.get("summary", "")),
                        "beats": beats,
                    })

            if not chapters_out:
                raise ValueError("Draft pass produced no beats")

            # ── Pass 3 (cheap): SEO block in one call ──────────────────
            all_text = " ".join(
                b["narration_text"] for c in chapters_out for b in c["beats"])
            seo = {}
            try:
                seo = self._chat_json(llm, (
                    "You are a YouTube SEO expert. Given a video's narration, write its metadata. "
                    "Honest metadata only — never promise what the video doesn't deliver. "
                    "Respond ONLY with JSON: "
                    '{"description": "<hooky first line, 2-3 sentence summary, value statement, keywords woven naturally, end with 3-5 hashtags>", '
                    '"tags": ["15-25 search tags viewers actually type"], "hashtags": ["#..."]}'
                ), (
                    f"Title: {outline.get('title', topic)}\n\n"
                    f"Narration (first 2500 chars):\n{all_text[:2500]}"
                ), temperature=0.6, max_tokens=2048)
            except Exception as seo_err:
                logger.warning(f"[NarrationWriter] SEO pass failed (non-fatal): {seo_err}")

            script = {
                "title": str(outline.get("title", topic)),
                "hook": str(outline.get("hook", "")),
                "style": str(outline.get("style", "explainer")),
                "chapters": chapters_out,
                "seo": {
                    "description": str(seo.get("description", "")),
                    "tags": list(seo.get("tags", []))[:25],
                    "hashtags": list(seo.get("hashtags", []))[:5],
                    "thumbnail_prompt": str(outline.get("thumbnail_prompt", "")),
                },
            }
            n_beats = sum(len(c["beats"]) for c in chapters_out)
            n_words = len(all_text.split())
            logger.info(f"[NarrationWriter] Script done: {len(chapters_out)} chapters, "
                        f"{n_beats} beats, {n_words} words (~{n_words / WORDS_PER_MINUTE * 60:.0f}s)")
            return script
        finally:
            if unload_after:
                try:
                    self.manager.unload()
                except Exception:
                    pass
