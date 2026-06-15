"""
AI Engine for VideoMaker
========================
Supports Claude (Anthropic) and Gemini (Google) for:
- Video prompt generation based on channel config + history
- SEO metadata generation (titles, description, tags)
- Video topic ideation
"""

import json, re, os
from pathlib import Path

try:
    import anthropic
    HAS_CLAUDE = True
except ImportError:
    HAS_CLAUDE = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

CONTEXT_FILE = Path(__file__).parent / "channel_context.json"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

PROMPT_SYSTEM = """You are an elite YouTube content strategist and AI video prompt engineer who has grown multiple channels to millions of subscribers.

Your dual expertise:
1. YouTube algorithm mastery — you understand watch time, CTR, retention curves, autoplay behavior, and YouTube Kids recommendations
2. AI video prompt engineering — you know exactly what AI video models (LTX, Stable Diffusion) render beautifully and what they struggle with

Rules for AI video clip prompts:
- Each clip is one continuous scene (no cuts within a clip)
- Describe: subject/action, environment/setting, lighting direction and quality, camera angle and movement, color palette, atmosphere/mood
- Use specific visual language: "slow cinematic pan", "warm golden backlight", "shallow depth of field", "soft volumetric fog"
- NEVER include: human faces or realistic people (AI artifacts), text/words/numbers in frames, rapid complex motion, multiple simultaneous actions
- ALWAYS ensure: visual continuity between clips (same characters, consistent style/palette), gradual scene progression (don't teleport), emotional arc across the full video
- Keep each prompt under 80 words for best generation quality
- For kids content: bright saturated colors, gentle movements, magical/whimsical elements, nothing scary"""

PROMPT_GENERATE = """Channel: {channel_name}
Niche: {niche}
Target Audience: {target_audience}
Visual Style Guide: {style_prompt}
Language: {language}

Previously covered topics (DO NOT repeat these):
{history}

VIDEO ASSIGNMENT:
Create {num_clips} sequential clip prompts for a {total_duration}-second video about: "{topic}"
Each clip = {seconds_per_clip} seconds of footage.

Build a clear visual narrative arc:
- Clips 1-2: Establishing shots, introduce the scene/world
- Middle clips: Explore, discover, build wonder
- Final 1-2 clips: Climax moment, satisfying conclusion

Return ONLY valid JSON (no markdown, no explanation):
{{"topic": "final video topic title", "prompts": ["clip 1 prompt", "clip 2 prompt", ...]}}"""

SEO_SYSTEM = """You are the #1 YouTube SEO expert in the world. You've personally optimized metadata for channels generating 100M+ monthly views.

Your deep knowledge:
- Title psychology: curiosity gaps, power words, emotional triggers, pattern interrupts
- YouTube search ranking: keyword placement, description density, tag strategy
- YouTube Kids algorithm: safe content signals, educational tags, age-appropriate language
- Suggested video optimization: high CTR + watch time = algorithmic push
- Hashtag discovery: trending hashtags for niche visibility

You write metadata that feels authentic — never spammy, never misleading — but is engineered to maximize impressions, CTR, and watch time."""

SEO_GENERATE = """Channel: {channel_name}
Niche: {niche}
Target Audience: {target_audience}
Channel Keywords: {keywords}
Made for Kids: {for_kids}

Video clip prompts (this is what the video shows):
{prompts_summary}

Generate YouTube-optimized metadata. Return ONLY valid JSON:
{{
  "titles": [
    "title 1 — emotional hook angle",
    "title 2 — curiosity gap angle",
    "title 3 — benefit/value angle",
    "title 4 — question angle",
    "title 5 — list/compilation angle"
  ],
  "description": "Full YouTube description here (2000+ characters). Start with a compelling hook + primary keyword in first 2 lines (shown in search). Include scene timestamps. Add subscribe CTA. End with hashtags.",
  "tags": ["tag1", "tag2", "...up to 30 tags mixing broad head terms and specific long-tail keywords"],
  "hashtags": ["#relevant1", "#relevant2", "#relevant3"]
}}

TITLE RULES:
- Under 60 characters each
- 1-2 relevant emojis per title
- Include primary keyword naturally
- Use power words: magical, beautiful, enchanting, amazing, ultimate, incredible
- Each title uses a DIFFERENT psychological hook

DESCRIPTION RULES:
- First 2 lines: hook + primary keyword (this shows in search results — make it count)
- Timestamps for major scenes (00:00, 00:30, 01:00, etc.)
- Naturally weave in 5-8 keywords
- Include: "Subscribe for more [niche] videos!"
- 2000+ characters minimum (longer descriptions rank better)
- End with 3 hashtags

TAG RULES:
- Exactly 30 tags
- Mix: 5 broad terms, 10 medium terms, 15 long-tail specific phrases
- Include channel name, competitor-adjacent terms, trending variations
- Include both singular and plural forms of key terms"""

IDEA_SYSTEM = """You are a YouTube trend analyst who identifies viral content opportunities before they peak.

Your analysis framework:
- Seasonal relevance (holidays, seasons, events)
- Evergreen search volume (topics people always search for)
- Content gaps (what competitors haven't covered)
- Algorithm-friendly formats (compilations, series, satisfying/calming content)
- Audience retention patterns (what keeps people watching)
- YouTube Kids autoplay behavior (what gets recommended after similar content)"""

IDEA_GENERATE = """Channel: {channel_name}
Niche: {niche}
Target Audience: {target_audience}
Visual Style: {style_prompt}
Channel Keywords: {keywords}

Previously published topics:
{history}

Generate 5 fresh video ideas that:
1. Have NOT been covered by this channel yet
2. Have high search volume or trending potential
3. Work perfectly with AI video generation (no faces, simple motion)
4. Match the channel's visual style and audience
5. Would perform well in YouTube recommendations/autoplay

Return ONLY valid JSON:
{{"ideas": [
  {{"topic": "video topic", "why": "why this will get views", "hook": "the viewer hook that drives clicks"}},
  ...5 ideas total
]}}"""


# ---------------------------------------------------------------------------
# Context persistence
# ---------------------------------------------------------------------------

def load_context():
    if CONTEXT_FILE.exists():
        return json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))
    return {"channels": {}}


def save_context(ctx):
    CONTEXT_FILE.write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")


def update_channel_context(channel_id, topic):
    ctx = load_context()
    cid = str(channel_id)
    if cid not in ctx["channels"]:
        ctx["channels"][cid] = {"themes_used": [], "recent_topics": []}
    ch = ctx["channels"][cid]
    if topic and topic not in ch["themes_used"]:
        ch["themes_used"].append(topic)
    if topic and topic not in ch["recent_topics"]:
        ch["recent_topics"].append(topic)
        ch["recent_topics"] = ch["recent_topics"][-50:]
    save_context(ctx)


# ---------------------------------------------------------------------------
# AI Engine
# ---------------------------------------------------------------------------

class AIEngine:
    def __init__(self, provider="claude", claude_key=None, gemini_key=None, claude_model=None, gemini_model=None):
        self.provider = provider
        self.claude_key = claude_key
        self.gemini_key = gemini_key
        self.claude_model = claude_model or "claude-sonnet-4-6"
        self.gemini_model = gemini_model or "gemini-2.5-flash"
        self._claude_client = None
        self._gemini_client = None

    def _get_claude(self):
        if not HAS_CLAUDE:
            raise ImportError("pip install anthropic")
        if not self.claude_key:
            raise ValueError("Claude API key not set. Go to Settings to add it.")
        if not self._claude_client:
            self._claude_client = anthropic.Anthropic(api_key=self.claude_key)
        return self._claude_client

    def _get_gemini(self):
        if not HAS_GEMINI:
            raise ImportError("pip install google-generativeai")
        if not self.gemini_key:
            raise ValueError("Gemini API key not set. Go to Settings to add it.")
        if not self._gemini_client:
            genai.configure(api_key=self.gemini_key)
            self._gemini_client = genai.GenerativeModel(self.gemini_model)
        return self._gemini_client

    def _call(self, system, user_msg):
        if self.provider == "claude":
            client = self._get_claude()
            resp = client.messages.create(
                model=self.claude_model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            return resp.content[0].text
        else:
            client = self._get_gemini()
            resp = client.generate_content(
                f"{system}\n\n---\n\n{user_msg}",
                generation_config=genai.GenerationConfig(max_output_tokens=4096, temperature=0.8),
            )
            return resp.text

    def _parse_json(self, text):
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from AI response: {text[:300]}")

    # ----- public methods -----

    def generate_prompts(self, channel, history_titles, topic, num_clips=6, seconds_per_clip=5):
        history = "\n".join(f"- {t}" for t in history_titles[-30:]) if history_titles else "(none yet — this is the first video)"
        user = PROMPT_GENERATE.format(
            channel_name=channel.get("name", ""),
            niche=channel.get("niche", ""),
            target_audience=channel.get("target_audience", ""),
            style_prompt=channel.get("style_prompt", ""),
            language=channel.get("language", "English"),
            history=history,
            num_clips=num_clips,
            topic=topic,
            seconds_per_clip=seconds_per_clip,
            total_duration=num_clips * seconds_per_clip,
        )
        result = self._parse_json(self._call(PROMPT_SYSTEM, user))
        update_channel_context(channel.get("id", 0), result.get("topic", topic))
        return result

    def generate_seo(self, channel, prompts_text):
        user = SEO_GENERATE.format(
            channel_name=channel.get("name", ""),
            niche=channel.get("niche", ""),
            target_audience=channel.get("target_audience", ""),
            keywords=channel.get("keywords", ""),
            for_kids="YES — optimize for YouTube Kids" if channel.get("for_kids") else "NO",
            prompts_summary=prompts_text[:3000],
        )
        return self._parse_json(self._call(SEO_SYSTEM, user))

    def suggest_topics(self, channel, history_titles):
        history = "\n".join(f"- {t}" for t in history_titles[-30:]) if history_titles else "(none yet)"
        user = IDEA_GENERATE.format(
            channel_name=channel.get("name", ""),
            niche=channel.get("niche", ""),
            target_audience=channel.get("target_audience", ""),
            style_prompt=channel.get("style_prompt", ""),
            keywords=channel.get("keywords", ""),
            history=history,
        )
        return self._parse_json(self._call(IDEA_SYSTEM, user))


def get_available_providers():
    p = []
    if HAS_CLAUDE:
        p.append("claude")
    if HAS_GEMINI:
        p.append("gemini")
    return p
