"""
AI Director — Universal YouTube Safety Gate
Runs on EVERY project (song AND narration) BEFORE any GPU generation time.

Two layers:
  1. Rule layer (deterministic, instant, no model):
     lexicon scans, quote-length limits, kids-content strictness, CTA spam.
  2. LLM critic layer (Qwen director brain, JSON verdict):
     advertiser-friendly guidelines, reused/repetitious-content originality,
     misinformation, made-for-kids/COPPA correctness, misleading metadata.

Result is persisted as a SafetyReport row; the pipeline refuses to spend GPU
time until the latest verdict is pass (or a recorded human override).
"""
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ── Rule-layer lexicons ─────────────────────────────────────────────────────
# Advertiser-unfriendly / policy-risk terms. Word-boundary matched, case-
# insensitive. Grouped by category so the report explains WHY.

PROFANITY = [
    "fuck", "fucking", "shit", "bitch", "asshole", "bastard", "dick",
    "cunt", "motherfucker", "nigger", "faggot", "slut", "whore",
]

VIOLENCE_GRAPHIC = [
    "beheading", "decapitat", "dismember", "torture", "massacre",
    "mutilat", "gore", "bloodbath", "execution video", "mass shooting",
    "school shooting", "killed on camera",
]

DANGEROUS_ACTS = [
    "how to make a bomb", "build a gun", "3d printed gun", "make explosives",
    "poison someone", "hard drug", "how to overdose", "self-harm tutorial",
    "suicide method", "choking challenge", "blackout challenge",
    "tide pod", "drink bleach",
]

MEDICAL_MISINFO = [
    "vaccines cause autism", "cure cancer with", "cures cancer",
    "miracle cure", "doctors don't want you to know", "big pharma is hiding",
    "covid is a hoax", "5g causes", "drink your urine",
]

MISLEADING_CLICKBAIT = [
    "you won't believe", "gone wrong gone sexual", "(not clickbait)",
    "free robux", "free v-bucks", "get rich quick", "guaranteed profit",
    "100% guaranteed returns",
]

KIDS_UNSAFE = [
    # extra strictness when channel.made_for_kids — things fine for adults
    # but that trip made-for-kids review
    "kill", "gun", "knife", "blood", "die", "dead", "death", "scary",
    "horror", "demon", "devil", "drug", "beer", "cigarette", "kidnap",
    "gambling", "casino",
]

COPYRIGHT_MARKERS = [
    "lyrics by", "official lyrics", "©", "(c) 20", "all rights reserved",
    "originally performed by", "cover of the song",
]


@dataclass
class SafetyIssue:
    severity: str      # "block" | "high" | "medium" | "low"
    category: str      # profanity | violence | dangerous | medical | kids | copyright | clickbait | originality | disclosure
    where: str         # which field: narration/lyrics/prompt scene N/title/description/tags
    detail: str
    suggestion: str = ""


@dataclass
class GateResult:
    verdict: str                      # pass | revise | block
    issues: list = field(default_factory=list)          # list[SafetyIssue]
    checked_fields: dict = field(default_factory=dict)  # field -> chars scanned
    auto_revisions: list = field(default_factory=list)  # [{where, before, after}]
    llm_used: bool = False

    def to_dict(self):
        d = asdict(self)
        return d


class SafetyGateService:
    """Universal pre-GPU YouTube policy gate."""

    def __init__(self, model_manager, config):
        self.manager = model_manager
        self.config = config

    # ── Content collection ─────────────────────────────────────────────

    def _collect(self, project, scenes) -> dict:
        """Gather every user-facing text surface of the project, by field name."""
        fields = {}
        if project.title:
            fields["title"] = project.title
        if project.context:
            fields["context"] = project.context

        ptype = getattr(project, "project_type", None) or "song"
        if ptype == "narration":
            script = getattr(project, "narration_script", None)
            if script:
                try:
                    data = json.loads(script)
                    narr_parts, i = [], 0
                    for ch in data.get("chapters", []):
                        for b in ch.get("beats", []):
                            i += 1
                            if b.get("narration_text"):
                                narr_parts.append(f"[beat {i}] {b['narration_text']}")
                    fields["narration"] = "\n".join(narr_parts)
                    seo = data.get("seo", {})
                    if seo.get("description"):
                        fields["description"] = seo["description"]
                    if seo.get("tags"):
                        fields["tags"] = ", ".join(seo["tags"])
                except Exception:
                    fields["narration"] = script
        else:
            if project.lyrics:
                fields["lyrics"] = project.lyrics
            if project.music_style:
                fields["music_style"] = project.music_style
            if project.script_raw:
                try:
                    sr = json.loads(project.script_raw)
                    if sr.get("description"):
                        fields["description"] = sr["description"]
                    if sr.get("tags"):
                        fields["tags"] = ", ".join(sr["tags"])
                    if sr.get("thumbnail_prompt"):
                        fields["thumbnail_prompt"] = sr["thumbnail_prompt"]
                except Exception:
                    pass

        for s in scenes:
            key = f"scene {s.scene_number} prompt"
            fields[key] = s.prompt or ""
            if s.narration_text:
                fields[f"scene {s.scene_number} narration"] = s.narration_text
            notes = s.director_notes or {}
            if notes.get("lyric_text"):
                fields[f"scene {s.scene_number} lyric"] = notes["lyric_text"]
        return fields

    # ── Layer 1: rules ─────────────────────────────────────────────────

    def _scan_lexicon(self, text: str, terms: list[str], whole_word: bool = True):
        hits = []
        low = text.lower()
        for t in terms:
            if whole_word and " " not in t and not t.endswith(("at", "ing")):
                if re.search(rf"\b{re.escape(t)}\b", low):
                    hits.append(t)
            elif t in low:
                hits.append(t)
        return hits

    def run_rules(self, project, fields: dict) -> list[SafetyIssue]:
        issues: list[SafetyIssue] = []
        made_for_kids = bool(project.channel and project.channel.made_for_kids)

        for where, text in fields.items():
            if not text:
                continue

            for hit in self._scan_lexicon(text, PROFANITY):
                issues.append(SafetyIssue(
                    "high", "profanity", where,
                    f"Profanity '{hit}' — limits ads, trips age-restriction.",
                    "Remove or replace the word."))
            for hit in self._scan_lexicon(text, VIOLENCE_GRAPHIC, whole_word=False):
                issues.append(SafetyIssue(
                    "block", "violence", where,
                    f"Graphic-violence phrase '{hit}' — advertiser-unfriendly / possible strike.",
                    "Describe events without graphic detail."))
            for hit in self._scan_lexicon(text, DANGEROUS_ACTS, whole_word=False):
                issues.append(SafetyIssue(
                    "block", "dangerous", where,
                    f"Dangerous-acts phrase '{hit}' — harmful/dangerous-content policy.",
                    "Remove the instructional/dangerous framing entirely."))
            for hit in self._scan_lexicon(text, MEDICAL_MISINFO, whole_word=False):
                issues.append(SafetyIssue(
                    "block", "medical", where,
                    f"Medical-misinformation phrase '{hit}'.",
                    "State only mainstream-consensus health information."))
            for hit in self._scan_lexicon(text, MISLEADING_CLICKBAIT, whole_word=False):
                issues.append(SafetyIssue(
                    "medium", "clickbait", where,
                    f"Clickbait/scam phrase '{hit}' — misleading-metadata risk.",
                    "Rewrite as an honest, specific hook."))
            for hit in self._scan_lexicon(text, COPYRIGHT_MARKERS, whole_word=False):
                issues.append(SafetyIssue(
                    "high", "copyright", where,
                    f"Copyright marker '{hit}' — suggests third-party lyrics/content.",
                    "All lyrics and text must be original to this project."))

            if made_for_kids and where != "context":
                for hit in self._scan_lexicon(text, KIDS_UNSAFE):
                    issues.append(SafetyIssue(
                        "high", "kids", where,
                        f"'{hit}' on a made-for-kids channel — fails kids review.",
                        "Keep made-for-kids content gentle: no weapons/death/scary/substances."))

            # Long verbatim quotes (copyright / reused-content risk)
            for m in re.finditer(r'["“]([^"”]{90,})["”]', text):
                issues.append(SafetyIssue(
                    "medium", "copyright", where,
                    f"Verbatim quote of {len(m.group(1))} chars — keep quotes under ~90 chars.",
                    "Paraphrase in original words."))

        return issues

    # ── Layer 2: LLM critic ────────────────────────────────────────────

    LLM_SYSTEM = """You are a strict YouTube Trust & Safety + monetization reviewer.
You review the FULL text surface of a video before production. Judge it against:
1. Advertiser-friendly guidelines: violence, shocking content, drugs, dangerous acts, hateful content, sexual content, sensitive events, profanity.
2. Harmful/dangerous content policy and medical/election misinformation policies.
3. REUSED/REPETITIOUS CONTENT policy (the #1 faceless-channel demonetization cause): the script must be ORIGINAL and TRANSFORMATIVE — original narration with commentary/analysis/structure, not a rehash a template could produce, not content scraped from elsewhere.
4. Made-for-kids (COPPA) correctness when flagged: gentle content, no scary/violent/adult themes.
5. Metadata honesty: title/description/tags must not promise what the content doesn't deliver.

Be pragmatic: normal storytelling conflict, mild peril in fiction, historical facts, and tech content are FINE. Flag only real policy risk.

Respond ONLY with JSON:
{"verdict": "pass|revise|block",
 "issues": [{"severity": "block|high|medium|low", "category": "...", "where": "<field name>", "detail": "...", "suggestion": "..."}],
 "rewrites": [{"where": "<field name>", "before": "<exact offending sentence>", "after": "<safe replacement sentence>"}]}
verdict=pass when there are no block/high issues. rewrites: give a drop-in replacement for every high/block sentence you can fix in place."""

    def run_llm_critic(self, project, fields: dict) -> Optional[dict]:
        """Run the Qwen policy critic. Returns parsed dict or None on failure.
        Caller is responsible for model load/unload economics."""
        try:
            from app.services.model_manager import ModelType
            try:
                from app.services.comfyui_client import ComfyUIClient
                ComfyUIClient().free_vram()
                import time as _t; _t.sleep(3)
            except Exception:
                pass
            llm = self.manager.load(ModelType.LLM).model

            made_for_kids = bool(project.channel and project.channel.made_for_kids)
            body = "\n\n".join(
                f"### {name}\n{text[:6000]}" for name, text in fields.items() if text
            )
            user_msg = (
                f"Project type: {getattr(project, 'project_type', 'song')}\n"
                f"Made for kids: {made_for_kids}\n"
                f"Channel: {project.channel.name if project.channel else '?'}\n\n"
                f"Review this content:\n\n{body}"
            )

            resp = llm.create_chat_completion(
                messages=[{"role": "system", "content": self.LLM_SYSTEM},
                          {"role": "user", "content": user_msg}],
                temperature=0.2,           # reviewer, not writer — deterministic
                max_tokens=4096,
                response_format={"type": "json_object"},
                stream=False,
            )
            raw = resp["choices"][0]["message"].get("content", "") or ""
            data = json.loads(self._clean_json(raw))
            if not isinstance(data, dict) or "verdict" not in data:
                return None
            return data
        except Exception as e:
            logger.warning(f"[Safety] LLM critic failed (rules-only verdict stands): {e}")
            return None

    @staticmethod
    def _clean_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        return text

    # ── Auto-revise ────────────────────────────────────────────────────

    def _apply_rewrites(self, session, project, scenes, rewrites: list[dict]) -> list[dict]:
        """Apply drop-in sentence replacements from the LLM critic to the
        project's own text fields. Lyrics are NEVER silently rewritten (the
        song may already be generated from them) — those stay as issues for
        the human. Returns the revisions actually applied."""
        applied = []
        scene_by_num = {}
        for s in scenes:
            scene_by_num[f"scene {s.scene_number} prompt"] = (s, "prompt")
            scene_by_num[f"scene {s.scene_number} narration"] = (s, "narration_text")

        narration_data = None
        if getattr(project, "narration_script", None):
            try:
                narration_data = json.loads(project.narration_script)
            except Exception:
                narration_data = None

        for rw in rewrites or []:
            where = str(rw.get("where", ""))
            before = str(rw.get("before", "")).strip()
            after = str(rw.get("after", "")).strip()
            if not before or not after or before == after:
                continue

            if where in scene_by_num:
                scene, attr = scene_by_num[where]
                cur = getattr(scene, attr) or ""
                if before in cur:
                    setattr(scene, attr, cur.replace(before, after))
                    applied.append({"where": where, "before": before, "after": after})
            elif where.startswith("narration") and narration_data:
                changed = False
                for ch in narration_data.get("chapters", []):
                    for b in ch.get("beats", []):
                        txt = b.get("narration_text", "")
                        if before in txt:
                            b["narration_text"] = txt.replace(before, after)
                            changed = True
                if changed:
                    applied.append({"where": "narration", "before": before, "after": after})
            elif where == "description":
                # description lives inside script_raw / narration seo blocks
                if narration_data and narration_data.get("seo", {}).get("description", ""):
                    desc = narration_data["seo"]["description"]
                    if before in desc:
                        narration_data["seo"]["description"] = desc.replace(before, after)
                        applied.append({"where": where, "before": before, "after": after})
                elif project.script_raw:
                    try:
                        sr = json.loads(project.script_raw)
                        if before in sr.get("description", ""):
                            sr["description"] = sr["description"].replace(before, after)
                            project.script_raw = json.dumps(sr)
                            applied.append({"where": where, "before": before, "after": after})
                    except Exception:
                        pass
            # lyrics deliberately not auto-rewritten

        if narration_data is not None and applied:
            project.narration_script = json.dumps(narration_data, ensure_ascii=False)
        return applied

    # ── Orchestration ──────────────────────────────────────────────────

    def _scan_ip_denylist(self, fields: dict, denylist: Optional[list]) -> list:
        """Block-severity issue per archetype IP deny-list hit (word-boundary,
        case-insensitive). Copyrighted characters/brands = monetization risk."""
        if not denylist:
            return []
        issues = []
        for term in denylist:
            term = (term or "").strip()
            if not term:
                continue
            pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            for where, text in fields.items():
                if text and pat.search(text):
                    issues.append(SafetyIssue(
                        severity="block", category="copyright",
                        where=where,
                        detail=f"Copyrighted IP '{term}' found in {where}.",
                        suggestion=f"Remove '{term}' and use an original character/name."))
                    break  # one issue per term is enough
        return issues

    def run_gate(self, project_id: str, use_llm: bool = True,
                 auto_revise: bool = True, unload_after: bool = True,
                 ip_denylist: Optional[list] = None) -> "GateResult":
        """Full gate: collect → rules → (LLM critic) → auto-revise → re-run rules
        → persist SafetyReport. Returns the GateResult.

        `ip_denylist`: extra copyrighted-IP terms (from the project's archetype,
        e.g. "bheem") that force a BLOCK verdict when found — monetization risk."""
        from app.database import (get_session, Project, Scene, SafetyReport,
                                  SafetyVerdict)
        session = get_session()
        try:
            project = session.query(Project).get(project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            scenes = session.query(Scene).filter(
                Scene.project_id == project_id
            ).order_by(Scene.scene_number).all()

            fields = self._collect(project, scenes)
            rule_issues = self.run_rules(project, fields)
            rule_issues += self._scan_ip_denylist(fields, ip_denylist)
            llm_issues: list[SafetyIssue] = []
            llm_used = False
            revisions: list[dict] = []

            if use_llm:
                critic = self.run_llm_critic(project, fields)
                if critic:
                    llm_used = True
                    llm_issues = [SafetyIssue(
                        severity=str(it.get("severity", "medium")),
                        category=str(it.get("category", "policy")),
                        where=str(it.get("where", "?")),
                        detail=str(it.get("detail", "")),
                        suggestion=str(it.get("suggestion", "")),
                    ) for it in critic.get("issues", [])]
                    if auto_revise and critic.get("rewrites"):
                        revisions = self._apply_rewrites(
                            session, project, scenes, critic["rewrites"])
                        if revisions:
                            # revised text may clear hits — rescan rules and
                            # drop LLM issues whose offending sentence was fixed
                            session.flush()
                            fields = self._collect(project, scenes)
                            rule_issues = self.run_rules(project, fields)
                            fixed_wheres = {r["where"] for r in revisions}
                            fixed_texts = {r["before"] for r in revisions}
                            llm_issues = [
                                i for i in llm_issues
                                if i.where not in fixed_wheres
                                and not any(ft in i.detail for ft in fixed_texts if ft)
                            ]
                if unload_after:
                    try:
                        self.manager.unload()
                    except Exception:
                        pass

            issues = rule_issues + llm_issues

            severities = {i.severity for i in issues}
            if "block" in severities:
                verdict = "block"
            elif "high" in severities:
                verdict = "revise"
            else:
                verdict = "pass"

            result = GateResult(
                verdict=verdict,
                issues=[asdict(i) for i in issues],
                checked_fields={k: len(v or "") for k, v in fields.items()},
                auto_revisions=revisions,
                llm_used=llm_used,
            )

            report = SafetyReport(
                project_id=project_id,
                verdict=SafetyVerdict(verdict),
                issues=result.issues,
                checked_fields=result.checked_fields,
                auto_revisions=result.auto_revisions,
                llm_used=llm_used,
            )
            session.add(report)
            session.commit()
            logger.info(f"[Safety] Gate verdict for {project_id}: {verdict} "
                        f"({len(issues)} issues, {len(revisions)} auto-fixes, "
                        f"llm={llm_used})")
            return result
        finally:
            session.close()


def latest_verdict(session, project_id: str) -> Optional[str]:
    """Latest safety verdict for a project, or None if never checked."""
    from app.database import SafetyReport
    row = (session.query(SafetyReport)
           .filter(SafetyReport.project_id == project_id)
           .order_by(SafetyReport.created_at.desc())
           .first())
    return row.verdict.value if row else None
