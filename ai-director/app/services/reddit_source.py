"""
AI Director — Reddit source (reddit_story archetype, source: scrape)

Fetches top threads from a subreddit via Reddit's PUBLIC read-only JSON endpoint
(no login, no OAuth, no credentials — we never automate an account). The pulled
text is handed to the narration writer as context; the safety gate still scans
the formatted narration + SEO before any GPU spend.

Respect Reddit: identify with a descriptive User-Agent, keep request volume low,
and use only public listings. This module reads; it never posts, votes, or logs in.
"""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

USER_AGENT = "AiDirector/1.0 (local content pipeline; read-only public JSON)"

# NSFW / over_18 threads are dropped — a made-for-YouTube pipeline must not
# narrate adult content, and it protects the safety gate downstream.


@dataclass
class RedditThread:
    title: str
    selftext: str
    score: int
    num_comments: int
    permalink: str
    over_18: bool

    @property
    def word_count(self) -> int:
        return len((self.title + " " + self.selftext).split())


def _listing_url(subreddit: str, time_filter: str, limit: int) -> str:
    sub = subreddit.strip().lstrip("r/").strip("/")
    tf = time_filter if time_filter in ("hour", "day", "week", "month", "year", "all") else "week"
    return f"https://www.reddit.com/r/{sub}/top.json?t={tf}&limit={int(limit)}"


def fetch_top(subreddit: str, limit: int = 10, time_filter: str = "week",
              min_words: int = 40, timeout: float = 15.0) -> list[RedditThread]:
    """Return the top self-post threads for a subreddit, SFW only, longest first.

    Network-dependent; raises RedditFetchError on transport/parse failure so the
    caller can fall back to a manual context or fail the run cleanly."""
    url = _listing_url(subreddit, time_filter, limit)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, TimeoutError) as e:
        raise RedditFetchError(f"Reddit fetch failed for r/{subreddit}: {e}") from e

    return parse_listing(payload, min_words=min_words)


def parse_listing(payload: dict, min_words: int = 40) -> list[RedditThread]:
    """Parse a Reddit listing JSON dict into SFW self-post threads (longest
    first). Split out so it can be unit-tested with a fixture, no network."""
    threads: list[RedditThread] = []
    for child in (payload.get("data", {}) or {}).get("children", []) or []:
        d = child.get("data", {}) or {}
        if d.get("over_18") or d.get("stickied"):
            continue
        selftext = (d.get("selftext") or "").strip()
        if not selftext:
            continue  # need a self-post body to narrate (skip link/image posts)
        t = RedditThread(
            title=(d.get("title") or "").strip(),
            selftext=selftext,
            score=int(d.get("score") or 0),
            num_comments=int(d.get("num_comments") or 0),
            permalink=str(d.get("permalink") or ""),
            over_18=bool(d.get("over_18")),
        )
        if t.word_count >= min_words:
            threads.append(t)
    threads.sort(key=lambda x: x.word_count, reverse=True)
    return threads


def pick_best(threads: list[RedditThread]) -> Optional[RedditThread]:
    """Highest-scoring thread among the fetched set (they're already SFW)."""
    if not threads:
        return None
    return max(threads, key=lambda t: t.score)


def format_for_narration(thread: RedditThread) -> str:
    """Turn a thread into a compact context block for the narration writer.
    The writer rewrites/summarizes this — we do NOT paste it verbatim into the
    final video (avoids reproducing the source and keeps it YT-original)."""
    return (
        f"Source: a popular Reddit story titled \"{thread.title}\".\n\n"
        f"Story:\n{thread.selftext}\n\n"
        f"Task: retell this as an original, engaging narrated short. Rewrite in "
        f"your own words, keep it faithful but concise, remove usernames and "
        f"personal identifiers, and make it advertiser-friendly."
    )


class RedditFetchError(RuntimeError):
    """Reddit listing could not be fetched or parsed."""
    pass
