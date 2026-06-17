"""
AI Director — YouTube Upload Service
Upload videos, set metadata, and manage scheduling via YouTube Data API v3.

Setup required:
1. Create a Google Cloud project
2. Enable YouTube Data API v3
3. Create OAuth2 credentials (Desktop app type)
4. Download client_secrets.json to D:/ai-director/credentials/
5. First run will open browser for OAuth consent
"""
import os
import time
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    video_id: str
    youtube_url: str
    title: str
    status: str
    publish_at: Optional[str] = None
    upload_time: float = 0.0


class YouTubeUploadService:
    """
    Upload finished videos to YouTube via Data API v3.
    Handles OAuth2 authentication, video upload, metadata, thumbnails,
    and scheduled publishing.
    """

    def __init__(self, config):
        self.config = config
        self.credentials_dir = config.paths.base_dir / "credentials"
        self.client_secrets = self.credentials_dir / "client_secrets.json"
        self.token_file = self.credentials_dir / "youtube_token.json"
        self._youtube = None

    # ── Authentication ─────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """
        Authenticate with YouTube API via OAuth2.
        First run opens browser for consent. Token is cached for reuse.
        """
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

            creds = None

            # Load cached token
            if self.token_file.exists():
                creds = Credentials.from_authorized_user_file(
                    str(self.token_file), SCOPES
                )

            # Refresh or get new token
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not self.client_secrets.exists():
                        logger.error(
                            f"[YouTube] client_secrets.json not found at "
                            f"{self.client_secrets}. Download from Google Cloud Console."
                        )
                        return False

                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.client_secrets), SCOPES
                    )
                    creds = flow.run_local_server(port=8090)

                # Save token
                self.credentials_dir.mkdir(parents=True, exist_ok=True)
                with open(self.token_file, 'w') as f:
                    f.write(creds.to_json())

            self._youtube = build("youtube", "v3", credentials=creds)
            logger.info("[YouTube] Authenticated successfully")
            return True

        except ImportError:
            logger.error(
                "[YouTube] Required packages not installed. Run: "
                "pip install google-auth-oauthlib google-api-python-client"
            )
            return False
        except Exception as e:
            logger.error(f"[YouTube] Authentication failed: {e}")
            return False

    # ── Upload ─────────────────────────────────────────────────────────

    def upload(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: list[str],
        category_id: str = "24",   # Entertainment (22=People, 24=Entertainment, 27=Education)
        privacy: str = "private",  # private, unlisted, public
        made_for_kids: bool = False,
        thumbnail_path: Optional[str] = None,
        publish_at: Optional[str] = None,  # ISO 8601 datetime for scheduled publish
        default_language: str = "en",
    ) -> UploadResult:
        """
        Upload a video to YouTube with full metadata.

        Args:
            video_path: Path to the rendered MP4 file
            title: Video title (max 100 chars)
            description: Video description (max 5000 chars)
            tags: List of tags (max 500 chars total)
            category_id: YouTube category ID
            privacy: Privacy status
            made_for_kids: COPPA compliance flag
            thumbnail_path: Custom thumbnail image path
            publish_at: Schedule publish time (requires privacy="private")
            default_language: Content language
        """
        if not self._youtube:
            if not self.authenticate():
                raise RuntimeError("YouTube authentication failed")

        from googleapiclient.http import MediaFileUpload

        t0 = time.time()
        logger.info(f"[YouTube] Uploading: {title}")

        # Build metadata
        body = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tags,
                "categoryId": category_id,
                "defaultLanguage": default_language,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": made_for_kids,
            },
        }

        # Scheduled publishing
        if publish_at and privacy == "private":
            body["status"]["publishAt"] = publish_at
            body["status"]["privacyStatus"] = "private"

        # Upload
        media = MediaFileUpload(
            video_path,
            mimetype="video/mp4",
            resumable=True,
            chunksize=10 * 1024 * 1024,  # 10MB chunks
        )

        request = self._youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                logger.info(f"[YouTube] Upload progress: {pct}%")

        video_id = response["id"]
        logger.info(f"[YouTube] Uploaded: https://youtube.com/watch?v={video_id}")

        # Set custom thumbnail
        if thumbnail_path and Path(thumbnail_path).exists():
            try:
                self._set_thumbnail(video_id, thumbnail_path)
            except Exception as e:
                logger.warning(f"[YouTube] Thumbnail upload failed: {e}")

        elapsed = time.time() - t0

        return UploadResult(
            video_id=video_id,
            youtube_url=f"https://youtube.com/watch?v={video_id}",
            title=title,
            status=privacy,
            publish_at=publish_at,
            upload_time=elapsed,
        )

    def _set_thumbnail(self, video_id: str, thumbnail_path: str):
        """Upload custom thumbnail for a video."""
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(thumbnail_path, mimetype="image/jpeg")
        self._youtube.thumbnails().set(
            videoId=video_id,
            media_body=media,
        ).execute()
        logger.info(f"[YouTube] Custom thumbnail set for {video_id}")

    # ── Description Builder ────────────────────────────────────────────

    @staticmethod
    def build_description(
        title: str,
        channel_profile: dict,
        video_summary: str = "",
        hashtags: list[str] = None,
    ) -> str:
        """Build an SEO-optimized YouTube description from channel profile."""
        parts = []

        if video_summary:
            parts.append(video_summary)
            parts.append("")

        # Tags as hashtags
        if hashtags:
            tag_line = " ".join(f"#{t.replace(' ', '')}" for t in hashtags[:15])
            parts.append(tag_line)
            parts.append("")

        # Channel-specific boilerplate
        audience = channel_profile.get("audience", "")
        if "kids" in audience:
            parts.extend([
                "🌟 A magical journey for little ones! 🌟",
                "",
                "Perfect for bedtime or quiet playtime.",
                "Gentle visuals and soft music to help your child relax.",
                "",
                "Subscribe for daily magical adventures! 🦄✨",
            ])

        return "\n".join(parts)

    @staticmethod
    def build_tags(channel_profile: dict, title: str) -> list[str]:
        """Build tag list from channel profile template + title keywords."""
        tags = []

        # Template tags from profile
        template = channel_profile.get("monetization", {}).get("tags_template", [])
        tags.extend(template)

        # Add title words as tags
        title_words = [w.lower() for w in title.split() if len(w) > 3]
        tags.extend(title_words[:5])

        # Deduplicate and limit
        seen = set()
        unique_tags = []
        for t in tags:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_tags.append(t)

        return unique_tags[:30]  # YouTube limit ~500 chars total
