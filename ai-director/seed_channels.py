"""
AI Director — Channel Seeder
Reads every *.yaml in the channels/ dir and upserts a DB Channel row so the
profile is selectable when creating a project. Re-run safely any time you add
or edit a channel profile.

Run:  python_embeded\\python.exe seed_channels.py
"""
import sys, os, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.database import get_session, Channel


def upsert_channels():
    cdir = settings.paths.channels_dir
    session = get_session()
    seeded, updated = [], []
    try:
        for yml in sorted(cdir.glob("*.yaml")):
            with open(yml, "r", encoding="utf-8") as f:
                p = yaml.safe_load(f) or {}
            slug = p.get("slug") or yml.stem
            name = p.get("name", slug)
            gen = p.get("generation", {}) or {}
            row = session.query(Channel).filter(Channel.slug == slug).first()
            fields = dict(
                name=name,
                profile_path=str(yml),
                system_prompt=p.get("vibe", ""),
                still_ratio=float(p.get("still_ratio", 0.4)),
                target_resolution=gen.get("target_resolution", "1080p"),
                made_for_kids=bool(p.get("made_for_kids", False)),
                default_video_model=gen.get("video_model", "ltx-2.3"),
            )
            if row:
                for k, v in fields.items():
                    setattr(row, k, v)
                updated.append(slug)
            else:
                session.add(Channel(slug=slug, **fields))
                seeded.append(slug)
        session.commit()
    finally:
        session.close()

    print(f"Seeded new: {seeded or '(none)'}")
    print(f"Updated:    {updated or '(none)'}")
    print("\nAll channels in DB:")
    s = get_session()
    for c in s.query(Channel).order_by(Channel.name).all():
        print(f"  - {c.slug:24} | {c.name:28} | kids={c.made_for_kids} | still_ratio={c.still_ratio}")
    s.close()


if __name__ == "__main__":
    upsert_channels()
