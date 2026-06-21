"""
Run the LLM prompt-enhancement pass on a project's scenes, save the richer
prompts to the DB, and print before/after. Then the project can be regenerated
with the enhanced (more cinematic, detailed) prompts.

Usage: python enhance_prompts.py <project_prefix>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from app.config import settings
from app.database import get_session, Project, Scene
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.director import DirectorService

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "ec26a326"
mm = ModelManager(); register_all_loaders(mm, settings)
director = DirectorService(mm, settings)

s = get_session()
p = s.query(Project).filter(Project.id.like(PREFIX + "%")).first()
pid = p.id
slug = p.channel.slug
scenes = s.query(Scene).filter(Scene.project_id == pid).order_by(Scene.scene_number).all()
scene_dicts = [{"scene_number": sc.scene_number, "prompt": sc.prompt,
                "negative_prompt": sc.negative_prompt} for sc in scenes]
s.close()

print(f"Enhancing {len(scene_dicts)} scene prompts for {slug} (LLM)...\n")
enhanced = director.enhance_prompts(scene_dicts, slug)
director.manager.unload()

by_num = {e.get("scene_number"): e for e in enhanced}
s = get_session()
for sc in s.query(Scene).filter(Scene.project_id == pid).order_by(Scene.scene_number).all():
    e = by_num.get(sc.scene_number)
    if not e:
        continue
    print("=" * 70)
    print(f"SCENE {sc.scene_number}")
    print(f"  BEFORE ({len(sc.prompt.split())}w): {sc.prompt[:160]}")
    new_p = e.get("prompt", "").strip()
    if new_p:
        print(f"  AFTER  ({len(new_p.split())}w): {new_p[:240]}")
        sc.prompt = new_p
        if e.get("negative_prompt"):
            sc.negative_prompt = e["negative_prompt"]
s.commit(); s.close()
print("\n" + "=" * 70)
print(f"Saved enhanced prompts to project {pid[:8]}. Re-generate to see the richer scenes.")
print(f"PROJECT_ID={pid}")
