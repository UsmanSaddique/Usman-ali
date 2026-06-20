import sys, os
sys.path.insert(0, os.getcwd())
from app.database import get_session, Project, Scene, Generation
s = get_session()
p = s.query(Project).filter(Project.id.like("b6575e5c%")).first()
print("project:", p.title[:40], "| status:", p.status)
for sc in s.query(Scene).filter(Scene.project_id==p.id).order_by(Scene.scene_number).all():
    g = s.query(Generation).filter(Generation.scene_id==sc.id).order_by(Generation.created_at.desc()).first()
    print(f"  scene {sc.scene_number} type={sc.scene_type.value} status={sc.status.value} err={(g.error_log if g else 'no-gen')!r}"[:180])
s.close()
