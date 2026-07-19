"""
Native LTX Director End-to-End QA Test
Bootstraps a real project into the SQLite database and pushes it through the native Pipeline.
This verifies the exact logic that the web frontend uses.
"""
import os
import sys
import uuid
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import get_session, Project, Scene, ProjectStatus, SceneStatus
from app.config import settings
from app.services.model_manager import ModelManager
from app.services.pipeline import PipelineOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STORY_SCENES = [
    {
        "prompt": "wide shot, a cute 6-year-old cartoon fairy girl with sparkling blue wings, long silver hair, wearing a glowing pastel dress takes flight into the cool night air followed by a flock of cute glowing baby birds in a vast starry sky, slow zoom in, 3D cartoon animation, soft dreamlike lighting, high quality render, magical pastel tones, deep indigo and glowing golds",
        "dialogue": "Once upon a time, a little fairy named Elara took to the night sky."
    },
    {
        "prompt": "close up, the fairy girl smiling gently as a glowing baby bird lands on her outstretched hand. Magical particles floating in the air, 3D cartoon animation, soft dreamlike lighting.",
        "dialogue": "She was never alone, for the glowing birds were her dearest friends."
    },
    {
        "prompt": "wide shot, the fairy girl soaring above a glowing enchanted forest, massive glowing mushrooms and sparkling trees. 3D cartoon animation, magical pastel tones, cinematic lighting.",
        "dialogue": "Together, they flew over the whispering woods..."
    },
    {
        "prompt": "dynamic shot, the fairy girl twirling in the air, leaving a trail of sparkling stardust behind her as she flies towards a giant glowing moon. 3D cartoon animation.",
        "dialogue": "...leaving trails of stardust in their wake."
    }
]

def run_native_qa():
    logger.info("Initializing Native Pipeline for QA...")
    pipeline = PipelineOrchestrator(settings, ModelManager())
    project_id = str(uuid.uuid4())
    
    session = get_session()
    try:
        # 1. Create a real project
        proj = Project(
            id=project_id,
            title="Native LTX Director QA Test",
            channel_id="test-channel-123",
            duration_target=60,
            status=ProjectStatus.SCRIPTED,
            project_type="narration"
        )
        session.add(proj)
        
        # 2. Inject 10 Scenes
        for i, sc in enumerate(STORY_SCENES):
            scene = Scene(
                id=str(uuid.uuid4()),
                project_id=project_id,
                scene_number=i+1,
                scene_type="txt2vid",
                prompt=sc["prompt"],
                narration_text=sc["dialogue"],
                duration=6.0,
                status=SceneStatus.PENDING
            )
            session.add(scene)
        
        session.commit()
        logger.info(f"Created real project {project_id} with 10 scenes in DB.")
    finally:
        session.close()
        
    # 3. Generate Stills (Natively updates DB and project dir)
    logger.info("Triggering Native Image Generation...")
    pipeline.start_generation(project_id)
    
    # 4. Generate LTX Video (Natively handles the multi-segment request + FFmpeg upscale + QA check)
    logger.info("Triggering Native LTX Director generation...")
    final_out = pipeline.generate_ltx_director(project_id)
    
    logger.info(f"NATIVE QA COMPLETE! Check the web UI for project {project_id}.")
    logger.info(f"Final upscaled video is at: {final_out}")

if __name__ == "__main__":
    run_native_qa()
