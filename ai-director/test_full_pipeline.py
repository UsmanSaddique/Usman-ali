import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.database import init_db, get_session, Channel, Project, Scene, SceneType, ProjectStatus
from app.services.model_manager import ModelManager, register_all_loaders
from app.services.pipeline import PipelineOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("TestPipeline")

def run_test():
    logger.info("Initializing Database...")
    db_path = str(settings.paths.database)
    init_db(db_path)
    session = get_session()

    # Create dummy channel
    channel_slug = "test-automation-channel"
    channel = session.query(Channel).filter(Channel.slug == channel_slug).first()
    if not channel:
        channel = Channel(
            name="Test Automation Channel",
            slug=channel_slug,
            system_prompt="Test channel for overnight automation.",
            target_resolution="1080p",
        )
        session.add(channel)
        session.commit()
        logger.info("Created test channel.")

    # Create test project
    project = Project(
        title="Overnight Automation Test",
        channel_id=channel.id,
        duration_target=10,
        context="A quick test for automation pipeline",
        num_scenes_target=2,
        video_model=settings.video.name,
        status=ProjectStatus.DRAFT
    )
    session.add(project)
    session.commit()
    logger.info(f"Created test project: {project.id}")

    # Add 2 scenes: one text-to-video, one still image pan (Ken Burns)
    scene1 = Scene(
        project_id=project.id,
        scene_number=1,
        scene_type=SceneType.TXT2VID,
        prompt="A cute golden retriever running in a green park, sunny day, 4k",
        duration=4.0,
    )
    
    scene2 = Scene(
        project_id=project.id,
        scene_number=2,
        scene_type=SceneType.STILL_PAN,
        prompt="A beautiful sunset over the mountains, digital art",
        duration=4.0,
        camera_motion="zoom_in"
    )

    session.add_all([scene1, scene2])
    session.commit()
    logger.info("Added test scenes.")

    # Initialize Managers
    logger.info("Initializing ModelManager and PipelineOrchestrator...")
    model_manager = ModelManager()
    register_all_loaders(model_manager, settings)
    pipeline = PipelineOrchestrator(settings, model_manager)

    # We will simulate the `run_full_auto` pipeline but manually call start_generation and render
    # to avoid needing the director LLM for scripting in this specific test.
    project.status = ProjectStatus.APPROVED
    session.commit()

    logger.info("Starting Phase 3: Asset Generation")
    try:
        pipeline.start_generation(project.id)
        logger.info("Asset Generation Complete.")
    except Exception as e:
        logger.error(f"Asset Generation Failed: {e}", exc_info=True)
        return

    logger.info("Starting Phase 4: Upscale")
    try:
        pipeline.start_upscale(project.id)
        logger.info("Upscale Complete.")
    except Exception as e:
        logger.error(f"Upscale Failed: {e}", exc_info=True)
        return

    logger.info("Starting Phase 6: Assembly")
    try:
        pipeline.render(project.id)
        logger.info("Assembly Complete.")
    except Exception as e:
        logger.error(f"Assembly Failed: {e}", exc_info=True)
        return

    session.refresh(project)
    if project.status == ProjectStatus.RENDERED:
        logger.info(f"Test Successful! Final Video Path: {project.output_path}")
    else:
        logger.error(f"Test finished but project status is {project.status.value}")

if __name__ == "__main__":
    run_test()
