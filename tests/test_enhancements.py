import pytest
from pathlib import Path
from src.multiagent_planning.prompts import PromptRegistry
from src.multiagent_planning.telemetry.logger import TelemetryCapture
from src.multiagent_planning.agents.schemas import PlannerOutput, WorkerOutput, VerifierOutput
from src.multiagent_planning.peft.replay import replay_and_validate_trajectory
from src.multiagent_planning.peft.finetune import export_trajectory_dataset

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

def test_prompt_registry_loading():
    """Verify that PromptRegistry loads and formats variables from config/prompts.yaml."""
    registry = PromptRegistry()
    assert registry.get_prompt("planner", "system") != ""
    formatted = registry.format_user_prompt("planner", goal="Solve resource conflicts")
    assert "Solve resource conflicts" in formatted

def test_pydantic_output_guardrails():
    """Verify that Pydantic models validate agent payloads correctly."""
    # Valid Planner payload
    planner_data = {"plan": ["Task 1", "Task 2"], "justification": "Good logic"}
    validated = PlannerOutput(**planner_data)
    assert validated.plan == ["Task 1", "Task 2"]

    # Invalid Planner payload
    with pytest.raises(ValueError):
        PlannerOutput(plan="Not a list", justification="Bad logic")

def test_telemetry_profiling_and_dataset_export():
    """Verify that TelemetryCapture records VRAM/duration profiles and exports them to dataset logs."""
    session = "test_profiling_session"
    
    with TelemetryCapture(session) as telemetry:
        # Simulate simple agent step
        metrics = telemetry.get_profile_metrics()
        assert "duration_sec" in metrics
        
    trajectories = [
        {"prompt": "Step 1 goal", "completion": "Task 1 complete"}
    ]
    
    export_file = Path(f"logs/{session}_trajectory.jsonl")
    export_trajectory_dataset(trajectories, str(export_file), profile_metrics=metrics)
    
    assert export_file.exists()
    
    # 4. Verify using SFT Trajectory Replay validator
    success = replay_and_validate_trajectory(str(export_file))
    assert success is True
    
    # Cleanup logs
    export_file.unlink()
    Path(f"logs/{session}_stdout.log").unlink()
    Path(f"logs/{session}_stderr.log").unlink()
