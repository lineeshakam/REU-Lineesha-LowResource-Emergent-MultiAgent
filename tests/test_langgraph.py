import pytest
import os
from pathlib import Path
from src.multiagent_planning.models.factory import get_model_backend
from src.multiagent_planning.agents.graph import build_planning_graph
from src.multiagent_planning.agents.nodes import WorkerNode
from src.multiagent_planning.telemetry.logger import TelemetryCapture

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

def test_langgraph_compilation_and_execution():
    """
    Verifies that the compiled LangGraph executes node-to-node state transitions correctly.
    """
    # 1. Resolve mock model
    model = get_model_backend("transformers", "gpt2", use_mock=True)
    
    # 2. Build graph
    graph = build_planning_graph(model)
    assert graph is not None
    
    # 3. Invoke graph
    initial_state = {
        "goal": "Test agent coordination",
        "plan": [],
        "tasks_completed": [],
        "verification_feedback": "",
        "approved": False,
        "iterations": 0,
        "logs": []
    }
    
    outcome = graph.invoke(initial_state)
    
    assert outcome["goal"] == "Test agent coordination"
    assert len(outcome["plan"]) > 0
    assert len(outcome["tasks_completed"]) > 0
    assert outcome["approved"] is True
    assert outcome["iterations"] == 1

def test_worker_parallel_execution():
    """
    Verifies that WorkerNode processes multiple planning tasks concurrently.
    """
    model = get_model_backend("transformers", "gpt2", use_mock=True)
    worker = WorkerNode(model)
    
    state = {
        "plan": ["Task A description", "Task B description", "Task C description"],
        "tasks_completed": [],
        "logs": []
    }
    
    updated_state = worker.execute(state)
    assert len(updated_state["tasks_completed"]) == 3
    assert updated_state["tasks_completed"][0]["task"] in state["plan"]

def test_telemetry_capturing():
    """
    Verifies that TelemetryCapture redirects output and saves logs.
    """
    session = "test_telemetry_run"
    with TelemetryCapture(session) as tel:
        print("Standard output test content log line")
        
    stdout_file = Path(f"logs/{session}_stdout.log")
    assert stdout_file.exists()
    
    logs_content = stdout_file.read_text(encoding="utf-8")
    assert "Standard output test content log line" in logs_content
    
    # Clean up test output logs
    stdout_file.unlink()
    Path(f"logs/{session}_stderr.log").unlink()
