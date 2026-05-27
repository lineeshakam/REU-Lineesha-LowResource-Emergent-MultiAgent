import pytest
import torch
from src.multiagent_planning.config import OFFLINE_MODE
from src.multiagent_planning.models.local_llm import LocalLLM
from src.multiagent_planning.agents.planning_agent import PlanningAgent
from src.multiagent_planning.planning.environment import CoordinationEnvironment
from src.multiagent_planning.planning.mcts import MCTSPlanner, MCTSNode

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

def test_offline_mode_configuration():
    """Verify that offline mode environment configuration is set."""
    assert isinstance(OFFLINE_MODE, bool)

def test_mock_llm_generation():
    """Verify that LocalLLM runs in mock mode correctly."""
    model = LocalLLM(model_name_or_path="gpt2", use_mock=True)
    res = model.generate("Run a plan")
    assert "Thought:" in res or "Action:" in res or "Mock" in res

def test_coordination_environment():
    """Verify state transitions and rewards in coordination environment."""
    env = CoordinationEnvironment(
        tasks=["task_1", "task_2"],
        resources=["res_1", "res_2"]
    )
    
    state = env.reset()
    assert len(state["unallocated_tasks"]) == 2
    assert len(state["available_resources"]) == 2
    
    # Run valid action
    next_state, reward, obs = env.step("Agent_1", "Allocate res_1 to task_1")
    assert "task_1" not in next_state["unallocated_tasks"]
    assert "res_1" not in next_state["available_resources"]
    assert reward > 0
    
    # Run invalid action
    next_state, reward, obs = env.step("Agent_1", "Allocate invalid_res to task_2")
    assert reward < 0

def test_mcts_search():
    """Verify that MCTS tree search produces a trajectory."""
    model = LocalLLM(model_name_or_path="gpt2", use_mock=True)
    agent1 = PlanningAgent(agent_id="Agent_1", role="Planner", model_wrapper=model)
    env = CoordinationEnvironment(
        tasks=["task_a"],
        resources=["res_a"]
    )
    
    planner = MCTSPlanner(environment=env, agents=[agent1])
    trajectory = planner.plan(simulations=5)
    
    # The trajectory should contain steps
    assert isinstance(trajectory, list)
