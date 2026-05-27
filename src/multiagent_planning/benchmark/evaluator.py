import time
import logging
from typing import List, Dict, Any, Union, Optional
from tqdm import tqdm
from langgraph.graph.state import CompiledStateGraph
from src.multiagent_planning.planning.environment import CoordinationEnvironment
from src.multiagent_planning.agents.base_agent import BaseAgent
from src.multiagent_planning.telemetry.logger import TelemetryCapture

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger(__name__)

class PlanningEvaluator:
    """
    Evaluator suite that benchmarks either standard agent loops or compiled LangGraph setups.
    Includes VRAM monitoring and W&B logging.
    """
    def __init__(self, environment: CoordinationEnvironment, agents: Optional[List[BaseAgent]] = None):
        self.env = environment
        self.agents = agents or []

    def run_graph_benchmark(self, graph: CompiledStateGraph, goal: str, sessions: int = 1) -> List[Dict[str, Any]]:
        """
        Runs multiple benchmark sessions on the compiled LangGraph, displaying a tqdm progress bar.
        """
        logger.info(f"Starting LangGraph benchmark ({sessions} runs)...")
        results = []

        # Run tqdm progress tracker over sessions
        for i in tqdm(range(sessions), desc="Evaluating LangGraph sessions"):
            start_time = time.time()
            
            # Execute compiled graph state machine
            initial_state = {
                "goal": goal,
                "plan": [],
                "tasks_completed": [],
                "verification_feedback": "",
                "approved": False,
                "iterations": 0,
                "logs": []
            }
            
            outcome = graph.invoke(initial_state)
            duration = time.time() - start_time
            
            success = outcome.get("approved", False)
            metrics = {
                "session_index": i,
                "success": success,
                "iterations": outcome.get("iterations", 0),
                "tasks_completed_count": len(outcome.get("tasks_completed", [])),
                "duration_sec": duration,
                "logs_count": len(outcome.get("logs", []))
            }
            results.append(metrics)
            logger.info(f"Session {i} finished. Success: {success}, Iterations: {metrics['iterations']}.")
            
        return results

    def run_benchmark(self, max_steps: int = 10) -> Dict[str, Any]:
        """
        Fallback/legacy benchmark for simple alternating agent interactions.
        """
        logger.info("Starting legacy planning benchmark...")
        start_time = time.time()
        
        state = self.env.reset()
        history = "Planning started.\n"
        
        step_count = 0
        rewards = []
        conflicts_encountered = 0

        # Run tqdm progress tracker over steps
        pbar = tqdm(total=max_steps, desc="Legacy evaluation steps")
        while not state["done"] and step_count < max_steps:
            agent = self.agents[step_count % len(self.agents)]
            
            decision = agent.plan_step(state, history)
            action = decision["action"]
            thought = decision["thought"]
            
            step_log = f"Step {step_count}: Agent {agent.agent_id} decided:\n  Thought: {thought}\n  Action: {action}\n"
            history += step_log
            
            state, reward, observation = self.env.step(agent.agent_id, action)
            rewards.append(reward)
            
            if state["conflicts"]:
                conflicts_encountered += len(state["conflicts"])
                
            history += f"  Observation: {observation}\n"
            step_count += 1
            pbar.update(1)
            
        pbar.close()
        duration = time.time() - start_time
        success = (not state["unallocated_tasks"]) and (not state["conflicts"])

        metrics = {
            "success": success,
            "total_steps": step_count,
            "unallocated_tasks_remaining": len(state["unallocated_tasks"]),
            "final_conflicts_remaining": len(state["conflicts"]),
            "conflicts_encountered_count": conflicts_encountered,
            "total_reward": sum(rewards),
            "execution_duration_sec": duration,
            "steps_per_second": step_count / max(duration, 0.001),
            "history_log": history
        }

        logger.info(f"Benchmark finished. Success: {success}, Steps: {step_count}, Reward: {sum(rewards):.2f}")
        return metrics
