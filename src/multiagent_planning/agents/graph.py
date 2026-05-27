import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from src.multiagent_planning.agents.nodes import PlannerNode, WorkerNode, VerifierNode
from src.multiagent_planning.models.factory import ModelBackend
from src.multiagent_planning.config import MAX_PLANNING_STEPS

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    goal: str
    plan: List[str]
    tasks_completed: List[Dict[str, Any]]
    verification_feedback: str
    approved: bool
    iterations: int
    logs: List[str]


def build_planning_graph(model: ModelBackend) -> StateGraph:
    """
    Compiles the multi-agent execution state machine.
    """
    # 1. Initialize nodes
    planner = PlannerNode(model)
    worker = WorkerNode(model)
    verifier = VerifierNode(model)

    # 2. Build graph builder
    builder = StateGraph(AgentState)

    # 3. Add graph nodes
    builder.add_node("planner", planner.execute)
    builder.add_node("worker", worker.execute)
    builder.add_node("verifier", verifier.execute)

    # 4. Set edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "worker")
    builder.add_edge("worker", "verifier")

    # 5. Define Routing Decision Edge
    def should_continue(state: AgentState) -> str:
        if state.get("approved", False):
            logger.info("[Routing] Verifier approved plan completion. Ending run.")
            return "end"
        if state.get("iterations", 0) >= MAX_PLANNING_STEPS:
            logger.warning("[Routing] Maximum iteration limit reached. Ending run.")
            return "end"
        
        logger.info(f"[Routing] Verifier rejected tasks. Requesting replan (Iteration: {state.get('iterations', 0)}).")
        return "replan"

    builder.add_conditional_edges(
        "verifier",
        should_continue,
        {
            "end": END,
            "replan": "planner"
        }
    )

    # 6. Compile graph
    logger.info("LangGraph compiled successfully.")
    return builder.compile()
