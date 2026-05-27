import math
import logging
from typing import Any, Dict, List, Optional
from src.multiagent_planning.planning.environment import CoordinationEnvironment
from src.multiagent_planning.agents.base_agent import BaseAgent

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger(__name__)

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search.
    Tracks state, visits, quality value, and parent/children relationships.
    """
    def __init__(self, state: Dict[str, Any], parent: Optional["MCTSNode"] = None, action_taken: Optional[str] = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0

    @property
    def uct(self) -> float:
        """
        Calculates Upper Confidence bound for Trees (UCT).
        """
        if self.visits == 0:
            return float("inf")
        if not self.parent or self.parent.visits == 0:
            return self.value / self.visits
        # Exploration constant (sqrt(2))
        exploration = 1.414 * math.sqrt(math.log(self.parent.visits) / self.visits)
        return (self.value / self.visits) + exploration


class MCTSPlanner:
    """
    MCTS implementation that wraps the simulation environment and guides agent trajectory search.
    """
    def __init__(self, environment: CoordinationEnvironment, agents: List[BaseAgent]):
        self.env = environment
        self.agents = agents

    def plan(self, simulations: int = 10) -> List[str]:
        """
        Runs MCTS to search for the best planning trajectory.
        Returns the optimal sequence of actions.
        """
        initial_state = self.env.reset()
        root = MCTSNode(initial_state)

        for sim in range(simulations):
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        # Retrieve best trajectory path
        best_path = []
        curr = root
        while curr.children:
            # Select child with most visits
            best_child = max(curr.children, key=lambda c: c.visits)
            if best_child.action_taken:
                best_path.append(best_child.action_taken)
            curr = best_child
            if curr.state.get("done", False):
                break
        return best_path

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Traverses tree selecting best UCT children until finding a leaf node or terminal state.
        """
        curr = node
        while not curr.state.get("done", False):
            if not curr.children:
                return self._expand(curr)
            curr = max(curr.children, key=lambda c: c.uct)
        return curr

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Generates actions from agents to create new child nodes.
        """
        if node.state.get("done", False):
            return node

        # Determine which agent's turn it is
        agent_idx = node.visits % len(self.agents)
        agent = self.agents[agent_idx]

        # Use the agent to propose an action
        history_str = f"Prev actions: {node.action_taken}" if node.action_taken else "Start planning."
        agent_decision = agent.plan_step(node.state, history_str)
        action = agent_decision["action"]

        # Run mock environment step to see the outcome
        # (Save current state first to reset later)
        # Note: In a production MCTS, we would have a copy/clone mechanism for the env.
        # Here, we do a basic rollback by tracking steps.
        orig_tasks = list(self.env.tasks)
        orig_resources = list(self.env.resources)
        orig_allocations = dict(self.env.allocations)
        orig_conflicts = list(self.env.conflicts)
        orig_steps = self.env.steps
        orig_done = self.env.done

        # Re-apply node's environment context to simulate accurately
        self.env.tasks = list(node.state["unallocated_tasks"])
        self.env.resources = list(node.state["available_resources"])
        self.env.allocations = dict(node.state["current_allocations"])
        self.env.conflicts = list(node.state["conflicts"])
        self.env.steps = node.state["steps"]
        self.env.done = node.state["done"]

        next_state, reward, _ = self.env.step(agent.agent_id, action)

        # Restore env original state
        self.env.tasks = orig_tasks
        self.env.resources = orig_resources
        self.env.allocations = orig_allocations
        self.env.conflicts = orig_conflicts
        self.env.steps = orig_steps
        self.env.done = orig_done

        # Add child
        child = MCTSNode(next_state, parent=node, action_taken=action)
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """
        Performs rollouts or quick evaluations.
        Returns final accumulated reward.
        """
        if node.state.get("done", False):
            # Calculate score
            score = 10.0 - len(node.state["unallocated_tasks"]) * 2.0 - len(node.state["conflicts"]) * 3.0
            return max(-10.0, score)
            
        # Fast random rollout/heuristic evaluation
        unallocated = len(node.state["unallocated_tasks"])
        conflicts = len(node.state["conflicts"])
        steps = node.state["steps"]
        
        # Heuristic reward
        heuristic_val = 5.0 - (unallocated * 1.5) - (conflicts * 2.0) - (steps * 0.1)
        return max(-10.0, heuristic_val)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Propagates simulated rewards back to root node.
        """
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.value += reward
            curr = curr.parent
