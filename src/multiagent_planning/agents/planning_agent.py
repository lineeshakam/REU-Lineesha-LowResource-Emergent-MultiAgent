import logging
import re
from typing import Any, Dict
from src.multiagent_planning.agents.base_agent import BaseAgent
from src.multiagent_planning.models.local_llm import LocalLLM

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger(__name__)

class PlanningAgent(BaseAgent):
    """
    Agent that uses chain-of-thought/ReAct reasoning to solve multi-agent coordination problems.
    """
    def __init__(self, agent_id: str, role: str, model_wrapper: LocalLLM):
        super().__init__(agent_id, role, model_wrapper)
        self.system_prompt = (
            f"You are Agent '{self.agent_id}', playing the role of '{self.role}' in a multi-agent planning scenario.\n"
            "Your objective is to coordinate with other agents to solve a resource allocation and scheduling problem.\n"
            "Analyze the current state and history, then generate a reasoning step using the format below:\n"
            "Thought: <your reasoning about coordination, task constraints, and what to do next>\n"
            "Action: <the specific action you want to take, e.g., 'allocate resource A to Agent 2', 'finalize plan'>\n"
            "Do not output anything else."
        )

    def plan_step(self, environment_state: Dict[str, Any], history: str) -> Dict[str, Any]:
        """
        Runs the LLM model to decide the next planning step.
        """
        # Build prompt using current environment state and conversation history
        prompt = (
            f"{self.system_prompt}\n\n"
            f"--- Current Environment State ---\n{environment_state}\n\n"
            f"--- Planning History ---\n{history}\n\n"
            f"Generate your next step now:"
        )
        
        logger.info(f"Agent {self.agent_id} is reasoning...")
        response = self.model_wrapper.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.4,
            do_sample=True
        )
        
        # Parse Thought and Action from response
        thought = "Could not parse thought."
        action = "none"
        
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", response, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"Action:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
        
        if thought_match:
            thought = thought_match.group(1).strip()
        if action_match:
            action = action_match.group(1).strip()
            
        return {
            "thought": thought,
            "action": action,
            "raw_response": response
        }
