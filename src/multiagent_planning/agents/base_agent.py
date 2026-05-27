from abc import ABC, abstractmethod
from typing import Any, Dict
from src.multiagent_planning.models.local_llm import LocalLLM

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

class BaseAgent(ABC):
    """
    Abstract Base Class for multi-agent system participants.
    """
    def __init__(self, agent_id: str, role: str, model_wrapper: LocalLLM):
        self.agent_id = agent_id
        self.role = role
        self.model_wrapper = model_wrapper

    @abstractmethod
    def plan_step(self, environment_state: Dict[str, Any], history: str) -> Dict[str, Any]:
        """
        Execute one reasoning/planning step.
        Returns a dictionary containing 'thought', 'action', and any parameters.
        """
        pass
