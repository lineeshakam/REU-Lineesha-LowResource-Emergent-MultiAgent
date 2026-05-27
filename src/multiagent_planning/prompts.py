import yaml
from pathlib import Path
from typing import Dict, Any

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

PROMPTS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "prompts.yaml"

class PromptRegistry:
    """
    Manages loading and formatting prompts from config/prompts.yaml.
    """
    def __init__(self, yaml_path: Path = PROMPTS_PATH):
        self.yaml_path = yaml_path
        self.prompts: Dict[str, Any] = {}
        self.load_registry()

    def load_registry(self):
        if not self.yaml_path.exists():
            self.prompts = {
                "planner": {
                    "system": "You are an expert agent planner. Break down the user goal into a clean JSON list of tasks.",
                    "user": "Goal: {goal}\n\nFormat your response as a valid JSON object matching the following structure:\n{{\"plan\": [\"Task 1 description\", \"Task 2 description\"], \"justification\": \"your reasoning\"}}"
                },
                "worker": {
                    "system": "You are a worker agent executing specific planning instructions.",
                    "user": "Execute this task and report completion state:\nTask: {task}"
                },
                "verifier": {
                    "system": "You are a quality verifier agent. Check if the task outputs are correct and address the main goal.",
                    "user": "Goal: {goal}\nTask Outputs: {tasks_completed}\n\nEvaluate outputs and generate a JSON response:\n{{\"approved\": true, \"evaluation\": \"reasoning details\"}}\nSet approved to false if tasks are missing details or contain conflicts."
                }
            }
            return

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f) or {}

    def get_prompt(self, agent_name: str, prompt_type: str = "system") -> str:
        """
        Returns the prompt string.
        """
        return self.prompts.get(agent_name, {}).get(prompt_type, "")

    def format_user_prompt(self, agent_name: str, **kwargs) -> str:
        """
        Retrieves and formats the user prompt with parameters.
        """
        template = self.get_prompt(agent_name, "user")
        return template.format(**kwargs)
