import logging
from typing import Any, Dict, List, Tuple

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger(__name__)

class CoordinationEnvironment:
    """
    Simulates a multi-agent coordination environment.
    Agents must cooperatively allocate tasks or resources without conflicts.
    """
    def __init__(self, tasks: List[str], resources: List[str]):
        self.initial_tasks = list(tasks)
        self.initial_resources = list(resources)
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.tasks = list(self.initial_tasks)
        self.resources = list(self.initial_resources)
        self.allocations: Dict[str, str] = {} # task -> resource
        self.conflicts: List[str] = []
        self.steps = 0
        self.done = False
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        return {
            "unallocated_tasks": self.tasks,
            "available_resources": self.resources,
            "current_allocations": self.allocations,
            "conflicts": self.conflicts,
            "steps": self.steps,
            "done": self.done
        }

    def step(self, agent_id: str, action: str) -> Tuple[Dict[str, Any], float, str]:
        """
        Executes action proposed by an agent.
        Returns:
            Tuple of (new_state, reward, observation_message)
        """
        self.steps += 1
        reward = 0.0
        obs = ""

        action_clean = action.strip().lower()

        # Parse actions like "allocate resource A to task B" or "allocate B to A"
        # We will look for mentions of tasks and resources in the action string
        task_found = None
        resource_found = None

        for t in self.initial_tasks:
            if t.lower() in action_clean:
                task_found = t
                break
        
        for r in self.initial_resources:
            if r.lower() in action_clean:
                resource_found = r
                break

        if "finalize" in action_clean or "done" in action_clean:
            if not self.tasks and not self.conflicts:
                reward = 10.0
                obs = "Plan finalized successfully! All tasks completed without conflict."
                self.done = True
            else:
                reward = -5.0
                obs = f"Plan finalization rejected. Remaining tasks: {self.tasks}. Conflicts: {self.conflicts}."
                
        elif task_found and resource_found:
            # Check if resource is already allocated to someone else
            current_owner = [t for t, r in self.allocations.items() if r == resource_found]
            
            if resource_found not in self.resources and current_owner:
                # Conflict!
                conflict_msg = f"Conflict: Resource {resource_found} is already allocated to task {current_owner[0]}."
                if conflict_msg not in self.conflicts:
                    self.conflicts.append(conflict_msg)
                reward = -2.0
                obs = f"Failed: {conflict_msg}"
            else:
                # Successful allocation
                if task_found in self.tasks:
                    self.tasks.remove(task_found)
                self.allocations[task_found] = resource_found
                if resource_found in self.resources:
                    self.resources.remove(resource_found)
                
                # Resolve conflict message if it existed
                resolved_msg = f"Conflict: Resource {resource_found} is already allocated"
                self.conflicts = [c for c in self.conflicts if resolved_msg not in c]
                
                reward = 2.0
                obs = f"Success: Allocated {resource_found} to {task_found}."
        else:
            reward = -1.0
            obs = f"Action could not be executed: '{action}'. Invalid task or resource identifier."

        # Maximum depth cutoff
        if self.steps >= 15:
            self.done = True
            if self.tasks:
                reward -= 5.0
                obs += " Planning timeout reached with uncompleted tasks."

        return self.get_state(), reward, obs
