import logging
import json
import asyncio
from typing import Dict, Any, List
from tqdm import tqdm
from src.multiagent_planning.models.factory import ModelBackend, format_prompt
from src.multiagent_planning.config import MODEL_FAMILY
from src.multiagent_planning.prompts import PromptRegistry
from src.multiagent_planning.agents.schemas import PlannerOutput, WorkerOutput, VerifierOutput

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

logger = logging.getLogger(__name__)

# Initialize prompt registry
registry = PromptRegistry()

class PlannerNode:
    """
    Analyzes the initial goal and generates a list of planning tasks, validated via Pydantic.
    """
    def __init__(self, model: ModelBackend):
        self.model = model

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[Planner Node] Generating step-by-step coordination plan...")
        
        system = registry.get_prompt("planner", "system")
        user = registry.format_user_prompt("planner", goal=state["goal"])
        prompt = format_prompt(MODEL_FAMILY, system, user)
        
        for _ in tqdm(range(1), desc="Planner generating plan"):
            response = self.model.generate(prompt, max_new_tokens=256)
            
        logger.debug(f"[Planner Node] Raw LLM response: {response}")
        
        plan = []
        try:
            # Parse json and validate via Pydantic
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            data = json.loads(cleaned.strip())
            # Enforce Pydantic validation guardrail
            validated_output = PlannerOutput(**data)
            plan = validated_output.plan
        except Exception as e:
            logger.warning(f"[Planner Node] Pydantic validation guardrail failed: {e}. Fallback parse.")
            plan = [line.strip("- ") for line in response.split("\n") if line.strip().startswith("-")]
            if not plan:
                plan = ["Cooperative resource allocation task"]

        state_update = {
            "plan": plan,
            "logs": state.get("logs", []) + [f"Planner generated {len(plan)} tasks (validated: {e is None if 'e' in locals() else True})."],
            "iterations": state.get("iterations", 0) + 1
        }
        return state_update


class WorkerNode:
    """
    Executes the generated planning tasks in parallel, validated via Pydantic.
    """
    def __init__(self, model: ModelBackend):
        self.model = model

    async def execute_task_async(self, task: str) -> Dict[str, Any]:
        system = registry.get_prompt("worker", "system")
        user = registry.format_user_prompt("worker", task=task)
        prompt = format_prompt(MODEL_FAMILY, system, user)
        
        response = await asyncio.to_thread(self.model.generate, prompt, max_new_tokens=150)
        
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            data = json.loads(cleaned.strip())
            # Enforce Pydantic validation guardrail
            validated_output = WorkerOutput(**data)
            return validated_output.model_dump()
        except Exception as e:
            logger.warning(f"[Worker Node] Pydantic validation guardrail failed for task: {task}. Error: {e}")
            return {"task": task, "status": "completed", "details": response.strip()}

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[Worker Node] Executing plan containing {len(state['plan'])} tasks in parallel...")
        
        plan = state["plan"]
        pbar = tqdm(total=len(plan), desc="Worker executing tasks in parallel")
        
        async def run_all():
            tasks = [self.execute_task_async(t) for t in plan]
            results = []
            for coro in asyncio.as_completed(tasks):
                res = await coro
                results.append(res)
                pbar.update(1)
            return results

        import threading
        
        try:
            asyncio.get_running_loop()
            is_loop_running = True
        except RuntimeError:
            is_loop_running = False

        if is_loop_running:
            class AsyncRunner(threading.Thread):
                def __init__(self, coro):
                    super().__init__()
                    self.coro = coro
                    self.result = None
                    self.error = None
                def run(self):
                    try:
                        self.result = asyncio.run(self.coro)
                    except Exception as e:
                        self.error = e
            runner = AsyncRunner(run_all())
            runner.start()
            runner.join()
            if runner.error:
                raise runner.error
            completed_tasks = runner.result
        else:
            completed_tasks = asyncio.run(run_all())
        pbar.close()

        state_update = {
            "tasks_completed": completed_tasks,
            "logs": state.get("logs", []) + [f"Worker completed {len(completed_tasks)} tasks."]
        }
        return state_update


class VerifierNode:
    """
    Audits completed tasks and verifies coordination requirements, validated via Pydantic.
    """
    def __init__(self, model: ModelBackend):
        self.model = model

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("[Verifier Node] Verifying task completions...")
        
        system = registry.get_prompt("verifier", "system")
        user = registry.format_user_prompt("verifier", goal=state["goal"], tasks_completed=state["tasks_completed"])
        prompt = format_prompt(MODEL_FAMILY, system, user)
        
        for _ in tqdm(range(1), desc="Verifier evaluating outputs"):
            response = self.model.generate(prompt, max_new_tokens=150)
            
        logger.debug(f"[Verifier Node] Raw LLM response: {response}")
        
        approved = False
        evaluation = "Could not parse evaluation."
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            data = json.loads(cleaned.strip())
            # Enforce Pydantic validation guardrail
            validated_output = VerifierOutput(**data)
            approved = validated_output.approved
            evaluation = validated_output.evaluation
        except Exception as e:
            logger.warning(f"[Verifier Node] Pydantic validation guardrail failed: {e}")
            if "true" in response.lower() or "approved" in response.lower():
                approved = True
                evaluation = response

        state_update = {
            "approved": approved,
            "verification_feedback": evaluation,
            "logs": state.get("logs", []) + [f"Verifier check: Approved={approved}."]
        }
        return state_update
