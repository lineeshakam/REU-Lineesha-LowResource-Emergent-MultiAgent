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

from dataclasses import dataclass
import re
from typing import List, Dict, Any

@dataclass
class ReasoningState:
    language: str
    variables: Dict[str, Any]
    intermediate: str
    constraints: List[str]
    confidence: float
    final: str = ""

class LanguageAgent:
    """ 
    class for language specialized agent; expects model to provide a method .generate(prompt: str) -> str.
    """
    def __init__(self, language: str, model, prompt_template: str = None):
        self.language = language
        self.model = model
        self.prompt = prompt_template or "Problem: {problem}\n\nRespond in the labeled format: INTERMEDIATE:, FINAL:, VARIABLES:, CONSTRAINTS:, CONFIDENCE:"

    def generate_initial_state(self, problem: str) -> ReasoningState:
        raw = self.model.generate(self.prompt.format(problem=problem))
        return self._parse_raw(raw)

    def update_state(self, original_problem: str, current_state: ReasoningState, received_states: List[ReasoningState]) -> ReasoningState:
        others = "\n".join(f"Agent[{s.language}] FINAL: {s.final}" for s in received_states)
        prompt = f"You previously reasoned:\n{current_state.intermediate}\n\nOther agents:\n{others}\n\nIf helpful, revise your INTERMEDIATE and CONFIDENCE and output in the same labeled format.\nProblem: {original_problem}"
        raw = self.model.generate(prompt)
        updated = self._parse_raw(raw)
        updated.language = self.language
        return updated

    def plan_step(self, environment: Dict[str, Any], history: str = "") -> Dict[str, Any]:
        problem = environment.get("problem") if isinstance(environment, dict) else history
        state = self.generate_initial_state(problem or "")
        return {"thought": state.intermediate, "action": "propose", "raw": state}

    def _parse_raw(self, raw: str) -> ReasoningState:
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", raw or "")
        #INTERMEDIATE / FINAL
        m = re.search(r"INTERMEDIATE:\\s*(.*?)\\s*(?:FINAL:|$)", text, re.S | re.I)
        intermediate = m.group(1).strip() if m else text.strip()
        m2 = re.search(r"FINAL:\\s*(.*?)(?:\\n\\n|$)", text, re.S | re.I)
        final = m2.group(1).strip() if m2 else ""
        #VARIABLES (key=value lines)
        vars_block = ""
        mv = re.search(r"VARIABLES:\\s*(.*?)(?:\\n\\n|CONSTRAINTS:|INTERMEDIATE:|$)", text, re.S | re.I)
        if mv: vars_block = mv.group(1)
        vars_map = {}
        for line in vars_block.splitlines():
            if "=" in line:
                k, v = line.split("=",1)
                vars_map[k.strip()] = v.strip()
        #CONSTRAINTS (lines starting with -) (from the prompts.yaml)
        cons = []
        mc = re.search(r"CONSTRAINTS:\\s*(.*?)(?:\\n\\n|INTERMEDIATE:|$)", text, re.S | re.I)
        if mc:
            for line in mc.group(1).splitlines():
                line = line.strip()
                if line.startswith("-"):
                    cons.append(line[1:].strip())
        #CONFIDENCE
        conf_m = re.search(r"CONFIDENCE:\\s*([0-9]*\\.?[0-9]+)", text, re.I)
        try:
            confidence = float(conf_m.group(1)) if conf_m else 0.0
        except Exception:
            confidence = 0.0
        return ReasoningState(language=self.language, variables=vars_map, intermediate=intermediate, constraints=cons, confidence=confidence, final=final)