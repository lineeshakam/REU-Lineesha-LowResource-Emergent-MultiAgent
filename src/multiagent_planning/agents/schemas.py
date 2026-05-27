from pydantic import BaseModel, Field
from typing import List

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu, himonthakur@gmail.com"
__status__ = "prototype"

class PlannerOutput(BaseModel):
    """
    Validation schema for Planner agent outputs.
    """
    plan: List[str] = Field(description="List of step-by-step coordination tasks.")
    justification: str = Field(description="Detailed planning justification and resource reasoning.")


class WorkerOutput(BaseModel):
    """
    Validation schema for Worker agent outputs.
    """
    task: str = Field(description="The task that was executed.")
    status: str = Field(default="completed", description="Execution status, e.g., 'completed', 'failed'.")
    details: str = Field(description="Output log, configuration results, or allocations made.")


class VerifierOutput(BaseModel):
    """
    Validation schema for Verifier agent outputs.
    """
    approved: bool = Field(description="Whether the worker task completions are approved.")
    evaluation: str = Field(description="Feedback justification detailing task validation checks.")
