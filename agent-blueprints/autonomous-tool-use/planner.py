"""
Pydantic models for the Plan-and-Execute agent's planning layer.

Defines the structured data contracts used by planner, executor, and replanner nodes.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in an execution plan."""
    step_number: int = Field(description="1-based index of this step in the plan")
    description: str = Field(description="Clear, actionable description of what to do in this step")
    tool_hint: Optional[str] = Field(
        default=None,
        description="Optional hint about which tool to use: 'weather', 'calculator', 'wikipedia', 'file_ops', or None"
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="Step numbers that must complete before this step can run"
    )


class Plan(BaseModel):
    """A structured execution plan decomposed from the user's goal."""
    goal: str = Field(description="The original user goal being planned for")
    steps: list[str] = Field(description="Ordered list of step descriptions to accomplish the goal")
    reasoning: str = Field(description="Brief explanation of the planning approach")

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def get_step(self, index: int) -> Optional[str]:
        """Get step description by 0-based index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None


class ExecutionResult(BaseModel):
    """Result of executing a single plan step."""
    step_number: int = Field(description="1-based step number that was executed")
    step_description: str = Field(description="Description of the step that was executed")
    tool_used: Optional[str] = Field(default=None, description="Name of the tool used, if any")
    tool_input: Optional[dict[str, Any]] = Field(default=None, description="Input passed to the tool")
    output: str = Field(description="Result or output from executing this step")
    success: bool = Field(description="Whether the step completed successfully")
    error: Optional[str] = Field(default=None, description="Error message if step failed")

    def summary(self) -> str:
        status = "OK" if self.success else "FAIL"
        tool = f" [{self.tool_used}]" if self.tool_used else ""
        return f"Step {self.step_number}{tool} [{status}]: {self.output[:200]}"


class ReplanDecision(BaseModel):
    """Replanner's decision on whether to continue, revise, or finish."""
    action: str = Field(
        description="One of: 'continue' (proceed with remaining steps), 'revise' (update remaining steps), 'finish' (all goals met)"
    )
    revised_steps: list[str] = Field(
        default_factory=list,
        description="Updated remaining steps if action is 'revise'"
    )
    reasoning: str = Field(description="Explanation for why this replan decision was made")
    final_answer: Optional[str] = Field(
        default=None,
        description="Final synthesized answer if action is 'finish'"
    )
