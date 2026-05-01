"""
Shared state definition for the multi-agent supervisor system.
All agents (supervisor + workers) read from and write to this state.
"""

from typing import Annotated, Any, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# Valid specialist worker names
WorkerName = Literal["researcher", "coder", "critic", "FINISH"]


class WorkerOutput(TypedDict):
    """Structured output from a specialist worker."""
    worker: str
    content: str
    metadata: dict[str, Any]


class TeamState(TypedDict):
    """
    Shared state across the entire multi-agent supervisor system.

    Fields:
        messages: Full conversation history (supervisor + all workers).
        task: The original user task/goal.
        next_worker: Which specialist the supervisor routes to next.
        worker_outputs: Collected outputs from all specialist workers.
        current_plan: Supervisor's decomposition of the task into sub-tasks.
        iteration: Number of supervisor→worker cycles completed.
        final_response: The synthesized final answer from the supervisor.
        error: Any error message encountered during processing.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    next_worker: Optional[WorkerName]
    worker_outputs: list[WorkerOutput]
    current_plan: list[str]
    iteration: int
    final_response: Optional[str]
    error: Optional[str]
