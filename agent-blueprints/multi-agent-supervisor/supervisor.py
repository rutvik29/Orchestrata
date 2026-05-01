"""
Supervisor Agent for the multi-agent team.

The supervisor:
1. Receives the user task
2. Routes to the appropriate specialist worker (ROUTE)
3. Evaluates if more work is needed or synthesizes final response (SYNTHESIZE)
"""

import logging
import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from state import TeamState, WorkerName

logger = logging.getLogger(__name__)

MAX_SUPERVISOR_ITERATIONS = int(os.getenv("MAX_SUPERVISOR_ITERATIONS", "5"))

SUPERVISOR_ROUTE_PROMPT = """You are a Supervisor managing a team of AI specialists.

Your team members:
- **researcher**: Retrieves and summarizes information. Use for factual questions, background info, concept explanations.
- **coder**: Writes Python code and technical implementations. Use for coding tasks, algorithmic problems, technical demos.
- **critic**: Reviews and critiques outputs for accuracy and completeness. Use AFTER researcher and/or coder have contributed.
- **FINISH**: Signal that enough work has been done and you should synthesize the final response.

Current task: {task}

Work completed so far ({num_outputs} specialist outputs):
{outputs_summary}

Iteration: {iteration}/{max_iterations}

Decide who should work next. Rules:
- Start with researcher for knowledge-intensive tasks
- Add coder when code is needed
- Always run critic before finishing if researcher or coder have contributed
- Return FINISH when you have sufficient information to answer the task fully

Respond with ONLY the name of the next worker: researcher, coder, critic, or FINISH
"""

SUPERVISOR_SYNTHESIZE_PROMPT = """You are a Supervisor synthesizing a final, comprehensive response for the user.

Original Task: {task}

Work from your specialist team:
{all_outputs}

Create a unified, polished final response that:
1. Directly addresses the original task
2. Integrates the best insights from all specialists
3. Incorporates feedback from the critic (if available)
4. Is well-structured and easy to understand
5. Includes code examples if the coder provided them
6. Notes any limitations or caveats

Be comprehensive but concise. This is the final response the user will see.
"""


class RoutingDecision(BaseModel):
    """Structured routing decision from the supervisor."""
    next_worker: WorkerName = Field(description="Which specialist to route to next, or FINISH")
    reasoning: str = Field(description="Brief explanation of the routing decision")


def _get_llm():
    """Get configured LLM for the supervisor."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), temperature=0)


def supervisor_route(state: TeamState, config: RunnableConfig) -> TeamState:
    """
    ROUTE node: Supervisor decides which specialist to call next.

    Analyzes completed work and determines the best next action.
    Forces FINISH if max iterations exceeded.
    """
    logger.info(f"[supervisor:route] Iteration {state['iteration']}")

    # Force finish if max iterations reached
    if state["iteration"] >= MAX_SUPERVISOR_ITERATIONS:
        logger.warning(f"[supervisor:route] Max iterations reached. Forcing FINISH.")
        return {**state, "next_worker": "FINISH"}

    worker_outputs = state.get("worker_outputs", [])

    # Build summary of work done
    outputs_summary = "None yet."
    if worker_outputs:
        outputs_summary = "\n".join(
            f"- {o['worker']}: {len(o['content'])} chars of output"
            for o in worker_outputs
        )

    prompt = SUPERVISOR_ROUTE_PROMPT.format(
        task=state["task"],
        num_outputs=len(worker_outputs),
        outputs_summary=outputs_summary,
        iteration=state["iteration"],
        max_iterations=MAX_SUPERVISOR_ITERATIONS,
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content="You are a team supervisor. Respond with only the worker name."),
        HumanMessage(content=prompt),
    ], config)

    # Parse the routing decision (simple text parsing for robustness)
    decision_text = response.content.strip().lower()
    next_worker: WorkerName = "FINISH"

    for worker in ["researcher", "coder", "critic"]:
        if worker in decision_text:
            next_worker = worker  # type: ignore
            break
    else:
        if "finish" in decision_text:
            next_worker = "FINISH"

    logger.info(f"[supervisor:route] Routing to: {next_worker}")

    return {
        **state,
        "messages": [response],
        "next_worker": next_worker,
        "iteration": state["iteration"] + 1,
    }


def supervisor_synthesize(state: TeamState, config: RunnableConfig) -> TeamState:
    """
    SYNTHESIZE node: Supervisor creates the final unified response.

    Integrates all specialist outputs into a coherent final answer.
    """
    logger.info("[supervisor:synthesize] Creating final synthesis")

    worker_outputs = state.get("worker_outputs", [])

    all_outputs = ""
    for output in worker_outputs:
        all_outputs += f"\n{'='*50}\n{output['worker'].upper()} SPECIALIST:\n{'='*50}\n"
        all_outputs += output["content"] + "\n"

    if not all_outputs:
        all_outputs = "No specialist work was completed."

    prompt = SUPERVISOR_SYNTHESIZE_PROMPT.format(
        task=state["task"],
        all_outputs=all_outputs,
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content="You are synthesizing a final response for the user. Be comprehensive and well-structured."),
        HumanMessage(content=prompt),
    ], config)

    final_response = response.content
    logger.info(f"[supervisor:synthesize] Synthesis complete. Length: {len(final_response)} chars")

    return {
        **state,
        "messages": [response],
        "final_response": final_response,
        "next_worker": None,
    }


def route_to_worker(state: TeamState) -> Literal["researcher", "coder", "critic", "synthesize"]:
    """
    Conditional edge: Route to the appropriate worker or synthesis step.
    """
    next_worker = state.get("next_worker")

    if next_worker == "FINISH" or next_worker is None:
        return "synthesize"

    return next_worker  # type: ignore
