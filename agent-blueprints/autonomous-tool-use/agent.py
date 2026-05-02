"""
Autonomous Plan-and-Execute Agent.

Architecture:
  planner -> executor -> replanner -> [continue | finish]
                  ^_________________________|

The agent decomposes a goal into steps, executes them one at a time using
the available tools, and dynamically replans based on execution results.

Supports: weather lookups, math calculations, Wikipedia searches, file I/O.
"""

import logging
import os
import sys
from typing import Annotated, Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from planner import ExecutionResult, Plan, ReplanDecision
from tools import ALL_TOOLS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlanExecuteState(TypedDict):
    """Shared state for the Plan-and-Execute agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    goal: str
    plan: list[str]                   # Remaining steps to execute
    step_index: int                    # Current step number (1-based)
    execution_results: list[dict]      # All completed step results
    current_step: Optional[str]        # Step currently being executed
    final_answer: Optional[str]
    total_steps_taken: int


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), temperature=0)


_tool_map = {t.name: t for t in ALL_TOOLS}
_llm_with_tools = _get_llm().bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Node prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are a strategic planner for an autonomous agent.

Given a user's goal, decompose it into a clear, ordered list of concrete steps.
Each step should be:
1. Specific and actionable
2. Executable using one of the available tools, OR a reasoning/synthesis step
3. Building logically on previous steps

Available tools:
- get_weather(city): Get current weather for any city
- calculate(expr): Evaluate math expressions (supports sqrt, log, sin, cos, pi, e, **)
- search_wikipedia(query): Search Wikipedia for information
- read_file(path): Read a file from the sandbox
- write_file(path, content): Write content to a file in the sandbox
- list_sandbox_files(): List all files in the sandbox

Be practical: 3-8 steps is ideal. Don't over-plan.
"""

EXECUTOR_SYSTEM = """You are a precise executor working through a plan step by step.

Your job: Execute the CURRENT STEP using the most appropriate tool.
- Choose the right tool for the step
- Pass well-formed arguments to the tool
- If no tool is needed (reasoning/synthesis), provide the output directly
- Be thorough but efficient
"""

REPLANNER_SYSTEM = """You are a dynamic replanner for an autonomous agent.

Given the original goal, the original plan, and the execution results so far,
decide what to do next:

1. 'finish' -- All goals have been met. Provide a comprehensive final answer.
2. 'continue' -- The remaining steps are still valid. Continue with them.
3. 'revise' -- The results suggest the remaining steps should be updated.

Be pragmatic: if the goal is achieved, finish. If results reveal a better path, revise.
"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner(state: PlanExecuteState, config: RunnableConfig) -> PlanExecuteState:
    """PLANNER node: Decompose the user's goal into an ordered list of steps."""
    logger.info(f"[planner] Planning for goal: '{state['goal']}'")

    llm = _get_llm().with_structured_output(Plan)

    response: Plan = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Goal: {state['goal']}\n\nCreate a step-by-step plan to accomplish this goal."),
    ], config)

    logger.info(f"[planner] Created {len(response.steps)}-step plan: {response.steps}")

    return {
        **state,
        "plan": response.steps,
        "step_index": 1,
        "messages": [AIMessage(content=f"Plan created ({len(response.steps)} steps):\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(response.steps)))],
    }


def executor(state: PlanExecuteState, config: RunnableConfig) -> PlanExecuteState:
    """
    EXECUTOR node: Execute the current step using the appropriate tool.
    The LLM decides which tool to call; we execute it and record the result.
    """
    if not state["plan"]:
        logger.warning("[executor] No steps remaining in plan")
        return state

    current_step = state["plan"][0]
    step_number = state["step_index"]

    logger.info(f"[executor] Executing step {step_number}: '{current_step}'")

    prior_context = ""
    if state["execution_results"]:
        prior_context = "\n\nCompleted steps:\n"
        for r in state["execution_results"]:
            prior_context += f"  Step {r['step_number']}: {r['output'][:200]}\n"

    prompt = (
        f"Goal: {state['goal']}\n"
        f"{prior_context}\n"
        f"Current step to execute (Step {step_number}): {current_step}\n\n"
        f"Execute this step using the most appropriate tool. "
        f"If no tool is needed, provide the result directly."
    )

    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM),
        HumanMessage(content=prompt),
    ]

    response = _llm_with_tools.invoke(messages, config)

    tool_output = ""
    tool_name = None
    tool_messages = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            t_name = tc.get("name", "")
            t_args = tc.get("args", {})
            t_id = tc.get("id", "unknown")
            tool_name = t_name

            logger.info(f"[executor] Tool call: {t_name}({t_args})")

            if t_name in _tool_map:
                try:
                    result = _tool_map[t_name].invoke(t_args, config)
                    tool_output = str(result)
                except Exception as e:
                    tool_output = f"Error: {e}"
                    logger.exception(f"[executor] Tool '{t_name}' failed")
            else:
                tool_output = f"Unknown tool: {t_name}"

            tool_messages.append(ToolMessage(content=tool_output, tool_call_id=t_id, name=t_name))
    else:
        tool_output = response.content

    exec_result = {
        "step_number": step_number,
        "step_description": current_step,
        "tool_used": tool_name,
        "output": tool_output,
        "success": "Error" not in tool_output[:20],
    }

    logger.info(f"[executor] Step {step_number} result: {tool_output[:200]}")

    return {
        **state,
        "messages": [response] + tool_messages,
        "current_step": current_step,
        "plan": state["plan"][1:],
        "step_index": step_number + 1,
        "execution_results": state["execution_results"] + [exec_result],
        "total_steps_taken": state["total_steps_taken"] + 1,
    }


def replanner(state: PlanExecuteState, config: RunnableConfig) -> PlanExecuteState:
    """REPLANNER node: Assess progress and decide to continue, revise, or finish."""
    logger.info(f"[replanner] Assessing progress after step {state['step_index'] - 1}")

    if state["total_steps_taken"] >= MAX_STEPS:
        logger.warning(f"[replanner] Max steps ({MAX_STEPS}) reached -- forcing finish")
        final = _synthesize_final_answer(state, config)
        return {**state, "final_answer": final}

    results_text = "\n".join(
        f"Step {r['step_number']} [{r.get('tool_used', 'reasoning')}]: {r['output'][:300]}"
        for r in state["execution_results"]
    )
    remaining_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["plan"])) or "None"

    prompt = (
        f"Original Goal: {state['goal']}\n\n"
        f"Execution Results So Far:\n{results_text}\n\n"
        f"Remaining Plan Steps:\n{remaining_text}\n\n"
        f"Steps taken: {state['total_steps_taken']}/{MAX_STEPS}\n\n"
        f"Decide: should I 'finish' (goal achieved), 'continue' (proceed), or 'revise' (update remaining steps)?"
    )

    llm = _get_llm().with_structured_output(ReplanDecision)
    decision: ReplanDecision = llm.invoke([
        SystemMessage(content=REPLANNER_SYSTEM),
        HumanMessage(content=prompt),
    ], config)

    logger.info(f"[replanner] Decision: {decision.action} -- {decision.reasoning[:100]}")

    if decision.action == "finish":
        final = decision.final_answer or _synthesize_final_answer(state, config)
        return {**state, "final_answer": final, "plan": []}

    if decision.action == "revise" and decision.revised_steps:
        logger.info(f"[replanner] Revised plan: {decision.revised_steps}")
        return {**state, "plan": decision.revised_steps}

    return state  # "continue" -- unchanged


def _synthesize_final_answer(state: PlanExecuteState, config: RunnableConfig) -> str:
    """Generate a synthesized final answer from all execution results."""
    results_text = "\n".join(
        f"Step {r['step_number']}: {r['output'][:500]}"
        for r in state["execution_results"]
    )
    prompt = (
        f"Goal: {state['goal']}\n\n"
        f"All execution results:\n{results_text}\n\n"
        f"Provide a comprehensive final answer to the goal, synthesizing all the information gathered."
    )
    response = _get_llm().invoke([HumanMessage(content=prompt)], config)
    return response.content


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_continue(state: PlanExecuteState) -> Literal["executor", "end"]:
    """Route: continue executing if plan has steps and no final answer yet."""
    if state.get("final_answer"):
        return "end"
    if not state.get("plan"):
        return "end"
    if state.get("total_steps_taken", 0) >= MAX_STEPS:
        return "end"
    return "executor"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """
    Build the Plan-and-Execute LangGraph agent.

    planner -> executor -> replanner -> [executor (loop) | END]
    """
    graph = StateGraph(PlanExecuteState)

    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("replanner", replanner)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges(
        "replanner",
        should_continue,
        {"executor": "executor", "end": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(goal: str) -> dict[str, Any]:
    """
    Run the Plan-and-Execute agent on a goal.

    Args:
        goal: The user's high-level goal or multi-step task.

    Returns:
        Dict with 'answer', 'steps_taken', and 'execution_results'.
    """
    graph = build_agent_graph()

    initial: PlanExecuteState = {
        "messages": [HumanMessage(content=goal)],
        "goal": goal,
        "plan": [],
        "step_index": 1,
        "execution_results": [],
        "current_step": None,
        "final_answer": None,
        "total_steps_taken": 0,
    }

    logger.info(f"[run_agent] Starting for goal: '{goal}'")
    final_state = graph.invoke(initial)

    return {
        "answer": final_state.get("final_answer", "No answer generated."),
        "steps_taken": final_state.get("total_steps_taken", 0),
        "execution_results": final_state.get("execution_results", []),
    }


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "What is the weather in Tokyo today, and what is the square root of the temperature in Celsius squared plus 100?"
    )

    print(f"\nGoal: {goal}\n{'='*60}")
    result = run_agent(goal)
    print(f"\nFINAL ANSWER:\n{result['answer']}")
    print(f"\nSteps taken: {result['steps_taken']}")
    for r in result["execution_results"]:
        print(f"  Step {r['step_number']} [{r.get('tool_used', '-')}]: {r['output'][:150]}")
