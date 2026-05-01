"""
ReAct (Reason + Act) Agent implemented as a LangGraph state machine.

Architecture:
  [think] → [act] → [observe] → [should_continue?] → [think] | END

Supports both Anthropic and OpenAI via LLM_PROVIDER env var.
Includes streaming output support.
"""

import logging
import os
from typing import Annotated, Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from prompts import get_full_prompt
from tools import ALL_TOOLS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the ReAct graph.

    Fields:
        messages: Full conversation history (auto-merged by LangGraph).
        tool_calls: List of tool call records from the most recent act step.
        iteration_count: Number of think→act cycles completed (used for max iteration guard).
        final_answer: Set when the agent is done. None means still processing.
        error: Optional error message if something went wrong.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls: list[dict[str, Any]]
    iteration_count: int
    final_answer: Optional[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm():
    """
    Build the LLM based on LLM_PROVIDER environment variable.
    Supports 'anthropic' (default) and 'openai'.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        logger.info(f"Using OpenAI model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            streaming=True,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        from langchain_anthropic import ChatAnthropic
        model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        logger.info(f"Using Anthropic model: {model_name}")
        return ChatAnthropic(
            model=model_name,
            temperature=0,
            streaming=True,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))

# Build LLM with tools bound
_llm = _build_llm()
_llm_with_tools = _llm.bind_tools(ALL_TOOLS)

# Tool executor map
_tool_map = {t.name: t for t in ALL_TOOLS}


def think(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    THINK node: The LLM reasons about the current state and decides what to do next.
    Either calls a tool (act) or produces a final answer.
    """
    logger.info(f"[think] Iteration {state['iteration_count']}, messages: {len(state['messages'])}")

    # Inject system prompt if not already present
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(content=get_full_prompt(max_iterations=MAX_ITERATIONS))
        messages = [system_msg] + list(messages)

    response = _llm_with_tools.invoke(messages, config)
    logger.info(f"[think] LLM response type: {type(response).__name__}, tool_calls: {len(getattr(response, 'tool_calls', []))}")

    return {
        "messages": [response],
        "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else [],
        "iteration_count": state["iteration_count"],
        "final_answer": None,
        "error": None,
    }


def act(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    ACT node: Execute all pending tool calls and collect results.
    """
    tool_calls = state.get("tool_calls", [])
    logger.info(f"[act] Executing {len(tool_calls)} tool call(s)")

    tool_messages = []
    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_call_id = tc.get("id", "unknown")

        logger.info(f"[act] Calling tool '{tool_name}' with args: {tool_args}")

        if tool_name not in _tool_map:
            result = f"Error: Unknown tool '{tool_name}'. Available tools: {list(_tool_map.keys())}"
        else:
            try:
                result = _tool_map[tool_name].invoke(tool_args, config)
                result = str(result)
            except Exception as e:
                logger.exception(f"[act] Tool '{tool_name}' raised exception")
                result = f"Error executing tool '{tool_name}': {e}"

        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )
        logger.info(f"[act] Tool '{tool_name}' result: {result[:200]}...")

    return {
        "messages": tool_messages,
        "tool_calls": [],
        "iteration_count": state["iteration_count"] + 1,
        "final_answer": None,
        "error": None,
    }


def observe(state: AgentState) -> AgentState:
    """
    OBSERVE node: Summarize what happened in this iteration.
    Lightweight node — mostly a pass-through for LangGraph graph clarity.
    """
    logger.info(f"[observe] Completed iteration {state['iteration_count']}")
    return state


def should_continue(state: AgentState) -> Literal["think", "end"]:
    """
    Conditional edge: Decide whether to continue the ReAct loop or stop.

    Returns 'end' if:
    - The last AI message has no tool calls (final answer given)
    - Max iterations reached

    Returns 'think' to continue the loop.
    """
    iteration = state["iteration_count"]

    if iteration >= MAX_ITERATIONS:
        logger.warning(f"[should_continue] Max iterations ({MAX_ITERATIONS}) reached. Stopping.")
        return "end"

    messages = state.get("messages", [])
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    if not ai_messages:
        return "think"

    last_ai = ai_messages[-1]
    has_tool_calls = bool(getattr(last_ai, "tool_calls", []))

    if has_tool_calls:
        logger.info(f"[should_continue] Tool calls pending → continuing to act")
        return "think"
    else:
        logger.info(f"[should_continue] No tool calls → agent has final answer")
        return "end"


def build_graph() -> StateGraph:
    """
    Construct and compile the ReAct LangGraph state machine.
    """
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("observe", observe)
    graph.set_entry_point("think")
    graph.add_conditional_edges(
        "think",
        lambda state: "act" if state.get("tool_calls") else "end",
        {"act": "act", "end": END},
    )
    graph.add_edge("act", "observe")
    graph.add_conditional_edges(
        "observe",
        should_continue,
        {"think": "think", "end": END},
    )
    return graph.compile()


def run_agent(user_query: str, stream: bool = False) -> str:
    """
    Run the ReAct agent on a user query.
    """
    graph = build_graph()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "tool_calls": [],
        "iteration_count": 0,
        "final_answer": None,
        "error": None,
    }
    logger.info(f"[run_agent] Starting agent for query: '{user_query}'")
    if stream:
        print("\n--- Agent Streaming Output ---\n")
        final_state = None
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                if node_name == "think":
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content:
                            print(f"[{node_name}] {msg.content}", flush=True)
                elif node_name == "act":
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            print(f"[tool:{msg.name}] {msg.content[:300]}", flush=True)
            final_state = node_output
    else:
        final_state = graph.invoke(initial_state)
    all_messages = final_state.get("messages", [])
    ai_messages = [m for m in all_messages if isinstance(m, AIMessage)]
    if ai_messages:
        return ai_messages[-1].content
    return "No response generated."


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 2^10 + sqrt(144)? Also what day of the week is today?"
    print(f"\nQuery: {query}\n")
    result = run_agent(query, stream=True)
    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(result)
    print("=" * 60)
