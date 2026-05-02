"""
Orchestrator: Wires the supervisor and specialist workers into a LangGraph multi-agent system.

Graph topology:
    supervisor_route → [researcher | coder | critic | synthesize]
        ↑                          ↓
        └──────────────────────────┘ (loop back to supervisor)
    synthesize → END
"""

import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from state import TeamState
from supervisor import route_to_worker, supervisor_route, supervisor_synthesize
from workers.researcher import researcher_node
from workers.coder import coder_node
from workers.critic import critic_node

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_team_graph() -> StateGraph:
    """
    Build and compile the multi-agent supervisor graph.

    Nodes:
        - supervisor_route: Decides which specialist to call
        - researcher: Research specialist
        - coder: Code specialist
        - critic: Quality critic
        - supervisor_synthesize: Creates final response

    Edges:
        - supervisor_route → (researcher | coder | critic | synthesize) [conditional]
        - researcher → supervisor_route
        - coder → supervisor_route
        - critic → supervisor_route
        - supervisor_synthesize → END
    """
    graph = StateGraph(TeamState)

    # Register all nodes
    graph.add_node("supervisor_route", supervisor_route)
    graph.add_node("researcher", researcher_node)
    graph.add_node("coder", coder_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesize", supervisor_synthesize)

    # Entry: always start at supervisor
    graph.set_entry_point("supervisor_route")

    # Supervisor routes to workers or synthesis
    graph.add_conditional_edges(
        "supervisor_route",
        route_to_worker,
        {
            "researcher": "researcher",
            "coder": "coder",
            "critic": "critic",
            "synthesize": "synthesize",
        },
    )

    # Workers all loop back to supervisor for next decision
    graph.add_edge("researcher", "supervisor_route")
    graph.add_edge("coder", "supervisor_route")
    graph.add_edge("critic", "supervisor_route")

    # Synthesis is terminal
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_team(task: str) -> str:
    """
    Run the multi-agent team on a given task.

    Args:
        task: The user's task or question.

    Returns:
        The synthesized final response.
    """
    graph = build_team_graph()

    initial_state: TeamState = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "next_worker": None,
        "worker_outputs": [],
        "current_plan": [],
        "iteration": 0,
        "final_response": None,
        "error": None,
    }

    logger.info(f"[orchestrator] Starting multi-agent team for task: '{task}'")

    final_state = graph.invoke(initial_state)

    return final_state.get("final_response", "No response generated.")


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Explain transformer attention mechanisms and provide a Python implementation of scaled dot-product attention."
    )

    print(f"\nTask: {task}\n{'='*60}")
    result = run_team(task)
    print(f"\nFINAL RESPONSE:\n{'='*60}")
    print(result)
    print("=" * 60)
