"""
Research Specialist Worker.

Retrieves and summarizes information relevant to the task.
"""

import logging
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from state import TeamState, WorkerOutput

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a Research Specialist with expertise in information retrieval and synthesis.

Your role:
1. Analyze the research task given to you
2. Identify key concepts, facts, and relevant information
3. Provide a comprehensive, well-structured research summary
4. Cite your reasoning and note any gaps in knowledge
5. Be thorough but concise — focus on what's most relevant

Output format:
- Start with a brief summary (2-3 sentences)
- List key findings with evidence
- Note any uncertainties or areas needing further investigation
- End with recommendations for the next steps
"""


def _get_llm():
    """Get configured LLM for the researcher worker."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), temperature=0)


def researcher_node(state: TeamState, config: RunnableConfig) -> TeamState:
    """
    Research specialist node.

    Takes the current task and any previous worker outputs,
    produces a research summary, and appends it to worker_outputs.
    """
    logger.info("[researcher] Starting research task")

    task = state["task"]
    previous_outputs = state.get("worker_outputs", [])

    # Build context from previous outputs
    context = ""
    if previous_outputs:
        context = "

Previous work done by other specialists:
"
        for output in previous_outputs:
            context += f"
[{output['worker'].upper()}]:
{output['content']}
"

    prompt = f"""Research Task: {task}
{context}

Please provide comprehensive research on this topic. Focus on:
- Core concepts and definitions
- Current state of knowledge
- Key facts and data points
- Relevant examples or case studies
- Important nuances or considerations
"""

    llm = _get_llm()
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages, config)
    content = response.content

    logger.info(f"[researcher] Research complete. Output length: {len(content)} chars")

    worker_output: WorkerOutput = {
        "worker": "researcher",
        "content": content,
        "metadata": {"task": task, "iteration": state["iteration"]},
    }

    return {
        **state,
        "messages": [response],
        "worker_outputs": state.get("worker_outputs", []) + [worker_output],
    }
