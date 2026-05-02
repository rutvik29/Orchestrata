"""
Code Specialist Worker.

Writes, explains, and analyzes Python code for the given task.
"""

import logging
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from state import TeamState, WorkerOutput

logger = logging.getLogger(__name__)

CODER_SYSTEM_PROMPT = """You are a Code Specialist with deep expertise in Python and software engineering best practices.

Your role:
1. Write clean, production-quality Python code for the task
2. Include type hints, docstrings, and inline comments
3. Handle errors gracefully with proper exception handling
4. Follow PEP 8 style guidelines
5. Explain your implementation choices

Output format:
- Brief explanation of the approach (2-3 sentences)
- Complete, runnable Python code in a code block
- Step-by-step explanation of how the code works
- Example usage with expected output
- Any dependencies or installation notes
"""


def _get_llm():
    """Get configured LLM for the coder worker."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), temperature=0)


def coder_node(state: TeamState, config: RunnableConfig) -> TeamState:
    """
    Code specialist node.

    Analyzes the task and any prior research, then produces
    a working Python implementation with full explanation.
    """
    logger.info("[coder] Starting code generation task")

    task = state["task"]
    previous_outputs = state.get("worker_outputs", [])

    # Build context from previous outputs
    context = ""
    if previous_outputs:
        context = "

Context from other specialists:
"
        for output in previous_outputs:
            if output["worker"] == "researcher":
                context += f"
RESEARCH BACKGROUND:
{output['content']}
"

    prompt = f"""Coding Task: {task}
{context}

Please provide a complete Python implementation. Requirements:
- Use type hints throughout
- Include comprehensive docstrings
- Handle edge cases and errors
- Write clean, maintainable code
- Provide example usage

If the task is conceptual (not directly codeable), create a relevant Python demonstration or utility that illustrates the concept.
"""

    llm = _get_llm()
    messages = [
        SystemMessage(content=CODER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages, config)
    content = response.content

    logger.info(f"[coder] Code generation complete. Output length: {len(content)} chars")

    worker_output: WorkerOutput = {
        "worker": "coder",
        "content": content,
        "metadata": {"task": task, "iteration": state["iteration"]},
    }

    return {
        **state,
        "messages": [response],
        "worker_outputs": state.get("worker_outputs", []) + [worker_output],
    }
