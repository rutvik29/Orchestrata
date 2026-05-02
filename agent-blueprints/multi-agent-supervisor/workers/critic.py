"""
Critic Specialist Worker.

Reviews outputs from other specialists for accuracy, completeness, and quality.
"""

import logging
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from state import TeamState, WorkerOutput

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a rigorous Critic and Quality Assurance Specialist.

Your role:
1. Review all previous work from other specialists
2. Identify inaccuracies, gaps, or weaknesses
3. Verify factual claims and logical consistency
4. Assess completeness relative to the original task
5. Suggest specific improvements

Output format:
**Overall Assessment**: [PASS / NEEDS_IMPROVEMENT / FAIL] with brief justification

**Strengths**:
- List what was done well

**Issues Found**:
- List specific problems with severity (HIGH/MEDIUM/LOW)

**Missing Elements**:
- What's absent that should be present

**Recommended Improvements**:
- Specific, actionable suggestions

**Quality Score**: X/10 with reasoning

Be constructive but honest. Your goal is to improve the final output quality.
"""


def _get_llm():
    """Get configured LLM for the critic worker."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), temperature=0)


def critic_node(state: TeamState, config: RunnableConfig) -> TeamState:
    """
    Critic specialist node.

    Reviews all prior specialist outputs and produces a quality assessment
    with specific recommendations for improvement.
    """
    logger.info("[critic] Starting quality review")

    task = state["task"]
    previous_outputs = state.get("worker_outputs", [])

    if not previous_outputs:
        logger.warning("[critic] No previous outputs to review!")
        content = "No work to review yet. Please run researcher or coder first."
    else:
        # Build the review request
        outputs_text = ""
        for output in previous_outputs:
            outputs_text += f"
{'='*40}
SPECIALIST: {output['worker'].upper()}
{'='*40}
"
            outputs_text += output["content"] + "
"

        prompt = f"""Original Task: {task}

Work to Review:
{outputs_text}

Please provide a thorough critical review of all the above work relative to the original task.
Evaluate accuracy, completeness, quality, and usefulness of each piece of work.
"""

        llm = _get_llm()
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages, config)
        content = response.content

    logger.info(f"[critic] Review complete. Output length: {len(content)} chars")

    worker_output: WorkerOutput = {
        "worker": "critic",
        "content": content,
        "metadata": {"task": task, "iteration": state["iteration"], "num_reviewed": len(previous_outputs)},
    }

    return {
        **state,
        "messages": [response] if previous_outputs else [],
        "worker_outputs": state.get("worker_outputs", []) + [worker_output],
    }
