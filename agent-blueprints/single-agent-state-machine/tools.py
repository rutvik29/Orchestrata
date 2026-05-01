"""
Tool definitions for the ReAct agent.
Each tool is decorated with @tool for LangChain compatibility.
"""

import logging
import math
import re
from datetime import datetime, timezone
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def search_web(query: str) -> str:
    """
    Search the web for information about a given query.

    Args:
        query: The search query string.

    Returns:
        A string containing simulated search results relevant to the query.
    """
    logger.info(f"[search_web] Query: {query}")

    # Production stub: replace with real search API (Tavily, SerpAPI, etc.)
    # Example: from tavily import TavilyClient; client = TavilyClient(api_key=...); return client.search(query)
    stub_results = {
        "python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms.",
        "langchain": "LangChain is a framework for developing applications powered by large language models (LLMs). It provides tools for chaining LLM calls, memory, agents, and retrieval.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It uses a graph-based approach for agent orchestration.",
        "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence processes by computer systems, including learning, reasoning, and self-correction.",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed.",
    }

    query_lower = query.lower()
    for keyword, result in stub_results.items():
        if keyword in query_lower:
            return f"Search results for '{query}':\n{result}\n\n[Note: This is a stub. Integrate a real search API for production use.]"

    return (
        f"Search results for '{query}':\n"
        f"No specific results found. In production, integrate Tavily, SerpAPI, or Brave Search.\n"
        f"[Stub response - replace with real search implementation]"
    )


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, sqrt(), log(), sin(), cos(), tan(), abs(), round(), pi, e

    Args:
        expression: A mathematical expression string, e.g. '2 ** 10 + sqrt(144)'

    Returns:
        The result of the calculation as a string, or an error message.
    """
    logger.info(f"[calculate] Expression: {expression}")

    # Safe math namespace
    safe_namespace = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
        "factorial": math.factorial,
        "pow": math.pow,
        "exp": math.exp,
    }

    # Sanitize: only allow safe characters
    allowed_pattern = re.compile(r"^[0-9\s\+\-\*\/\(\)\.\,\_a-zA-Z\*\*]+$")
    if not allowed_pattern.match(expression.strip()):
        return f"Error: Expression contains invalid characters. Only numeric expressions and math functions are allowed."

    try:
        result = eval(expression, safe_namespace)  # noqa: S307
        return f"Result of '{expression}' = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except ValueError as e:
        return f"Error: Invalid operation — {e}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


@tool
def get_current_datetime(timezone_name: Optional[str] = "UTC") -> str:
    """
    Get the current date and time.

    Args:
        timezone_name: Timezone name. Currently supports 'UTC' only.
                       For other timezones, integrate the 'pytz' or 'zoneinfo' library.

    Returns:
        A formatted string with the current date, time, day of week, and UTC offset.
    """
    logger.info(f"[get_current_datetime] Timezone: {timezone_name}")

    now_utc = datetime.now(timezone.utc)

    return (
        f"Current DateTime Information:\n"
        f"  Date:        {now_utc.strftime('%Y-%m-%d')}\n"
        f"  Time (UTC):  {now_utc.strftime('%H:%M:%S')}\n"
        f"  Day:         {now_utc.strftime('%A')}\n"
        f"  ISO 8601:    {now_utc.isoformat()}\n"
        f"  Unix Epoch:  {int(now_utc.timestamp())}\n"
        f"  Week Number: {now_utc.isocalendar()[1]}\n"
        f"  Timezone:    UTC+00:00"
    )


# Export all tools as a list for easy import
ALL_TOOLS = [search_web, calculate, get_current_datetime]
