"""
Calculator tool using numexpr for safe, fast mathematical expression evaluation.

Falls back to Python's math module if numexpr is not installed.
"""

import logging
import re
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _safe_eval_fallback(expression: str) -> float:
    """Fallback evaluator using Python math (no numexpr)."""
    import math
    safe_ns = {
        "__builtins__": {},
        "abs": abs, "round": round,
        "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
        "pi": math.pi, "e": math.e,
        "ceil": math.ceil, "floor": math.floor,
        "exp": math.exp, "pow": math.pow,
        "factorial": math.factorial,
        "inf": math.inf, "nan": math.nan,
    }
    return eval(expression, safe_ns)  # noqa: S307


@tool
def calculate(expr: str) -> str:
    """
    Evaluate a mathematical expression using numexpr for safe, high-performance computation.

    Supports: arithmetic (+, -, *, /, **, %), comparison operators,
    functions (sin, cos, tan, sqrt, log, exp, abs, ceil, floor),
    and constants (pi, e).

    Falls back to Python math if numexpr is not installed.

    Args:
        expr: Mathematical expression string.
              Examples:
                - "2 ** 32"
                - "sqrt(2) * pi"
                - "log(1000) / log(10)"
                - "(5 + 3) * (12 - 4) / 2"
                - "sin(pi/6)"

    Returns:
        Formatted string with the result, or an error message.
    """
    logger.info(f"[calculate] Expression: {expr}")

    # Basic sanitization: reject obviously dangerous patterns
    dangerous_patterns = [
        r"import\s+", r"__\w+__", r"exec\s*\(", r"eval\s*\(",
        r"open\s*\(", r"os\.", r"sys\.", r"subprocess",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, expr, re.IGNORECASE):
            return f"Error: Expression contains disallowed content: '{pattern}'"

    # Try numexpr first (faster, more secure)
    try:
        import numexpr as ne
        import numpy as np
        result = ne.evaluate(expr)
        if hasattr(result, "item"):
            result = result.item()
        formatted = f"{result:.10g}" if isinstance(result, float) else str(result)
        return f"Result: {expr} = {formatted}"
    except ImportError:
        logger.warning("[calculate] numexpr not installed, falling back to Python math")
    except Exception as e:
        logger.debug(f"[calculate] numexpr failed: {e}, trying fallback")

    # Fallback to Python math
    try:
        result = _safe_eval_fallback(expr)
        formatted = f"{result:.10g}" if isinstance(result, float) else str(result)
        return f"Result: {expr} = {formatted}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except ValueError as e:
        return f"Error: Math domain error -- {e}"
    except SyntaxError:
        return f"Error: Invalid expression syntax in '{expr}'"
    except NameError as e:
        return f"Error: Unknown name in expression -- {e}. Supported: sqrt, log, sin, cos, tan, exp, pi, e"
    except Exception as e:
        return f"Error evaluating '{expr}': {e}"
