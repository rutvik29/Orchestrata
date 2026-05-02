"""
Sandboxed file operations tool.

All read/write operations are restricted to a configurable sandbox directory.
Attempts to escape via path traversal (../) are blocked.
"""

import logging
import os
from pathlib import Path
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Sandbox directory: defaults to ./sandbox relative to this file's location
_SANDBOX_DIR = Path(os.getenv("FILE_OPS_SANDBOX", os.path.join(os.path.dirname(__file__), "..", "sandbox"))).resolve()


def _get_sandbox() -> Path:
    """Get and ensure the sandbox directory exists."""
    _SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    return _SANDBOX_DIR


def _safe_path(filename: str) -> Path:
    """
    Resolve a filename to a safe path within the sandbox.

    Raises:
        PermissionError: If the resolved path escapes the sandbox directory.
        ValueError: If the filename is empty or contains null bytes.
    """
    if not filename or "\x00" in filename:
        raise ValueError(f"Invalid filename: {repr(filename)}")

    sandbox = _get_sandbox()
    safe_name = filename.lstrip("/\\").replace("..", "_")
    resolved = (sandbox / safe_name).resolve()

    try:
        resolved.relative_to(sandbox)
    except ValueError:
        raise PermissionError(
            f"Path '{filename}' resolves outside the sandbox. "
            f"Access is restricted to: {sandbox}"
        )

    return resolved


@tool
def read_file(path: str) -> str:
    """
    Read the contents of a file from the sandboxed workspace directory.

    Args:
        path: Relative file path within the sandbox (e.g., 'notes.txt', 'data/report.md').
              Absolute paths and path traversal (../) are blocked for security.

    Returns:
        The file contents as a string, or an error message if not found.
    """
    logger.info(f"[read_file] Reading: {path}")

    try:
        safe = _safe_path(path)

        if not safe.exists():
            sandbox = _get_sandbox()
            available = [str(f.relative_to(sandbox)) for f in sandbox.rglob("*") if f.is_file()]
            avail_text = "\n  ".join(available) if available else "(sandbox is empty)"
            return (
                f"Error: File '{path}' not found in sandbox.\n"
                f"Available files:\n  {avail_text}"
            )

        if not safe.is_file():
            return f"Error: '{path}' is a directory, not a file."

        size = safe.stat().st_size
        if size > 1_048_576:
            return f"Error: File '{path}' is too large ({size:,} bytes). Maximum is 1MB."

        content = safe.read_text(encoding="utf-8", errors="replace")
        logger.info(f"[read_file] Read {len(content)} chars from {safe}")
        return f"Contents of '{path}' ({len(content)} chars):\n\n{content}"

    except PermissionError as e:
        return f"Security Error: {e}"
    except ValueError as e:
        return f"Invalid path: {e}"
    except Exception as e:
        logger.exception(f"[read_file] Error reading '{path}'")
        return f"Error reading file '{path}': {e}"


@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a file in the sandboxed workspace directory.

    Creates parent directories as needed. Overwrites existing files.

    Args:
        path: Relative file path within the sandbox (e.g., 'output.txt', 'results/data.json').
              Absolute paths and path traversal (../) are blocked for security.
        content: Text content to write to the file.

    Returns:
        A success message with file location and size, or an error message.
    """
    logger.info(f"[write_file] Writing {len(content)} chars to: {path}")

    try:
        safe = _safe_path(path)
        safe.parent.mkdir(parents=True, exist_ok=True)

        if len(content.encode("utf-8")) > 5_242_880:
            return f"Error: Content is too large. Maximum write size is 5MB."

        safe.write_text(content, encoding="utf-8")
        size = safe.stat().st_size

        logger.info(f"[write_file] Wrote {size} bytes to {safe}")
        return (
            f"Successfully wrote {size:,} bytes to '{path}'\n"
            f"Full path: {safe}"
        )

    except PermissionError as e:
        return f"Security Error: {e}"
    except ValueError as e:
        return f"Invalid path: {e}"
    except OSError as e:
        return f"OS Error writing '{path}': {e}"
    except Exception as e:
        logger.exception(f"[write_file] Error writing '{path}'")
        return f"Error writing file '{path}': {e}"


@tool
def list_sandbox_files(subdirectory: str = "") -> str:
    """
    List all files in the sandboxed workspace directory.

    Args:
        subdirectory: Optional subdirectory within the sandbox to list (default: root).

    Returns:
        Formatted list of files with sizes, or an error message.
    """
    logger.info(f"[list_sandbox_files] Listing: {subdirectory or '(root)'}")

    try:
        sandbox = _get_sandbox()
        if subdirectory:
            target = _safe_path(subdirectory)
        else:
            target = sandbox

        if not target.exists():
            return f"Directory '{subdirectory}' does not exist in sandbox."

        files = sorted(target.rglob("*"))
        if not files:
            return f"Sandbox is empty: {sandbox}"

        lines = [f"Sandbox contents ({sandbox}):"]
        for f in files:
            if f.is_file():
                rel = f.relative_to(sandbox)
                size = f.stat().st_size
                lines.append(f"  {rel} ({size:,} bytes)")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing sandbox: {e}"


ALL_FILE_TOOLS = [read_file, write_file, list_sandbox_files]
