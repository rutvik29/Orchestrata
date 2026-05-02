"""Tool registry for the autonomous-tool-use agent."""
from tools.weather import get_weather
from tools.calculator import calculate
from tools.wikipedia import search_wikipedia
from tools.file_ops import read_file, write_file, list_sandbox_files

ALL_TOOLS = [get_weather, calculate, search_wikipedia, read_file, write_file, list_sandbox_files]

__all__ = ["get_weather", "calculate", "search_wikipedia", "read_file", "write_file", "list_sandbox_files", "ALL_TOOLS"]
