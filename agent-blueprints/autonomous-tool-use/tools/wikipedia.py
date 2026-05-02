"""
Wikipedia search tool using the wikipedia Python package.

Searches Wikipedia and returns structured article summaries.
"""

import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def search_wikipedia(query: str, sentences: int = 5) -> str:
    """
    Search Wikipedia and return a summary of the most relevant article.

    Args:
        query: Search query or article title to look up.
        sentences: Number of sentences to include in the summary (default: 5).

    Returns:
        A formatted string with the article title, URL, summary,
        and related page suggestions. Returns an error message if not found.
    """
    logger.info(f"[search_wikipedia] Query: '{query}', sentences: {sentences}")

    try:
        import wikipedia

        wikipedia.set_lang("en")

        try:
            page = wikipedia.page(query, auto_suggest=True, redirect=True)
            summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True)

            search_results = wikipedia.search(query, results=5)
            related = [r for r in search_results if r.lower() != page.title.lower()][:3]
            related_text = ", ".join(related) if related else "None"

            return (
                f"Wikipedia: {page.title}\n"
                f"URL: {page.url}\n\n"
                f"Summary:\n{summary}\n\n"
                f"Related articles: {related_text}"
            )

        except wikipedia.exceptions.DisambiguationError as e:
            logger.info(f"[search_wikipedia] Disambiguation for '{query}': {e.options[:5]}")
            if e.options:
                first_option = e.options[0]
                try:
                    page = wikipedia.page(first_option)
                    summary = wikipedia.summary(first_option, sentences=sentences)
                    return (
                        f"Wikipedia: {page.title} (disambiguation resolved)\n"
                        f"URL: {page.url}\n\n"
                        f"Summary:\n{summary}\n\n"
                        f"Other matches: {', '.join(e.options[1:4])}"
                    )
                except Exception:
                    pass
            options_text = "\n".join(f"  - {opt}" for opt in e.options[:8])
            return (
                f"Disambiguation: '{query}' could refer to multiple articles.\n"
                f"Please be more specific. Options include:\n{options_text}"
            )

        except wikipedia.exceptions.PageError:
            logger.info(f"[search_wikipedia] Page not found for '{query}', trying search")
            search_results = wikipedia.search(query, results=5)
            if search_results:
                try:
                    summary = wikipedia.summary(search_results[0], sentences=sentences)
                    page = wikipedia.page(search_results[0])
                    return (
                        f"Wikipedia: {page.title} (closest match for '{query}')\n"
                        f"URL: {page.url}\n\n"
                        f"Summary:\n{summary}\n\n"
                        f"Other suggestions: {', '.join(search_results[1:4])}"
                    )
                except Exception:
                    pass
            suggestions = ", ".join(search_results) if search_results else "none"
            return (
                f"No Wikipedia article found for '{query}'.\n"
                f"Suggestions: {suggestions}"
            )

    except ImportError:
        return (
            "Error: The 'wikipedia' package is not installed.\n"
            "Install it with: pip install wikipedia"
        )
    except Exception as e:
        logger.exception(f"[search_wikipedia] Unexpected error")
        return f"Error searching Wikipedia for '{query}': {e}"
