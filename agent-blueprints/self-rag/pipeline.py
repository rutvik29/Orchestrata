"""
Self-RAG Pipeline implemented as a LangGraph graph.

Flow:
  retrieve -> grade_documents -> [relevant?]
      YES -> generate -> grade_generation -> [grounded + useful?]
          YES -> END
          NO  -> transform_query -> retrieve (loop)
      NO  -> transform_query -> retrieve (loop)

Retries are capped at MAX_RETRIES to prevent infinite loops.
"""

import logging
import os
from typing import Annotated, Any, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from graders import AnswerGrader, HallucinationGrader, RelevanceGrader
from retriever import FAISSRetriever
from sample_docs import SAMPLE_DOCUMENTS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TOP_K = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    """Shared state for the Self-RAG pipeline."""
    question: str
    retrieved_docs: list[dict[str, Any]]
    relevant_docs: list[dict[str, Any]]
    generation: Optional[str]
    retry_count: int
    is_grounded: bool
    is_useful: bool
    final_answer: Optional[str]
    query_for_retrieval: str  # may be rewritten


# ---------------------------------------------------------------------------
# Singletons: retriever and graders
# ---------------------------------------------------------------------------

_retriever: Optional[FAISSRetriever] = None
_relevance_grader: Optional[RelevanceGrader] = None
_hallucination_grader: Optional[HallucinationGrader] = None
_answer_grader: Optional[AnswerGrader] = None


def _get_retriever() -> FAISSRetriever:
    global _retriever
    if _retriever is None:
        logger.info("Initializing FAISS retriever with sample documents...")
        _retriever = FAISSRetriever()
        _retriever.index_documents(SAMPLE_DOCUMENTS)
        logger.info(f"Retriever ready: {_retriever.num_documents} documents indexed")
    return _retriever


def _get_graders():
    global _relevance_grader, _hallucination_grader, _answer_grader
    if _relevance_grader is None:
        _relevance_grader = RelevanceGrader()
        _hallucination_grader = HallucinationGrader()
        _answer_grader = AnswerGrader()
    return _relevance_grader, _hallucination_grader, _answer_grader


def _get_llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), temperature=0)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> RAGState:
    """RETRIEVE node: Fetch top-k documents from the FAISS index."""
    query = state.get("query_for_retrieval") or state["question"]
    logger.info(f"[retrieve] Query: '{query[:80]}'")

    retriever = _get_retriever()
    docs = retriever.retrieve(query, k=TOP_K)

    logger.info(f"[retrieve] Retrieved {len(docs)} documents")
    return {**state, "retrieved_docs": docs}


def grade_documents(state: RAGState) -> RAGState:
    """GRADE_DOCUMENTS node: LLM grades each retrieved doc for relevance."""
    logger.info(f"[grade_documents] Grading {len(state['retrieved_docs'])} documents")

    relevance_grader, _, _ = _get_graders()
    relevant_docs = relevance_grader.filter_relevant(
        state["question"], state["retrieved_docs"]
    )

    logger.info(f"[grade_documents] {len(relevant_docs)}/{len(state['retrieved_docs'])} documents are relevant")
    return {**state, "relevant_docs": relevant_docs}


def generate(state: RAGState) -> RAGState:
    """GENERATE node: Generate an answer from relevant documents."""
    question = state["question"]
    relevant_docs = state["relevant_docs"]

    logger.info(f"[generate] Generating answer from {len(relevant_docs)} relevant docs")

    if not relevant_docs:
        logger.warning("[generate] No relevant docs available - generating without context")
        context = "No relevant documents were found."
    else:
        context = "\n\n".join(
            f"[Source {i+1}: {doc.get('title', 'Document')}]\n{doc['content']}"
            for i, doc in enumerate(relevant_docs)
        )

    system_prompt = """You are a knowledgeable AI assistant that answers questions based on provided source documents.

Rules:
- Base your answer ONLY on the provided source documents
- Be specific and cite which sources support your claims
- If the documents don't fully answer the question, say so clearly
- Be concise but comprehensive
"""

    prompt = f"""Question: {question}

Source Documents:
{context}

Please answer the question based solely on the source documents above."""

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])

    generation = response.content
    logger.info(f"[generate] Generation complete. Length: {len(generation)} chars")

    return {**state, "generation": generation}


def grade_generation(state: RAGState) -> RAGState:
    """GRADE_GENERATION node: Check for hallucinations and answer quality."""
    logger.info("[grade_generation] Grading generation for hallucinations and usefulness")

    _, hallucination_grader, answer_grader = _get_graders()

    hallucination_result = hallucination_grader.grade(
        documents=state["relevant_docs"],
        generation=state["generation"],
    )
    is_grounded = hallucination_result.score == "yes"
    logger.info(f"[grade_generation] Grounded: {is_grounded} - {hallucination_result.reasoning[:80]}")

    answer_result = answer_grader.grade(
        question=state["question"],
        generation=state["generation"],
    )
    is_useful = answer_result.score == "yes"
    logger.info(f"[grade_generation] Useful: {is_useful} - {answer_result.reasoning[:80]}")

    return {**state, "is_grounded": is_grounded, "is_useful": is_useful}


def transform_query(state: RAGState) -> RAGState:
    """TRANSFORM_QUERY node: Rewrite the query to improve retrieval."""
    original_query = state["question"]
    retry_count = state.get("retry_count", 0)

    logger.info(f"[transform_query] Rewriting query (attempt {retry_count + 1})")

    llm = _get_llm()
    prompt = f"""The original query failed to retrieve useful documents or generate a good answer.

Original query: {original_query}
Attempt number: {retry_count + 1}

Please rewrite this query to be more specific and likely to retrieve relevant documents.
Return ONLY the rewritten query, nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    new_query = response.content.strip().strip('"').strip("'")

    logger.info(f"[transform_query] Rewritten query: '{new_query[:80]}'")

    return {
        **state,
        "query_for_retrieval": new_query,
        "retry_count": retry_count + 1,
        "retrieved_docs": [],
        "relevant_docs": [],
        "generation": None,
    }


def finalize(state: RAGState) -> RAGState:
    """Terminal node: set final_answer from the current generation."""
    logger.info("[finalize] Setting final answer")
    return {**state, "final_answer": state.get("generation", "Unable to generate an answer.")}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_after_grading(state: RAGState) -> Literal["generate", "transform_query"]:
    """After grading documents: generate if we have relevant docs, else rewrite query."""
    if state["retry_count"] >= MAX_RETRIES:
        logger.warning("[route] Max retries reached - forcing generation")
        return "generate"
    if state["relevant_docs"]:
        return "generate"
    return "transform_query"


def route_after_generation(state: RAGState) -> Literal["finalize", "transform_query"]:
    """After grading generation: finalize if good, else retry."""
    if state["retry_count"] >= MAX_RETRIES:
        logger.warning("[route] Max retries reached - finalizing despite quality issues")
        return "finalize"
    if state["is_grounded"] and state["is_useful"]:
        logger.info("[route] Generation is grounded and useful -> finalizing")
        return "finalize"
    logger.info("[route] Generation failed quality check -> transforming query")
    return "transform_query"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_self_rag_graph() -> StateGraph:
    """
    Build and compile the Self-RAG LangGraph pipeline.

    Graph topology:
        retrieve -> grade_documents -> [relevant?]
            YES -> generate -> grade_generation -> [good?]
                YES -> finalize -> END
                NO  -> transform_query -> retrieve (loop)
            NO  -> transform_query -> retrieve (loop)
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("grade_generation", grade_generation)
    graph.add_node("transform_query", transform_query)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    graph.add_edge("generate", "grade_generation")
    graph.add_conditional_edges(
        "grade_generation",
        route_after_generation,
        {"finalize": "finalize", "transform_query": "transform_query"},
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("finalize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_self_rag(question: str) -> dict[str, Any]:
    """
    Run the Self-RAG pipeline on a question.

    Args:
        question: Natural language question to answer.

    Returns:
        Dict with 'answer', 'sources', and 'retry_count'.
    """
    graph = build_self_rag_graph()

    initial_state: RAGState = {
        "question": question,
        "query_for_retrieval": question,
        "retrieved_docs": [],
        "relevant_docs": [],
        "generation": None,
        "retry_count": 0,
        "is_grounded": False,
        "is_useful": False,
        "final_answer": None,
    }

    logger.info(f"[run_self_rag] Starting pipeline for: '{question}'")
    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state.get("final_answer", "No answer generated."),
        "sources": [
            {"title": d.get("title", "Unknown"), "score": d.get("score", 0)}
            for d in final_state.get("relevant_docs", [])
        ],
        "retry_count": final_state.get("retry_count", 0),
    }


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is Self-RAG and how does it differ from standard RAG?"

    print(f"\nQuestion: {question}\n{'='*60}")
    result = run_self_rag(question)
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSources used: {result['sources']}")
    print(f"Retries needed: {result['retry_count']}")
