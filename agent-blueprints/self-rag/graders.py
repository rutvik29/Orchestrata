"""
Pydantic-structured LLM graders for the Self-RAG pipeline.

Three graders:
- RelevanceGrader: Is a retrieved document relevant to the query?
- HallucinationGrader: Is the generated answer grounded in the retrieved documents?
- AnswerGrader: Does the generated answer actually address the question?
"""

import logging
import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------

class RelevanceScore(BaseModel):
    """Binary relevance score for a retrieved document."""
    score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the query, 'no' otherwise."
    )
    reasoning: str = Field(
        description="Brief explanation (1-2 sentences) for the relevance decision."
    )


class HallucinationScore(BaseModel):
    """Hallucination check - is the answer grounded in the documents?"""
    score: Literal["yes", "no"] = Field(
        description="'yes' if the answer is grounded in the provided documents (no hallucination), 'no' if it contains hallucinations."
    )
    reasoning: str = Field(
        description="Brief explanation of what is or isn't grounded."
    )


class AnswerScore(BaseModel):
    """Does the answer resolve the original question?"""
    score: Literal["yes", "no"] = Field(
        description="'yes' if the answer fully resolves the question, 'no' if it is incomplete or off-topic."
    )
    reasoning: str = Field(
        description="Brief explanation of whether the question was addressed."
    )


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm():
    """Get a structured-output-capable LLM."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    else:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            temperature=0,
        )


# ---------------------------------------------------------------------------
# Grader classes
# ---------------------------------------------------------------------------

class RelevanceGrader:
    """
    Grades whether a retrieved document is relevant to the user query.

    Uses structured output (Pydantic) to get a deterministic yes/no score.
    """

    SYSTEM_PROMPT = """You are a precise relevance grader for a retrieval system.

Your task: Determine if a retrieved document contains information relevant to answering the user's query.

Rules:
- Score 'yes' if the document has DIRECT, USEFUL information for the query
- Score 'no' if the document is off-topic or only tangentially related
- Be strict: partial relevance that doesn't help answer the question = 'no'
- Do not consider your own knowledge - only evaluate the document's content
"""

    def __init__(self):
        self._llm = None

    def _get_structured_llm(self):
        if self._llm is None:
            self._llm = _get_llm().with_structured_output(RelevanceScore)
        return self._llm

    def grade(self, query: str, document_content: str) -> RelevanceScore:
        """
        Grade document relevance.

        Args:
            query: The user's search query.
            document_content: The retrieved document text to evaluate.

        Returns:
            RelevanceScore with binary score and reasoning.
        """
        logger.debug(f"[RelevanceGrader] Grading doc for query: '{query[:60]}'")

        prompt = f"""Query: {query}

Retrieved Document:
{document_content[:2000]}

Is this document relevant to the query?"""

        structured_llm = self._get_structured_llm()
        result = structured_llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        logger.debug(f"[RelevanceGrader] Score: {result.score} - {result.reasoning[:80]}")
        return result

    def grade_batch(self, query: str, documents: list[dict]) -> list[tuple[dict, RelevanceScore]]:
        """Grade multiple documents for relevance."""
        results = []
        for doc in documents:
            score = self.grade(query, doc.get("content", ""))
            results.append((doc, score))
        return results

    def filter_relevant(self, query: str, documents: list[dict]) -> list[dict]:
        """Return only documents graded as relevant."""
        graded = self.grade_batch(query, documents)
        relevant = [doc for doc, score in graded if score.score == "yes"]
        logger.info(f"[RelevanceGrader] {len(relevant)}/{len(documents)} documents are relevant")
        return relevant


class HallucinationGrader:
    """
    Checks whether a generated answer is grounded in the retrieved documents.
    """

    SYSTEM_PROMPT = """You are a hallucination detector for a RAG system.

Your task: Determine if the generated answer is grounded in (supported by) the provided source documents.

Rules:
- Score 'yes' (no hallucination) if ALL factual claims in the answer can be traced to the documents
- Score 'no' (hallucination detected) if the answer contains facts NOT present in the documents
- Ignore general common knowledge - focus on specific claims, numbers, names, and technical details
"""

    def __init__(self):
        self._llm = None

    def _get_structured_llm(self):
        if self._llm is None:
            self._llm = _get_llm().with_structured_output(HallucinationScore)
        return self._llm

    def grade(self, documents: list[dict], generation: str) -> HallucinationScore:
        """Grade whether the generation is grounded in the documents."""
        logger.debug(f"[HallucinationGrader] Checking generation against {len(documents)} docs")

        docs_text = "\n\n".join(
            f"[Document {i+1}]: {doc.get('content', '')[:800]}"
            for i, doc in enumerate(documents)
        )

        prompt = f"""Source Documents:
{docs_text}

Generated Answer:
{generation}

Is the generated answer grounded in the source documents (no hallucination)?"""

        structured_llm = self._get_structured_llm()
        result = structured_llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        logger.debug(f"[HallucinationGrader] Score: {result.score} - {result.reasoning[:80]}")
        return result


class AnswerGrader:
    """Evaluates whether the generated answer resolves the user's question."""

    SYSTEM_PROMPT = """You are an answer quality evaluator.

Your task: Determine if the generated answer fully and correctly addresses the user's question.

Rules:
- Score 'yes' if the answer directly addresses what was asked with sufficient detail
- Score 'no' if the answer is incomplete, vague, off-topic, or dodges the question
"""

    def __init__(self):
        self._llm = None

    def _get_structured_llm(self):
        if self._llm is None:
            self._llm = _get_llm().with_structured_output(AnswerScore)
        return self._llm

    def grade(self, question: str, generation: str) -> AnswerScore:
        """Grade whether the answer resolves the question."""
        logger.debug(f"[AnswerGrader] Grading answer for question: '{question[:60]}'")

        prompt = f"""User Question: {question}

Generated Answer:
{generation}

Does this answer fully resolve the user's question?"""

        structured_llm = self._get_structured_llm()
        result = structured_llm.invoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        logger.debug(f"[AnswerGrader] Score: {result.score} - {result.reasoning[:80]}")
        return result
