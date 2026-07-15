"""Evaluator — runs eval cases through the live pipeline and scores them.

Two tiers of metrics:

* **Deterministic** (always computed): top-k recall, citation precision,
  citation density, keyword coverage, context keyword recall, intent accuracy,
  latency, token usage.
* **LLM-as-judge** (optional): faithfulness & groundedness, scored 0–1 by the
  configured LLM. Skipped gracefully (``None``) when the judge is unavailable.
"""

from __future__ import annotations

import time
from typing import Any

import orjson
from pydantic import BaseModel, Field

from dl_rag.evaluation import metrics as m
from dl_rag.generation.answer_generator import AnswerGenerator
from dl_rag.logging_config import get_logger
from dl_rag.models.domain import GeneratedAnswer, RetrievedChunk
from dl_rag.models.enums import QueryType
from dl_rag.protocols import LLMClient, QueryAnalyzer
from dl_rag.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger(__name__)

_JUDGE_SYSTEM = (
    "You are a strict evaluation judge. You receive a QUESTION, the SOURCES that "
    "were provided to an assistant, and the assistant's ANSWER. Score two things:\n"
    "1. faithfulness — are the factual claims in the answer supported by the sources?\n"
    "2. groundedness — does the answer stick to the sources rather than outside knowledge?\n"
    'Reply with ONLY a JSON object: {"faithfulness": <0..1>, "groundedness": <0..1>, '
    '"unsupported_claims": ["..."]}'
)


class EvalCase(BaseModel):
    """One evaluation question with optional gold data."""

    query: str
    expected_query_type: QueryType | None = None
    expected_urls: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    query: str
    query_type: str
    intent_correct: bool | None = None
    retrieved: int = 0
    top_k_recall: float | None = None
    context_keyword_recall: float | None = None
    citation_precision: float | None = None
    citation_density: float | None = None
    keyword_coverage: float | None = None
    faithfulness: float | None = None
    groundedness: float | None = None
    confidence: float = 0.0
    grounded_flag: bool = True
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    answer_preview: str = ""


class EvalReport(BaseModel):
    cases: list[CaseResult] = Field(default_factory=list)
    summary: dict[str, float | None] = Field(default_factory=dict)

    def compute_summary(self) -> None:
        cs = self.cases
        self.summary = {
            "intent_accuracy": m.mean(
                [1.0 if c.intent_correct else 0.0 for c in cs if c.intent_correct is not None]
            ),
            "top_k_recall": m.mean([c.top_k_recall for c in cs]),
            "context_keyword_recall": m.mean([c.context_keyword_recall for c in cs]),
            "citation_precision": m.mean([c.citation_precision for c in cs]),
            "citation_density": m.mean([c.citation_density for c in cs]),
            "keyword_coverage": m.mean([c.keyword_coverage for c in cs]),
            "faithfulness": m.mean([c.faithfulness for c in cs]),
            "groundedness": m.mean([c.groundedness for c in cs]),
            "avg_confidence": m.mean([c.confidence for c in cs]),
            "avg_latency_ms": m.mean([float(c.latency_ms) for c in cs]),
            "avg_prompt_tokens": m.mean([float(c.prompt_tokens) for c in cs]),
            "avg_completion_tokens": m.mean([float(c.completion_tokens) for c in cs]),
        }


class Evaluator:
    def __init__(
        self,
        *,
        analyzer: QueryAnalyzer,
        retriever: HybridRetriever,
        generator: AnswerGenerator | None = None,
        judge: LLMClient | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._retriever = retriever
        self._generator = generator
        self._judge = judge

    # ------------------------------------------------------------------ #
    async def evaluate_case(self, case: EvalCase) -> CaseResult:
        start = time.perf_counter()
        analysis = await self._analyzer.analyze(case.query)
        chunks = await self._retriever.retrieve(analysis)

        answer: GeneratedAnswer | None = None
        if self._generator is not None:
            answer = await self._generator.generate(analysis, chunks)
        latency_ms = int((time.perf_counter() - start) * 1000)

        answer_text = answer.answer if answer else ""
        n_sources = len(answer.citations) if answer else len(chunks)

        result = CaseResult(
            query=case.query,
            query_type=analysis.query_type.value,
            intent_correct=(
                None
                if case.expected_query_type is None
                else analysis.query_type == case.expected_query_type
            ),
            retrieved=len(chunks),
            top_k_recall=m.top_k_recall(chunks, case.expected_urls),
            context_keyword_recall=m.context_keyword_recall(chunks, case.expected_keywords),
            citation_precision=m.citation_precision(answer_text, n_sources),
            citation_density=m.citation_density(answer_text),
            keyword_coverage=m.keyword_coverage(answer_text, case.expected_keywords),
            confidence=answer.confidence if answer else 0.0,
            grounded_flag=answer.grounded if answer else True,
            latency_ms=latency_ms,
            prompt_tokens=answer.prompt_tokens if answer else 0,
            completion_tokens=answer.completion_tokens if answer else 0,
            answer_preview=answer_text[:240],
        )

        if self._judge is not None and answer is not None and chunks:
            judged = await self._judge_case(case.query, chunks, answer_text)
            if judged is not None:
                result.faithfulness = judged.get("faithfulness")
                result.groundedness = judged.get("groundedness")
        return result

    async def evaluate(self, cases: list[EvalCase]) -> EvalReport:
        report = EvalReport()
        for i, case in enumerate(cases, start=1):
            try:
                result = await self.evaluate_case(case)
            except Exception as exc:  # noqa: BLE001 - one bad case must not kill the run
                logger.error("eval.case_failed", query=case.query, error=str(exc))
                continue
            report.cases.append(result)
            logger.info(
                "eval.case_done",
                n=i,
                query_type=result.query_type,
                retrieved=result.retrieved,
                latency_ms=result.latency_ms,
            )
        report.compute_summary()
        return report

    # ------------------------------------------------------------------ #
    async def _judge_case(
        self, query: str, chunks: list[RetrievedChunk], answer_text: str
    ) -> dict[str, Any] | None:
        sources_blob = "\n\n".join(
            f"[{i}] {rc.chunk.metadata.title}\n{rc.chunk.text[:800]}"
            for i, rc in enumerate(chunks, start=1)
        )
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"QUESTION:\n{query}\n\nSOURCES:\n{sources_blob}\n\n"
                    f"ANSWER:\n{answer_text}"
                ),
            },
        ]
        try:
            text, _usage = await self._judge.complete(messages, temperature=0.0)
            payload = orjson.loads(_extract_json(text))
            return {
                "faithfulness": _clamp01(payload.get("faithfulness")),
                "groundedness": _clamp01(payload.get("groundedness")),
            }
        except Exception as exc:  # noqa: BLE001 - judge is best-effort
            logger.warning("eval.judge_failed", error=str(exc))
            return None


def _extract_json(text: str) -> bytes:
    """Pull the first {...} block out of a possibly chatty judge reply."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in judge reply")
    return text[start : end + 1].encode("utf-8")


def _clamp01(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
