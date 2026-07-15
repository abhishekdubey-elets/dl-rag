"""Prompts, citations, and answer-generator unit tests (fake LLM)."""

from __future__ import annotations

from dl_rag.config import Settings
from dl_rag.constants import NO_EVIDENCE_MESSAGE
from dl_rag.generation.answer_generator import AnswerGenerator
from dl_rag.generation.citations import (
    build_citations,
    extract_cited_indices,
    to_source_refs,
    used_citations,
)
from dl_rag.generation.prompts import build_messages, format_context
from dl_rag.models.domain import QueryAnalysis
from dl_rag.models.enums import ConfidenceBand, QueryType

from tests.conftest import FakeLLM, make_retrieved


def _analysis(q: str = "How has NEP evolved since 2020?") -> QueryAnalysis:
    return QueryAnalysis(
        original_query=q, normalized_query=q.lower(), query_type=QueryType.TIMELINE
    )


class TestPrompts:
    def test_format_context_numbers_sources(self, retrieved_chunks):
        ctx = format_context(retrieved_chunks)
        assert "[1]" in ctx and "[3]" in ctx
        assert "Article 0" in ctx
        assert "https://digitallearning.eletsonline.com/a0/" in ctx

    def test_build_messages_shape(self, retrieved_chunks):
        msgs = build_messages(_analysis(), retrieved_chunks)
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        user = msgs[-1]["content"]
        assert "[1]" in user  # sources block present
        assert "NEP" in user  # actual question present

    def test_build_messages_includes_history(self, retrieved_chunks):
        msgs = build_messages(
            _analysis(),
            retrieved_chunks,
            history_summary="Earlier we discussed SWAYAM.",
            history_turns=[{"role": "user", "content": "What is SWAYAM?"}],
        )
        joined = " ".join(m["content"] for m in msgs)
        assert "SWAYAM" in joined


class TestCitations:
    def test_build_and_dedupe(self):
        chunks = [
            make_retrieved(0, url="https://x/a/"),
            make_retrieved(1, url="https://x/a/"),  # same URL → dedup
            make_retrieved(2, url="https://x/b/"),
        ]
        cites = build_citations(chunks)
        urls = [c.url for c in cites]
        assert len(urls) == len(set(urls))

    def test_extract_indices(self):
        assert extract_cited_indices("Claim [1]. More [2][3]. None [12]") == {1, 2, 3, 12}
        assert extract_cited_indices("no citations") == set()

    def test_used_citations_subset_and_fallback(self):
        chunks = [make_retrieved(i, url=f"https://x/{i}/") for i in range(6)]
        cites = build_citations(chunks)
        used = used_citations(cites, "Only cites [2]")
        assert [c.index for c in used] == [2]
        fallback = used_citations(cites, "cites nothing")
        assert 1 <= len(fallback) <= 5

    def test_to_source_refs(self, retrieved_chunks):
        refs = to_source_refs(build_citations(retrieved_chunks))
        assert refs[0].url.startswith("https://")
        assert refs[0].date is not None


class TestAnswerGenerator:
    async def test_generate_grounded(self, settings: Settings, retrieved_chunks):
        gen = AnswerGenerator(FakeLLM("NEP evolved steadily [1][2]."), settings)
        out = await gen.generate(_analysis(), retrieved_chunks)
        assert out.grounded is True
        assert out.retrieved_documents == 3
        assert out.confidence > 0.3
        assert out.confidence_band in (ConfidenceBand.MEDIUM, ConfidenceBand.HIGH)
        assert out.prompt_tokens == 100 and out.completion_tokens == 50
        assert [c.index for c in out.citations]

    async def test_generate_no_evidence(self, settings: Settings):
        llm = FakeLLM()
        gen = AnswerGenerator(llm, settings)
        out = await gen.generate(_analysis(), [])
        assert out.answer == NO_EVIDENCE_MESSAGE
        assert out.grounded is False
        assert out.confidence == 0.0
        assert out.retrieved_documents == 0
        assert llm.calls == []  # LLM must not be called without evidence

    async def test_stream_and_finalize(self, settings: Settings, retrieved_chunks):
        gen = AnswerGenerator(FakeLLM("Streamed answer [1]."), settings)
        parts = [p async for p in gen.stream_tokens(_analysis(), retrieved_chunks)]
        full = "".join(parts)
        assert "Streamed" in full
        final = gen.finalize(_analysis(), retrieved_chunks, full)
        assert final.grounded is True
        assert final.citations

    async def test_stream_no_evidence(self, settings: Settings):
        gen = AnswerGenerator(FakeLLM(), settings)
        parts = [p async for p in gen.stream_tokens(_analysis(), [])]
        assert "".join(parts) == NO_EVIDENCE_MESSAGE
