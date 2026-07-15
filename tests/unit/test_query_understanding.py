"""Query-understanding unit tests: intent, entities, time ranges, sub-queries."""

from __future__ import annotations

import pytest

from dl_rag.config import Settings
from dl_rag.models.enums import ContentType, QueryType
from dl_rag.retrieval.query_understanding import HeuristicQueryAnalyzer


@pytest.fixture
def analyzer(settings: Settings) -> HeuristicQueryAnalyzer:
    return HeuristicQueryAnalyzer(settings)


class TestIntent:
    async def test_timeline(self, analyzer):
        a = await analyzer.analyze("How has NEP evolved since 2020?")
        assert a.query_type == QueryType.TIMELINE

    async def test_comparison(self, analyzer):
        a = await analyzer.analyze("Compare CBSE and State Board reforms")
        assert a.query_type == QueryType.COMPARISON
        assert len(a.sub_queries) >= 2

    async def test_definition(self, analyzer):
        a = await analyzer.analyze("What is SWAYAM?")
        assert a.query_type == QueryType.DEFINITION

    async def test_interview(self, analyzer):
        a = await analyzer.analyze("Show interviews featuring AI in education")
        assert a.query_type == QueryType.INTERVIEW
        assert ContentType.INTERVIEW in a.content_type_filter

    async def test_trend_or_ranking_signals(self, analyzer):
        a = await analyzer.analyze("Top trends in EdTech funding")
        assert a.query_type in (QueryType.TREND, QueryType.RANKING)

    async def test_statistics(self, analyzer):
        a = await analyzer.analyze("How many universities adopted online exams?")
        assert a.query_type == QueryType.STATISTICS


class TestEntities:
    async def test_gazetteer_canonicalisation(self, analyzer):
        a = await analyzer.analyze("What did the University Grants Commission say about SWAYAM?")
        assert "UGC" in a.entities
        assert "SWAYAM" in a.entities

    async def test_state_detection(self, analyzer):
        a = await analyzer.analyze("Digital classrooms in Karnataka and Kerala")
        assert "Karnataka" in a.entities and "Kerala" in a.entities


class TestTimeRange:
    async def test_since_year(self, analyzer):
        a = await analyzer.analyze("How has NEP evolved since 2020?")
        assert a.time_range.from_year == 2020

    async def test_between_years(self, analyzer):
        a = await analyzer.analyze("AI adoption in universities between 2018 and 2022")
        assert a.time_range.from_year == 2018
        assert a.time_range.to_year == 2022

    async def test_no_years(self, analyzer):
        a = await analyzer.analyze("What are recurring challenges in higher education?")
        assert not a.time_range.is_set()
