"""Deterministic evaluation-metric unit tests."""

from __future__ import annotations

from dl_rag.evaluation import metrics as m

from tests.conftest import make_retrieved


class TestMetrics:
    def test_top_k_recall(self):
        chunks = [make_retrieved(i, url=f"https://x/{i}/") for i in range(4)]
        assert m.top_k_recall(chunks, ["https://x/1/", "https://x/9/"]) == 0.5
        assert m.top_k_recall(chunks, ["HTTPS://X/2"]) == 1.0  # normalised match
        assert m.top_k_recall(chunks, []) is None

    def test_citation_precision(self):
        assert m.citation_precision("Fact [1]. Fact [2].", n_sources=3) == 1.0
        assert m.citation_precision("Fact [1]. Bogus [9].", n_sources=3) == 0.5
        assert m.citation_precision("No cites.", n_sources=3) is None

    def test_citation_density(self):
        text = "# Heading\n\nClaim one [1].\n\nClaim two without cite.\n\nClaim three [2]."
        assert m.citation_density(text) == 2 / 3

    def test_keyword_coverage(self):
        assert m.keyword_coverage("NEP transformed pedagogy", ["NEP", "SWAYAM"]) == 0.5
        assert m.keyword_coverage("anything", []) is None

    def test_context_keyword_recall(self):
        chunks = [make_retrieved(0, text="SWAYAM expanded MOOC access nationwide.")]
        assert m.context_keyword_recall(chunks, ["SWAYAM", "blockchain"]) == 0.5
        assert m.context_keyword_recall([], ["x"]) == 0.0

    def test_mean_ignores_none(self):
        assert m.mean([1.0, None, 0.0]) == 0.5
        assert m.mean([None, None]) is None
