"""Utils + config unit tests."""

from __future__ import annotations

from dl_rag.config import Settings
from dl_rag.utils.text import (
    clean_whitespace,
    extract_years,
    split_sentences,
    truncate_chars,
)
from dl_rag.utils.tokens import count_tokens, truncate_to_tokens


class TestTextUtils:
    def test_clean_whitespace_collapses(self):
        raw = "A  line\t with   runs\n\n\n\nand   blanks  "
        cleaned = clean_whitespace(raw)
        # Inline runs collapse to single spaces; paragraph breaks max out at \n\n.
        assert all("  " not in line for line in cleaned.split("\n"))
        assert "\n\n\n" not in cleaned
        assert cleaned.startswith("A line")
        assert cleaned.endswith("blanks")

    def test_split_sentences(self):
        text = "NEP was launched in 2020. It replaced the 1986 policy! Was it effective?"
        sents = split_sentences(text)
        assert len(sents) == 3
        assert sents[0].endswith("2020.")

    def test_extract_years_bounds(self):
        text = "From 1979 to 1985, then 2049, but not 2050 or 12020."
        years = extract_years(text)
        assert 1985 in years and 2049 in years
        assert 1979 not in years and 2050 not in years

    def test_truncate_chars(self):
        assert truncate_chars("hello world", 8).endswith("…")
        assert truncate_chars("short", 10) == "short"


class TestTokenUtils:
    def test_count_tokens_positive(self):
        assert count_tokens("The National Education Policy of India.") > 0
        assert count_tokens("") == 0

    def test_truncate_to_tokens(self):
        text = " ".join(["word"] * 500)
        out = truncate_to_tokens(text, 50)
        assert count_tokens(out) <= 55  # small tolerance across tokenizers
        assert truncate_to_tokens(text, 0) == ""


class TestSettings:
    def test_defaults(self, settings: Settings):
        assert settings.final_top_k == 8
        assert settings.retrieval_candidates == 40
        assert settings.chunk_max_tokens == 600
        assert settings.chunk_overlap_tokens == 90  # 15% of 600
        assert settings.is_production is False

    def test_api_key_set_parsing(self):
        s = Settings(api_keys="key-a, key-b ,key-c", _env_file=None)
        assert s.api_key_set == {"key-a", "key-b", "key-c"}
