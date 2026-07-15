"""Ingestion unit tests: extraction, content-type routing, chunking, entities."""

from __future__ import annotations

from datetime import date

import pytest

from dl_rag.config import Settings
from dl_rag.ingestion.chunking.semantic_chunker import SemanticChunker
from dl_rag.ingestion.crawler.extractors import detect_content_type, extract_document
from dl_rag.ingestion.crawler.markdown import html_to_markdown
from dl_rag.ingestion.entities.extractor import EntityExtractor
from dl_rag.models.domain import SourceDocument
from dl_rag.models.enums import ContentType, EntityType, RelationType

SAMPLE_HTML = """
<html><head>
<title>NEP Update 2022 | digitalLEARNING</title>
<link rel="canonical" href="https://digitallearning.eletsonline.com/2022/06/nep-update/"/>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"NewsArticle",
 "headline":"NEP Update 2022",
 "datePublished":"2022-06-14T10:00:00+05:30",
 "author":{"@type":"Person","name":"Ravi Kumar"},
 "articleSection":"Policy Matters",
 "keywords":"NEP, policy, education"}
</script>
</head><body>
<nav>Home | About</nav>
<article>
  <h1 class="entry-title">NEP Update 2022</h1>
  <div class="entry-content">
    <p>The National Education Policy has been implemented across several states,
    with Karnataka and Kerala leading adoption of competency-based learning.</p>
    <p>Implementation began in earnest in 2021, when the Ministry of Education
    issued detailed guidelines for higher education institutions. UGC introduced
    new curriculum frameworks aligned with NEP goals.</p>
  </div>
</article>
<footer>Copyright</footer>
</body></html>
"""


class TestExtraction:
    def test_extract_document_metadata(self):
        url = "https://digitallearning.eletsonline.com/2022/06/nep-update/?utm_source=x&gclid=1"
        doc = extract_document(url, SAMPLE_HTML)
        assert doc is not None
        assert doc.title == "NEP Update 2022"
        assert doc.author == "Ravi Kumar"
        assert doc.published_date == date(2022, 6, 14)
        assert "utm_source" not in doc.url and "gclid" not in doc.url
        assert doc.id == SourceDocument.id_for_url(doc.url)
        assert "National Education Policy" in doc.content_markdown

    def test_extract_document_garbage(self):
        assert extract_document("https://x.com/", "<html><body></body></html>") is None

    def test_html_to_markdown_strips_boilerplate(self):
        md = html_to_markdown(SAMPLE_HTML)
        assert "Home | About" not in md
        assert "Copyright" not in md
        assert "National Education Policy" in md


class TestContentTypeRouting:
    def test_interview_url(self):
        ct = detect_content_type(
            "https://digitallearning.eletsonline.com/interview/vc-speaks/", [], []
        )
        assert ct == ContentType.INTERVIEW

    def test_category_slug(self):
        ct = detect_content_type("https://x.com/2023/01/foo/", ["policy-matters"], [])
        assert ct == ContentType.POLICY

    def test_fallback_other(self):
        assert detect_content_type("https://x.com/random/", [], []) == ContentType.OTHER


class TestSemanticChunker:
    def _doc(self, body: str) -> SourceDocument:
        return SourceDocument(
            id="d1",
            url="https://x.com/a/",
            title="Digital Learning in India",
            published_date=date(2021, 3, 1),
            content_markdown=body,
        )

    def test_respects_token_budget_and_ids(self, settings: Settings):
        long_para = ("Education technology adoption accelerated across Indian states. " * 40)
        body = f"## Adoption\n\n{long_para}\n\n## Challenges\n\n{long_para}"
        chunks = SemanticChunker(settings).chunk_document(self._doc(body))
        assert len(chunks) >= 2
        for i, c in enumerate(chunks):
            assert c.token_count <= settings.chunk_max_tokens
            assert c.id == f"d1:{i}"
            assert c.metadata.year == 2021
            assert c.text.startswith("Digital Learning in India")

    def test_heading_path_tracked(self, settings: Settings):
        body = "## Adoption\n\nStates rolled out smart classrooms.\n\n## Challenges\n\nFunding gaps persist across districts."
        chunks = SemanticChunker(settings).chunk_document(self._doc(body))
        paths = {" > ".join(c.metadata.heading_path) for c in chunks}
        assert any("Adoption" in p for p in paths)

    def test_empty_content(self, settings: Settings):
        assert SemanticChunker(settings).chunk_document(self._doc("")) == []


class TestEntityExtractor:
    def test_gazetteer_and_relations(self, settings: Settings):
        text = (
            "UGC introduced SWAYAM to expand online learning. "
            "Meanwhile AICTE partnered with AWS to boost cloud skills in Karnataka."
        )
        extractor = EntityExtractor(settings, use_spacy=False)
        entities, keywords, relations = extractor.extract(text, "https://x.com/a/")

        names = {e.name for e in entities}
        assert {"UGC", "SWAYAM", "AICTE", "AWS", "Karnataka"} <= names

        by_name = {e.name: e for e in entities}
        assert by_name["UGC"].type == EntityType.REGULATOR
        assert by_name["SWAYAM"].type == EntityType.SCHEME
        assert by_name["Karnataka"].type == EntityType.STATE

        triples = {(r.subject_name, r.predicate, r.object_name) for r in relations}
        assert any(
            s == "UGC" and p == RelationType.INTRODUCED and o == "SWAYAM"
            for s, p, o in triples
        )
        assert any(
            s == "AICTE" and p == RelationType.PARTNERED_WITH and o == "AWS"
            for s, p, o in triples
        )
        assert isinstance(keywords, list)

    def test_spoke_at_requires_person_subject(self, settings: Settings):
        """Without a PERSON entity, spoke_at triggers must NOT emit a relation."""
        text = "UGC addressed the World Education Summit on regulatory reform."
        extractor = EntityExtractor(settings, use_spacy=False)
        _entities, _kw, relations = extractor.extract(text, "https://x/a/")
        assert not any(r.predicate == RelationType.SPOKE_AT for r in relations)

    def test_spoke_at_with_honorific_person(self, settings: Settings):
        """Honorific-prefixed names become PERSONs and pass the spoke_at gate.

        Deliberately runs without spaCy: en_core_web_sm misses many Indian
        names, so the deterministic honorific detector is the primary path.
        """
        text = (
            "Prof. Ramesh Pokhriyal delivered the keynote at the World Education "
            "Summit in New Delhi, urging universities to embrace digital learning."
        )
        extractor = EntityExtractor(settings, use_spacy=False)
        entities, _kw, relations = extractor.extract(text, "https://x/a/")

        people = {e.name for e in entities if e.type == EntityType.PERSON}
        assert "Ramesh Pokhriyal" in people

        speaker_edges = [r for r in relations if r.predicate == RelationType.SPOKE_AT]
        assert speaker_edges, "expected a spoke_at relation"
        edge = speaker_edges[0]
        assert edge.subject_name == "Ramesh Pokhriyal"
        assert edge.object_name == "World Education Summit"
