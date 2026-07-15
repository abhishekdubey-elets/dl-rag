"""Stateless ingestion components for the digitalLEARNING RAG pipeline.

This package holds the *pure* building blocks the integrator wires into a
pipeline: a WordPress crawler, an HTML→markdown extractor, a semantic chunker,
a knowledge-graph entity/relation extractor, and a PDF/OCR processor. None of
these touch databases, Qdrant, or embeddings — they only transform inputs
(URLs, HTML, PDFs, ``SourceDocument`` objects) into domain models.

Sub-packages
------------
- :mod:`dl_rag.ingestion.crawler` — discovery + HTML extraction.
- :mod:`dl_rag.ingestion.chunking` — semantic, heading-aware chunking.
- :mod:`dl_rag.ingestion.entities` — gazetteer + NER entity/relation extraction.
- :mod:`dl_rag.ingestion.ocr` — PDF text/table extraction with OCR fallback.
"""

from __future__ import annotations

__all__ = ["chunking", "crawler", "entities", "ocr"]
