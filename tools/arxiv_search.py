"""
Phase 4 / Step 16 — Academic search tool.

Deterministic function: given a query string, search arXiv and return
normalized results. No LLM involved here. Errors are caught and turned into
an empty result + logged reason rather than propagating as exceptions into
the graph (per the roadmap's guidance).

Uses the official `arxiv` Python package instead of hand-rolled HTTP calls —
it handles the correct headers, rate limiting, and response parsing that
arXiv's raw HTTP API is picky about (it was rejecting bare httpx requests
with 406/301 errors).
"""
from __future__ import annotations

import logging

import arxiv

logger = logging.getLogger("arxiv_search")

_client = arxiv.Client()


def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """
    Returns a list of normalized paper dicts:
    {id, title, authors, year, abstract, url}
    Returns [] on any failure (timeout, no results, API error, etc.) —
    callers should treat an empty list as "no results", not as an error.
    """
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for result in _client.results(search):
            papers.append(
                {
                    "id": result.entry_id,
                    "title": " ".join(result.title.split()),
                    "authors": [a.name for a in result.authors],
                    "year": result.published.year if result.published else None,
                    "abstract": " ".join(result.summary.split()),
                    "url": result.entry_id,
                }
            )
        return papers
    except Exception as e:
        logger.warning("arXiv search failed for query=%r: %s", query, e)
        return []
