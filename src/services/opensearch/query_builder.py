"""Builds OpenSearch query bodies. Pure functions, no I/O, so they're easy to unit test
and Phase 4's hybrid search can reuse the same filters and BM25 clause."""

from datetime import date, timedelta

# Title matches matter most, then abstract and authors; full text is a weak signal
# (long documents mention lots of terms in passing).
SEARCH_FIELDS = ["title^3", "abstract^2", "authors^1.5", "raw_text"]

HIGHLIGHT_TAGS = ("<mark>", "</mark>")


def build_filters(
    categories: list[str] | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[dict]:
    filters: list[dict] = []
    if categories:
        filters.append({"terms": {"categories": categories}})
    if date_from or date_to:
        date_range: dict[str, str] = {}
        if date_from:
            date_range["gte"] = date_from.isoformat()
        if date_to:
            # published_date has a time part, so "up to and including date_to" means "< next day".
            date_range["lt"] = (date_to + timedelta(days=1)).isoformat()
        filters.append({"range": {"published_date": date_range}})
    return filters


def build_bm25_clause(query: str) -> dict:
    """Keyword relevance: any-term match across fields, with a bonus when the words appear as a phrase."""
    return {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": SEARCH_FIELDS,
                        "type": "best_fields",
                        "tie_breaker": 0.3,
                    }
                }
            ],
            "should": [
                {"match_phrase": {"title": {"query": query, "slop": 2, "boost": 2}}},
                {"match_phrase": {"abstract": {"query": query, "slop": 3}}},
            ],
        }
    }


def build_search_query(
    query: str,
    size: int = 10,
    offset: int = 0,
    categories: list[str] | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    highlight: bool = True,
) -> dict:
    query = query.strip()
    filters = build_filters(categories, date_from, date_to)

    body: dict = {
        "size": size,
        "from": offset,
        "track_total_hits": True,
        # raw_text can be 100k+ characters; never send it back in results.
        "_source": {"excludes": ["raw_text"]},
    }

    if query:
        body["query"] = {"bool": {"must": [build_bm25_clause(query)], "filter": filters}}
        if highlight:
            body["highlight"] = {
                "pre_tags": [HIGHLIGHT_TAGS[0]],
                "post_tags": [HIGHLIGHT_TAGS[1]],
                "fields": {
                    "title": {"number_of_fragments": 0},  # whole title
                    "abstract": {"fragment_size": 200, "number_of_fragments": 2},
                    "raw_text": {"fragment_size": 200, "number_of_fragments": 2},
                },
            }
    else:
        # Empty query = browse: newest papers first (filters still apply).
        body["query"] = {"bool": {"must": [{"match_all": {}}], "filter": filters}}
        body["sort"] = [{"published_date": {"order": "desc"}}]

    return body


# ---------- chunk index (Phase 4) ----------

CHUNK_SEARCH_FIELDS = ["text", "title^2", "section_titles^1.5"]


def _filter_clause(filters: list[dict]) -> dict | None:
    return {"bool": {"filter": filters}} if filters else None


def build_chunk_bm25_query(query: str, size: int, filters: list[dict]) -> dict:
    return {
        "size": size,
        "_source": {"excludes": ["embedding"]},
        "query": {
            "bool": {
                "must": [{"multi_match": {"query": query, "fields": CHUNK_SEARCH_FIELDS, "type": "best_fields"}}],
                "filter": filters,
            }
        },
    }


def build_chunk_vector_query(vector: list[float], size: int, filters: list[dict]) -> dict:
    knn: dict = {"vector": vector, "k": size}
    if (clause := _filter_clause(filters)) is not None:
        knn["filter"] = clause  # applied during the graph search, so filtered queries still return k hits
    return {"size": size, "_source": {"excludes": ["embedding"]}, "query": {"knn": {"embedding": knn}}}
