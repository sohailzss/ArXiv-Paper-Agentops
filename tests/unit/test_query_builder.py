from datetime import date

from src.services.opensearch.query_builder import SEARCH_FIELDS, build_filters, build_search_query


def test_keyword_query_searches_all_fields_and_hides_raw_text():
    body = build_search_query("llm agents", size=5, offset=10)

    clause = body["query"]["bool"]["must"][0]["bool"]
    assert clause["must"][0]["multi_match"]["query"] == "llm agents"
    assert clause["must"][0]["multi_match"]["fields"] == SEARCH_FIELDS
    assert body["size"] == 5 and body["from"] == 10
    assert body["_source"] == {"excludes": ["raw_text"]}
    assert "raw_text" in body["highlight"]["fields"]
    assert "sort" not in body  # ranked by relevance


def test_empty_query_browses_newest_first():
    body = build_search_query("   ")

    assert body["query"]["bool"]["must"] == [{"match_all": {}}]
    assert body["sort"] == [{"published_date": {"order": "desc"}}]
    assert "highlight" not in body


def test_filters_categories_and_inclusive_date_range():
    filters = build_filters(categories=["cs.AI", "cs.LG"], date_from=date(2026, 11, 1), date_to=date(2026, 11, 3))

    assert {"terms": {"categories": ["cs.AI", "cs.LG"]}} in filters
    # date_to is inclusive, so the range ends before the next day starts.
    assert {"range": {"published_date": {"gte": "2026-11-01", "lt": "2026-11-04"}}} in filters


def test_no_filters_when_none_given():
    assert build_filters() == []
    body = build_search_query("transformers")
    assert body["query"]["bool"]["filter"] == []
