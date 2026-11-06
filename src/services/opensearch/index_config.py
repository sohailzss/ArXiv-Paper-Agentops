"""Mapping for the BM25 paper index.

The app always talks to the alias (settings.opensearch.index_name, e.g. "arxiv-papers").
The real index is versioned ("arxiv-papers-v1"). To change the mapping later: bump
INDEX_VERSION, build the new index, reindex, then move the alias. No downtime, no
code changes elsewhere.
"""

INDEX_VERSION = 1

PAPER_INDEX_BODY: dict = {
    "settings": {
        # Single node in dev, so replicas would never be assigned (and turn the cluster yellow).
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                # Like the built-in "english" analyzer (stemming: "agents" matches "agent"),
                # plus asciifolding so "Schölkopf" matches "Scholkopf".
                "paper_english": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["english_possessive", "lowercase", "asciifolding", "english_stop", "english_stemmer"],
                },
                # Names shouldn't be stemmed.
                "author_name": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "asciifolding"]},
            },
            "filter": {
                "english_possessive": {"type": "stemmer", "language": "possessive_english"},
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "english"},
            },
        },
    },
    "mappings": {
        "dynamic": "strict",  # typos in field names fail loudly instead of silently adding fields
        "properties": {
            "arxiv_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "paper_english",
                "fields": {"raw": {"type": "keyword", "ignore_above": 512}},
            },
            "abstract": {"type": "text", "analyzer": "paper_english"},
            "authors": {
                "type": "text",
                "analyzer": "author_name",
                "fields": {"raw": {"type": "keyword", "ignore_above": 256}},
            },
            "categories": {"type": "keyword"},
            "published_date": {"type": "date"},
            "pdf_url": {"type": "keyword", "index": False},
            "raw_text": {"type": "text", "analyzer": "paper_english"},
            "has_full_text": {"type": "boolean"},
            "indexed_at": {"type": "date"},
        },
    },
}
