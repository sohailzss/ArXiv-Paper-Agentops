"""Backfill: copy every paper from Neon into OpenSearch (BM25 paper index + chunk/embedding index).

The DAG indexes new papers on each run; use this for papers ingested earlier, after changing
a mapping, or after switching EMBEDDINGS__PROVIDER (each provider has its own chunk index).

    uv run python scripts/index_papers.py              # index all papers (safe to rerun)
    uv run python scripts/index_papers.py --recreate   # drop both indexes and rebuild from Neon
    uv run python scripts/index_papers.py --skip-chunks  # BM25 paper index only (no embedding calls)
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.database import Database  # noqa: E402
from src.services.embeddings.client import get_embeddings_client  # noqa: E402
from src.services.indexing import build_chunk_index, index_chunks, index_papers  # noqa: E402
from src.services.opensearch.client import OpenSearchClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("opensearch").setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild the indexes first")
    parser.add_argument("--skip-chunks", action="store_true", help="only the BM25 paper index")
    args = parser.parse_args()

    db = Database()
    opensearch = OpenSearchClient()
    try:
        if args.recreate:
            opensearch.setup_index(recreate=True)
        stats = index_papers(db, opensearch)
        print(f"Indexed {stats['indexed']} papers into '{opensearch.alias}' -> '{opensearch.versioned_index}'")
        print(f"Documents in index: {opensearch.count()}")
        errors = list(stats["errors"])

        if not args.skip_chunks:
            embedder = get_embeddings_client()
            try:
                chunk_index = build_chunk_index(opensearch, embedder)
                if args.recreate:
                    chunk_index.setup(recreate=True)
                chunks = index_chunks(db, chunk_index, embedder)
                print(f"Indexed {chunks['chunks']} chunks from {chunks['papers']} papers into '{chunk_index.name}'")
                print(f"Embedding model: {embedder.model_id}   chunks in index: {chunk_index.count()}")
                errors += chunks["errors"]
            finally:
                embedder.close()

        for error in errors[:5]:
            print(f"ERROR {error}")
        return 1 if errors else 0
    finally:
        opensearch.close()
        db.close()


if __name__ == "__main__":
    sys.exit(main())
