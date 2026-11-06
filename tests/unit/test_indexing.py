from datetime import UTC, datetime

from src.repositories.paper import PaperRepository
from src.schemas.arxiv import ArxivPaper, PaperSection, ParsedPDF
from src.services.indexing import index_papers, paper_to_document


def add_paper(database, arxiv_id: str, parsed: bool = False) -> None:
    with database.session() as session:
        repo = PaperRepository(session)
        repo.upsert_metadata(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=f"Paper {arxiv_id}",
                authors=["A. Author"],
                abstract="About agents",
                categories=["cs.AI"],
                published_date=datetime(2026, 11, 1, tzinfo=UTC),
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )
        )
        if parsed:
            repo.save_parsed_content(
                arxiv_id, ParsedPDF(raw_text="full text", sections=[PaperSection(title="Intro", content="x")])
            )


class FakeIndices:
    def refresh(self, index):
        self.refreshed = index


class FakeRawClient:
    def __init__(self):
        self.indices = FakeIndices()


class FakeOpenSearch:
    alias = "arxiv-papers"

    def __init__(self):
        self.client = FakeRawClient()
        self.documents: dict[str, dict] = {}
        self.setup_calls = 0

    def setup_index(self, recreate=False):
        self.setup_calls += 1
        return False

    def index_documents(self, documents, refresh=False):
        for doc in documents:
            self.documents[doc["arxiv_id"]] = doc
        return {"indexed": len(documents), "errors": []}


def test_document_includes_full_text_only_when_parsed(database):
    add_paper(database, "2611.00001", parsed=True)
    add_paper(database, "2611.00002", parsed=False)

    with database.session() as session:
        repo = PaperRepository(session)
        parsed = paper_to_document(repo.get_by_arxiv_id("2611.00001"))
        unparsed = paper_to_document(repo.get_by_arxiv_id("2611.00002"))

    assert parsed["has_full_text"] is True and parsed["raw_text"] == "full text"
    assert unparsed["has_full_text"] is False and unparsed["raw_text"] is None
    assert parsed["published_date"].startswith("2026-11-01T00:00:00")
    assert parsed["published_date"].endswith("+00:00")


def test_index_all_papers_in_batches_and_is_idempotent(database):
    for i in range(5):
        add_paper(database, f"2611.0000{i}")
    fake = FakeOpenSearch()

    first = index_papers(database, fake, batch_size=2)
    second = index_papers(database, fake, batch_size=2)

    assert first["indexed"] == 5 and second["indexed"] == 5
    assert len(fake.documents) == 5  # same _id each time, so no duplicates
    assert fake.client.indices.refreshed == "arxiv-papers"


def test_index_only_requested_ids(database):
    add_paper(database, "2611.00001")
    add_paper(database, "2611.00002")
    fake = FakeOpenSearch()

    stats = index_papers(database, fake, arxiv_ids=["2611.00002"])

    assert stats["indexed"] == 1
    assert list(fake.documents) == ["2611.00002"]


def test_empty_id_list_indexes_nothing(database):
    fake = FakeOpenSearch()
    assert index_papers(database, fake, arxiv_ids=[])["indexed"] == 0
    assert fake.documents == {}


# ---------- Phase 4: chunks ----------

from src.config import ChunkingSettings, Settings  # noqa: E402
from src.services.indexing import index_chunks  # noqa: E402


class FakeChunkIndex:
    name = "arxiv-papers-chunks-fake-4-v1"

    def __init__(self):
        self.documents: dict[str, dict] = {}
        self.setup_calls = 0

    def setup(self, recreate=False):
        self.setup_calls += 1

    def replace_paper_chunks(self, documents):
        for doc in documents:
            self.documents[doc["chunk_id"]] = doc
        return {"indexed": len(documents), "errors": []}

    def refresh(self):
        pass


class FakeEmbedder:
    def __init__(self):
        self.texts: list[str] = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [[0.5] * 4 for _ in texts]


def test_index_chunks_embeds_every_chunk_with_context(database):
    add_paper(database, "2611.00001", parsed=True)  # abstract + full text
    add_paper(database, "2611.00002", parsed=False)  # abstract only
    index, embedder = FakeChunkIndex(), FakeEmbedder()
    settings = Settings(chunking=ChunkingSettings(chunk_words=50, overlap_words=10, min_words=1))

    stats = index_chunks(database, index, embedder, settings=settings)

    assert stats == {"papers": 2, "chunks": 3, "errors": []}
    assert set(index.documents) == {"2611.00001_0", "2611.00001_1", "2611.00002_0"}
    doc = index.documents["2611.00001_1"]
    assert doc["section_titles"] == ["Intro"] and doc["embedding"] == [0.5] * 4
    assert doc["categories"] == ["cs.AI"] and doc["published_date"].startswith("2026-11-01")
    assert any(t.startswith("Paper 2611.00001\nSection: Intro") for t in embedder.texts)


def test_index_chunks_only_requested_papers(database):
    add_paper(database, "2611.00001")
    add_paper(database, "2611.00002")
    index = FakeChunkIndex()

    index_chunks(database, index, FakeEmbedder(), arxiv_ids=["2611.00002"])

    assert set(index.documents) == {"2611.00002_0"}
