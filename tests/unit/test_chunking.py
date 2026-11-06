import pytest

from src.config import ChunkingSettings
from src.services.chunking import chunk_paper, embedding_text

SETTINGS = ChunkingSettings(chunk_words=100, overlap_words=20, min_words=10)


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def section(title: str, n: int, tag: str = "w") -> dict:
    return {"title": title, "content": words(n, tag), "level": 2}


def test_metadata_only_paper_is_one_abstract_chunk():
    chunks = chunk_paper("2611.1", "My Title", "An abstract.", None, SETTINGS)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "2611.1_0"
    assert chunks[0].section_titles == ["Abstract"]
    assert chunks[0].text == "My Title\n\nAn abstract."


def test_skips_references_abstract_and_title_block():
    sections = [
        {"title": "My Title", "content": "authors and affiliations", "level": 2},
        section("Abstract", 50),
        section("1 Introduction", 60),
        section("References", 500),
        section("Acknowledgements", 40),
    ]
    chunks = chunk_paper("2611.1", "My Title", "abs", sections, SETTINGS)

    assert [c.section_titles for c in chunks] == [["Abstract"], ["1 Introduction"]]


def test_small_sections_share_a_chunk_until_full():
    sections = [section("A", 40), section("B", 40), section("C", 40)]
    chunks = chunk_paper("2611.1", "T", "abs", sections, SETTINGS)

    assert [c.section_titles for c in chunks[1:]] == [["A", "B"], ["C"]]
    assert all(c.word_count <= SETTINGS.chunk_words for c in chunks)


def test_long_section_is_split_with_overlap():
    chunks = chunk_paper("2611.1", "T", "abs", [section("Method", 250, tag="m")], SETTINGS)[1:]

    assert [c.word_count for c in chunks] == [100, 100, 90]
    assert all(c.section_titles == ["Method"] for c in chunks)
    # The last 20 words of one chunk start the next one.
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]
    # Every word is covered.
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(words(250, tag="m").split())


def test_tiny_trailing_piece_is_merged():
    # Main fills a chunk, so Footnote would start a 5-word chunk of its own; it's merged back instead.
    sections = [section("Main", 98), section("Footnote", 5)]
    chunks = chunk_paper("2611.1", "T", "abs", sections, SETTINGS)

    assert len(chunks) == 2
    assert chunks[1].section_titles == ["Main", "Footnote"]
    assert chunks[1].word_count == 103


def test_chunk_indexes_are_sequential():
    sections = [section("A", 250), section("B", 30), section("C", 30)]
    chunks = chunk_paper("2611.1", "T", "abs", sections, SETTINGS)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_embedding_text_adds_paper_and_section_context():
    chunk = chunk_paper("2611.1", "T", "abs", [section("Results", 20)], SETTINGS)[1]
    assert embedding_text("Paper Title", chunk).startswith("Paper Title\nSection: Results\n\n")


def test_rejects_overlap_not_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_paper("x", "T", "a", [], ChunkingSettings(chunk_words=50, overlap_words=50))
