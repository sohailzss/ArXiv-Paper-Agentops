from datetime import date

import httpx

from src.services.arxiv.client import ArxivClient

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2411.01234v2</id>
    <published>2024-11-02T17:59:59Z</published>
    <title>Agents That
      Read Papers</title>
    <summary>  We study agents.
      They read.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="http://arxiv.org/abs/2411.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2411.01234v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.AI"/>
    <category term="cs.AI"/>
    <category term="cs.CL"/>
  </entry>
</feed>"""


def test_parse_feed_extracts_clean_metadata():
    [paper] = ArxivClient.parse_feed(SAMPLE_FEED)
    assert paper.arxiv_id == "2411.01234"
    assert paper.title == "Agents That Read Papers"
    assert paper.abstract == "We study agents. They read."
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.categories == ["cs.AI", "cs.CL"]
    assert paper.pdf_url == "https://arxiv.org/pdf/2411.01234v2"
    assert paper.published_date.year == 2024


def test_build_query_with_date_range():
    q = ArxivClient.build_query("cs.AI", date(2026, 11, 1), date(2026, 11, 3))
    assert q == "cat:cs.AI AND submittedDate:[202611010000 TO 202611032359]"


async def test_fetch_papers_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        assert "cat:cs.AI" in request.url.params["search_query"]
        return httpx.Response(200, text=SAMPLE_FEED)

    client = ArxivClient(transport=httpx.MockTransport(handler))
    papers = await client.fetch_papers(max_results=1)
    assert len(papers) == 1
    assert calls["n"] == 2
