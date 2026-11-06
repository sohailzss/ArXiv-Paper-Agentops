"""Rate-limited arXiv API client.

arXiv rules (https://info.arxiv.org/help/api/tou.html): no more than one request
every 3 seconds, single connection. We enforce that with a lock + timestamp.
"""

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import httpx

from src.config import Settings, get_settings
from src.schemas.arxiv import ArxivPaper

logger = logging.getLogger(__name__)

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


class ArxivAPIError(Exception):
    pass


class ArxivClient:
    def __init__(self, settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = (settings or get_settings()).arxiv
        self._transport = transport  # tests inject a mock transport here
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.settings.timeout_seconds,
            follow_redirects=True,
            transport=self._transport,
            headers={"User-Agent": "agentic-rag-arxiv-curator/0.1 (portfolio project)"},
        )

    async def _respect_rate_limit(self) -> None:
        async with self._lock:
            wait = self.settings.rate_limit_delay - (time.monotonic() - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

    @staticmethod
    def build_query(category: str, from_date: date | None, to_date: date | None) -> str:
        query = f"cat:{category}"
        if from_date or to_date:
            start = (from_date or to_date).strftime("%Y%m%d") + "0000"
            end = (to_date or from_date).strftime("%Y%m%d") + "2359"
            query += f" AND submittedDate:[{start} TO {end}]"
        return query

    async def fetch_papers(
        self,
        from_date: date | None = None,
        to_date: date | None = None,
        max_results: int | None = None,
        start: int = 0,
    ) -> list[ArxivPaper]:
        params = {
            "search_query": self.build_query(self.settings.search_category, from_date, to_date),
            "start": start,
            "max_results": max_results or self.settings.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        xml_text = await self._get_with_retries(self.settings.base_url, params)
        papers = self.parse_feed(xml_text)
        logger.info("arXiv returned %d papers for %s", len(papers), params["search_query"])
        return papers

    async def _get_with_retries(self, url: str, params: dict) -> str:
        last_error: Exception | None = None
        async with self._client() as client:
            for attempt in range(1, self.settings.max_retries + 1):
                await self._respect_rate_limit()
                try:
                    response = await client.get(url, params=params)
                    if response.status_code in (429, 500, 502, 503, 504):
                        raise ArxivAPIError(f"arXiv returned HTTP {response.status_code}")
                    response.raise_for_status()
                    return response.text
                except (httpx.TransportError, ArxivAPIError) as exc:
                    last_error = exc
                    backoff = self.settings.rate_limit_delay * attempt
                    logger.warning("arXiv request failed (attempt %d): %s; retrying in %.0fs", attempt, exc, backoff)
                    await asyncio.sleep(backoff)
        raise ArxivAPIError(f"arXiv request failed after {self.settings.max_retries} attempts: {last_error}")

    @staticmethod
    def parse_feed(xml_text: str) -> list[ArxivPaper]:
        root = ET.fromstring(xml_text)
        papers: list[ArxivPaper] = []
        for entry in root.findall(f"{ATOM}entry"):
            entry_id = entry.findtext(f"{ATOM}id", default="")
            if "/abs/" not in entry_id:  # arXiv returns an error entry instead of an HTTP error sometimes
                continue
            arxiv_id = re.sub(r"v\d+$", "", entry_id.rsplit("/abs/", 1)[1])

            pdf_url = ""
            for link in entry.findall(f"{ATOM}link"):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
            pdf_url = (pdf_url or f"https://arxiv.org/pdf/{arxiv_id}").replace("http://", "https://")

            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=" ".join(entry.findtext(f"{ATOM}title", default="").split()),
                    authors=[a.findtext(f"{ATOM}name", default="").strip() for a in entry.findall(f"{ATOM}author")],
                    abstract=" ".join(entry.findtext(f"{ATOM}summary", default="").split()),
                    categories=[c.get("term", "") for c in entry.findall(f"{ATOM}category")],
                    published_date=datetime.fromisoformat(entry.findtext(f"{ATOM}published").replace("Z", "+00:00")),
                    pdf_url=pdf_url,
                )
            )
        return papers

    async def download_pdf(self, paper: ArxivPaper, max_size_mb: int = 20) -> Path:
        """Download a paper's PDF into the cache folder; reuse it if already there."""
        cache_dir = Path(self.settings.pdf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
        if path.exists() and path.stat().st_size > 0:
            return path

        max_bytes = max_size_mb * 1024 * 1024
        tmp_path = path.with_suffix(".part")
        try:
            async with self._client() as client:
                await self._respect_rate_limit()
                async with client.stream("GET", paper.pdf_url) as response:
                    response.raise_for_status()
                    size = 0
                    with tmp_path.open("wb") as fh:
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                raise ArxivAPIError(f"PDF for {paper.arxiv_id} is over {max_size_mb} MB")
                            fh.write(chunk)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        tmp_path.replace(path)
        return path
