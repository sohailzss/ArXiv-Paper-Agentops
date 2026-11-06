"""Scientific PDF parsing with Docling.

Docling is imported lazily, so the API image (which never parses PDFs) doesn't
need it installed. Install it with `uv sync --group ingest` to run this locally.
"""

import logging
import re
from pathlib import Path

from src.config import Settings, get_settings
from src.schemas.arxiv import PaperSection, ParsedPDF

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


class PDFParseError(Exception):
    pass


class PDFParserService:
    def __init__(self, settings: Settings | None = None):
        self.settings = (settings or get_settings()).pdf_parser
        self._converter = None  # created on first use; loading Docling models takes a few seconds

    def _get_converter(self):
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = PdfPipelineOptions(
                do_ocr=self.settings.do_ocr,
                do_table_structure=self.settings.do_table_structure,
            )
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    def parse(self, pdf_path: Path) -> ParsedPDF:
        """Blocking call; run it with asyncio.to_thread from async code."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            raise PDFParseError(f"PDF missing or empty: {pdf_path}")
        if not pdf_path.read_bytes()[:5].startswith(b"%PDF"):
            raise PDFParseError(f"Not a PDF file: {pdf_path}")

        try:
            result = self._get_converter().convert(
                str(pdf_path),
                max_num_pages=self.settings.max_pages,
                max_file_size=self.settings.max_file_size_mb * 1024 * 1024,
            )
        except Exception as exc:
            raise PDFParseError(f"Docling failed on {pdf_path.name}: {exc}") from exc

        markdown = result.document.export_to_markdown()
        if not markdown.strip():
            raise PDFParseError(f"No text extracted from {pdf_path.name}")

        page_count = len(result.document.pages) if getattr(result.document, "pages", None) else None
        return ParsedPDF(raw_text=markdown, sections=self.split_sections(markdown), page_count=page_count)

    @staticmethod
    def split_sections(markdown: str) -> list[PaperSection]:
        """Split Docling's markdown on headings into (title, content) sections."""
        sections: list[PaperSection] = []
        title, level, lines = "Preamble", 1, []

        def flush() -> None:
            content = "\n".join(lines).strip()
            if content:
                sections.append(PaperSection(title=title, content=content, level=level))

        for line in markdown.splitlines():
            match = _HEADING.match(line)
            if match:
                flush()
                title, level, lines = match.group(2).strip(), len(match.group(1)), []
            else:
                lines.append(line)
        flush()
        return sections
