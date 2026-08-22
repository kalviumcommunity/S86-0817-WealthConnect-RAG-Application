"""
document_loader.py — Multi-Format Document Intake for WealthConnect RAG
Assignment: GY3.19

Loads every approved wealth-management document format into a single
common form — plain text — so the rest of the pipeline (chunking,
embedding, retrieval) only needs to deal with strings.

Supported formats mapped to WealthConnect's document corpus:
  .txt   — plain-text policy extracts, notes
  .md    — Markdown guidelines, README-style docs
  .pdf   — product brochures, tax rules, investment policies (primary format)
  .html  — web-exported compliance pages, product pages
  .htm   — legacy HTML exports

PDFs are the hard case for WealthConnect:
  - Product brochures may have multi-column layouts
  - Some legacy documents may be scanned images (no extractable text)
  - pypdf extracts text page-by-page; scanned pages return empty strings
  - We detect and warn on empty-text PDFs so the admin team can re-upload
    a text-selectable version

Every loaded document carries its source filename so answers can later
be cited (FR-06 source references). Failures are caught per-file so one
corrupt document never stops the rest of the corpus from loading.
"""

from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm"}


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_txt(path: Path) -> str:
    """
    Load plain text (.txt) or Markdown (.md) files.
    Both are clean UTF-8 strings — no parsing needed.
    errors="ignore" silently drops any non-UTF-8 bytes rather than crashing.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    """
    Load a PDF file and extract text page-by-page using pypdf.

    PDFs are the dominant format in WealthConnect's corpus (product brochures,
    tax rules, investment policies). pypdf handles:
      - Single-column text documents (clean extraction)
      - Multi-column layouts (text may be interleaved — a known pypdf limitation)
      - Scanned PDFs return empty strings per page (no OCR)

    We join pages with newlines and strip leading/trailing whitespace per page
    to avoid accumulating blank lines from empty pages.

    Returns the full extracted text, or an empty string if no text is found
    (e.g. a fully scanned/image-only PDF).
    """
    reader = PdfReader(path)
    pages  = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            pages.append(page_text)

    full_text = "\n\n".join(pages)

    if not full_text.strip():
        # Warn admin team — likely a scanned/image PDF with no extractable text
        print(
            f"  [WARN] PDF has no extractable text: {path.name} "
            f"— may be a scanned document. "
            f"Re-upload a text-selectable version for indexing."
        )

    return full_text


def _load_html(path: Path) -> str:
    """
    Load an HTML or HTM file and strip all tags to plain text.

    BeautifulSoup parses the HTML tree and get_text() extracts visible
    text, inserting a space between elements to prevent words from
    adjacent tags running together (e.g. <td>Tax</td><td>Rate</td>
    becomes "Tax Rate" not "TaxRate").

    Typical WealthConnect HTML sources:
      - Web-exported compliance pages
      - Product information pages saved as HTML
    """
    raw_html = path.read_text(encoding="utf-8", errors="ignore")
    soup     = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


# ---------------------------------------------------------------------------
# Unified loader — dispatches to the right format handler
# ---------------------------------------------------------------------------

def load_text(path: Path) -> str:
    """
    Load any supported document format into plain text.

    Dispatch table:
        .txt / .md          → _load_txt()
        .pdf                → _load_pdf()
        .html / .htm        → _load_html()

    Raises:
        ValueError: if the file extension is not supported.

    The caller is responsible for catching this and any other IO errors
    (PermissionError, FileNotFoundError, etc.) — see load_all_documents().
    """
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return _load_txt(path)

    if suffix == ".pdf":
        return _load_pdf(path)

    if suffix in (".html", ".htm"):
        return _load_html(path)

    raise ValueError(
        f"Unsupported file format '{suffix}'. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


# ---------------------------------------------------------------------------
# Corpus loader — walks data/ recursively, survives failures
# ---------------------------------------------------------------------------

def load_all_documents(data_dir: str = "data") -> tuple[list[dict], list[dict]]:
    """
    Walk the data/ directory recursively and load every supported document
    into a common text form.

    For each file:
      - Attempt to load using load_text()
      - On success: append to loaded list with source metadata
      - On failure: append to skipped list with error details; continue

    One corrupt file must never stop the rest of the corpus from loading.
    In a 4,000-document corpus, some files will always be unreadable.

    Returns:
        loaded  : list of dicts with keys:
                    source    — filename (used for source citations)
                    filepath  — full path string
                    extension — file extension
                    text      — extracted plain text
                    char_count— length of extracted text
                    preview   — first 80 characters (for intake confirmation)

        skipped : list of dicts with keys:
                    source    — filename
                    filepath  — full path string
                    reason    — error message explaining the skip
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"[loader] data directory '{data_dir}' does not exist.")
        return [], []

    loaded : list[dict] = []
    skipped: list[dict] = []

    all_files = sorted(
        p for p in data_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not all_files:
        print(f"[loader] No supported documents found in '{data_dir}'.")
        return [], []

    print(f"[loader] Found {len(all_files)} supported file(s) in '{data_dir}'")
    print(f"[loader] Loading...")

    for path in all_files:
        try:
            text = load_text(path)

            record = {
                "source"    : path.name,
                "filepath"  : str(path),
                "extension" : path.suffix.lower(),
                "text"      : text,
                "char_count": len(text),
                "preview"   : repr(text[:80]),
            }
            loaded.append(record)
            print(f"  OK   {path.name:<50} {len(text):>8} chars | {repr(text[:60])}")

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append({
                "source"  : path.name,
                "filepath": str(path),
                "reason"  : reason,
            })
            print(f"  SKIP {path.name:<50} {reason}")

    return loaded, skipped


# ---------------------------------------------------------------------------
# Intake report — confirms what was loaded and flags gaps
# ---------------------------------------------------------------------------

def print_intake_report(loaded: list[dict], skipped: list[dict]) -> None:
    """
    Print a human-readable intake summary to confirm the corpus was
    loaded correctly before committing to chunking and embedding.

    Shows:
      - Total files attempted, loaded, skipped
      - Breakdown by format
      - Empty documents (loaded but no text extracted — likely scanned PDFs)
      - All skipped files and their error reasons
    """
    total     = len(loaded) + len(skipped)
    empty_doc = [d for d in loaded if d["char_count"] == 0]

    # Format breakdown
    from collections import Counter
    ext_counts = Counter(d["extension"] for d in loaded)

    print("\n" + "=" * 65)
    print("DOCUMENT INTAKE REPORT — WealthConnect")
    print("=" * 65)
    print(f"  Files found    : {total}")
    print(f"  Loaded OK      : {len(loaded)}")
    print(f"  Skipped        : {len(skipped)}")
    print(f"  Empty (no text): {len(empty_doc)}")
    print()

    print("  By format:")
    for ext, count in sorted(ext_counts.items()):
        print(f"    {ext:<8} {count} file(s)")

    if empty_doc:
        print()
        print("  Empty documents (check for scanned PDFs):")
        for d in empty_doc:
            print(f"    {d['source']}")

    if skipped:
        print()
        print("  Skipped files:")
        for s in skipped:
            print(f"    {s['source']:<50} {s['reason']}")

    print()
    total_chars = sum(d["char_count"] for d in loaded)
    print(f"  Total text loaded: {total_chars:,} characters")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point — intake run against data/
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loaded, skipped = load_all_documents("data")
    print_intake_report(loaded, skipped)
