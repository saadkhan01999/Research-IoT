"""Bibliographic look-ups (Crossref, arXiv, OpenAlex) with an on-disk cache.

Every HTTP answer is saved under ``cache_dir`` keyed by the URL, so a verification or a
search can be repeated offline and shown to an examiner exactly as it was received.
"""

from __future__ import annotations

import difflib
import hashlib
import html
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

USER_AGENT = "iot-attack-eval-literature-check/0.1 (academic research script)"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_last_call: dict[str, float] = {}


@dataclass
class Record:
    """Normalised bibliographic record from any source."""

    source: str
    title: str = ""
    authors: list[str] = field(default_factory=list)  # "Family, Given"
    year: int | None = None
    venue: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    arxiv: str = ""
    type: str = ""
    publisher: str = ""
    reference_count: int | None = None
    url: str = ""
    abstract: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def first_author_family(self) -> str:
        return self.authors[0].split(",")[0].strip() if self.authors else ""


# ---------------------------------------------------------------- HTTP with cache
def http_get(url: str, cache_dir: Path, *, min_interval: float = 1.0, retries: int = 3) -> bytes:
    """GET ``url`` once; later calls read the cached copy. Polite spacing per host."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".bin")
    if cache_file.exists():
        return cache_file.read_bytes()
    host = urllib.parse.urlparse(url).netloc
    for attempt in range(1, retries + 1):
        wait = min_interval - (time.time() - _last_call.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.time()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body: bytes = response.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                body = b""
                break
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    cache_file.write_bytes(body)
    with (cache_dir / "index.tsv").open("a", encoding="utf-8") as index:
        index.write(f"{cache_file.name}\t{time.strftime('%Y-%m-%d %H:%M:%S')}\t{url}\n")
    return body


# ---------------------------------------------------------------- text helpers
def normalize(text: str) -> str:
    """Lower-case ASCII words only, for fuzzy comparison."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"<[^>]+>", " ", text)  # JATS tags in Crossref titles
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(a: str, b: str) -> float:
    """0..1 similarity of two normalised strings."""
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def strip_tags(text: str) -> str:
    """Plain text: markup removed and HTML entities such as &amp; decoded."""
    plain = re.sub(r"<[^>]+>", " ", text or "")
    for _ in range(2):  # some records are escaped twice (&amp;amp;)
        plain = html.unescape(plain)
    return re.sub(r"\s+", " ", plain).strip()


# ---------------------------------------------------------------- Crossref
def _crossref_year(msg: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued", "published"):
        parts = msg.get(key, {}).get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def crossref_record(msg: dict[str, Any]) -> Record:
    authors = []
    for a in msg.get("author", []):
        if "family" in a:
            authors.append(f"{a['family']}, {a.get('given', '')}".strip().rstrip(","))
        elif "name" in a:
            authors.append(a["name"])
    pages = msg.get("page", "") or msg.get("article-number", "")
    title = strip_tags((msg.get("title") or [""])[0])
    subtitle = strip_tags((msg.get("subtitle") or [""])[0])
    if subtitle and normalize(subtitle) not in normalize(title):
        title = f"{title}: {subtitle}"
    return Record(
        source="crossref",
        title=title,
        authors=authors,
        year=_crossref_year(msg),
        venue=strip_tags((msg.get("container-title") or [""])[0]),
        volume=str(msg.get("volume", "")),
        issue=str(msg.get("issue", "")),
        pages=str(pages),
        doi=str(msg.get("DOI", "")).lower(),
        type=str(msg.get("type", "")),
        publisher=str(msg.get("publisher", "")),
        reference_count=msg.get("reference-count"),
        url=str(msg.get("URL", "")),
        abstract=strip_tags(msg.get("abstract", "")),
        extra={
            "published_online": msg.get("published-online", {}).get("date-parts"),
            "published_print": msg.get("published-print", {}).get("date-parts"),
            "relation": msg.get("relation", {}),
            "subtype": msg.get("subtype", ""),
        },
    )


def crossref_work(doi: str, cache_dir: Path) -> Record | None:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi.strip(), safe="/")
    body = http_get(url, cache_dir)
    if not body:
        return None
    return crossref_record(json.loads(body)["message"])


def crossref_search(query: str, cache_dir: Path, *, rows: int = 5,
                    filters: str = "") -> list[Record]:
    params = {"query.bibliographic": query, "rows": str(rows)}
    if filters:
        params["filter"] = filters
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    body = http_get(url, cache_dir)
    items = json.loads(body)["message"]["items"] if body else []
    return [crossref_record(m) for m in items]


# ---------------------------------------------------------------- arXiv
def _arxiv_entries(xml_bytes: bytes) -> list[Record]:
    root = ET.fromstring(xml_bytes)
    records = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        if not title or title.lower() == "error":
            continue
        arxiv_id = (entry.findtext(f"{_ATOM}id") or "").rsplit("/abs/", 1)[-1]
        published = entry.findtext(f"{_ATOM}published") or ""
        records.append(Record(
            source="arxiv",
            title=title,
            authors=[_arxiv_author(a.findtext(f"{_ATOM}name") or "")
                     for a in entry.findall(f"{_ATOM}author")],
            year=int(published[:4]) if published else None,
            venue="arXiv",
            doi=(entry.findtext(f"{_ARXIV}doi") or "").lower(),
            arxiv=re.sub(r"v\d+$", "", arxiv_id),
            type="preprint",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            abstract=" ".join((entry.findtext(f"{_ATOM}summary") or "").split()),
            extra={
                "published": published,
                "updated": entry.findtext(f"{_ATOM}updated"),
                "journal_ref": entry.findtext(f"{_ARXIV}journal_ref"),
                "comment": entry.findtext(f"{_ARXIV}comment"),
                "version_id": arxiv_id,
            },
        ))
    return records


def _arxiv_author(name: str) -> str:
    parts = name.split()
    return f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else name


def arxiv_entry(arxiv_id: str, cache_dir: Path) -> Record | None:
    url = "https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(arxiv_id)
    found = _arxiv_entries(http_get(url, cache_dir, min_interval=3.0))
    return found[0] if found else None


def arxiv_search(query: str, cache_dir: Path, *, max_results: int = 100) -> list[Record]:
    params = {"search_query": query, "start": "0", "max_results": str(max_results),
              "sortBy": "relevance"}
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    return _arxiv_entries(http_get(url, cache_dir, min_interval=3.0))


# ---------------------------------------------------------------- full texts
class _TextExtractor(HTMLParser):
    """Visible text of an HTML page, without scripts, styles and navigation."""

    _SKIP = {"script", "style", "noscript", "nav", "header", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._depth += 1
        elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "table"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._depth:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._depth:
            self.parts.append(data)


def html_to_text(raw: bytes) -> str:
    parser = _TextExtractor()
    parser.feed(raw.decode("utf-8", errors="replace"))
    text = "".join(parser.parts)
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def fetch_document(url: str, out_dir: Path, *, min_interval: float = 2.0) -> dict[str, Any]:
    """Download one document to ``out_dir`` and write its plain text next to it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    host = urllib.parse.urlparse(url).netloc
    wait = min_interval - (time.time() - _last_call.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.time()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    info: dict[str, Any] = {"url": url, "status": "", "content_type": "", "chars": 0}
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read()
            info["final_url"] = response.geturl()
            info["content_type"] = response.headers.get("Content-Type", "")
    except (urllib.error.URLError, TimeoutError) as exc:
        info["status"] = f"error: {exc}"
        return info
    is_pdf = body[:5] == b"%PDF-"
    raw_path = out_dir / ("source.pdf" if is_pdf else "source.html")
    raw_path.write_bytes(body)
    try:
        text = pdf_to_text(raw_path) if is_pdf else html_to_text(body)
    except Exception as exc:  # a broken PDF must not stop the batch
        info["status"] = f"error: cannot read ({exc})"
        return info
    (out_dir / "text.txt").write_text(text, encoding="utf-8")
    info.update({"status": "ok", "chars": len(text), "file": raw_path.name})
    return info


def openalex_by_doi(doi: str, cache_dir: Path) -> dict[str, Any] | None:
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi)
    body = http_get(url, cache_dir, min_interval=0.5)
    return json.loads(body) if body else None


# ---------------------------------------------------------------- OpenAlex
def _openalex_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
    return " ".join(w for _, w in words)


def openalex_record(work: dict[str, Any]) -> Record:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    biblio = work.get("biblio") or {}
    first, last = biblio.get("first_page") or "", biblio.get("last_page") or ""
    ids = work.get("ids") or {}
    return Record(
        source="openalex",
        title=strip_tags(work.get("display_name") or ""),
        authors=[_arxiv_author((a.get("author") or {}).get("display_name") or "")
                 for a in work.get("authorships", [])],
        year=work.get("publication_year"),
        venue=strip_tags(src.get("display_name") or ""),
        volume=biblio.get("volume") or "",
        issue=biblio.get("issue") or "",
        pages=f"{first}-{last}" if first and last and first != last else first,
        doi=(work.get("doi") or "").replace("https://doi.org/", "").lower(),
        type=work.get("type") or "",
        reference_count=work.get("referenced_works_count"),
        url=ids.get("openalex", ""),
        abstract=_openalex_abstract(work.get("abstract_inverted_index")),
        extra={"cited_by_count": work.get("cited_by_count"),
               "is_oa": (work.get("open_access") or {}).get("is_oa"),
               "oa_url": (work.get("open_access") or {}).get("oa_url")},
    )


def openalex_search(query: str, cache_dir: Path, *, year_from: int, year_to: int,
                    per_page: int = 100, pages: int = 1) -> tuple[int, list[Record]]:
    """Relevance-ranked OpenAlex search; returns (total hits, records of the first pages)."""
    total, records = 0, []
    for page in range(1, pages + 1):
        params = {
            "search": query,
            "filter": f"from_publication_date:{year_from}-01-01,"
                      f"to_publication_date:{year_to}-12-31",
            "per-page": str(per_page),
            "page": str(page),
        }
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        data = json.loads(http_get(url, cache_dir, min_interval=0.5))
        total = data.get("meta", {}).get("count", 0)
        records += [openalex_record(w) for w in data.get("results", [])]
        if len(data.get("results", [])) < per_page:
            break
    return total, records
