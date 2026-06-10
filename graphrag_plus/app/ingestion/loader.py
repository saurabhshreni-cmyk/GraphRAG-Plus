"""Document loading utilities."""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import re
import socket
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

from graphrag_plus.app.ingestion.models import Document
from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)

ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
MAX_URL_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_CONTENT_LENGTH = 100  # Skip pages with too little extractable content.

# User-Agent so sites (especially Wikipedia) don't reject us.
_USER_AGENT = (
    "Mozilla/5.0 (compatible; GraphRAGBot/1.0; " "+https://github.com/saurabhshreni-cmyk/GraphRAG-Plus)"
)

# Tags whose entire subtree is noise — remove before extracting text.
_NOISE_TAGS = frozenset(
    {
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "iframe",
        "svg",
        "figure",
        "img",
        "table",
        "sup",  # Wikipedia citation superscripts [1][2]
    }
)

# CSS classes / ids that are navigation / sidebar / reference noise.
_NOISE_CLASSES = re.compile(
    r"(navbox|sidebar|reflist|references|mw-editsection|toc|catlinks|"
    r"mw-jump-link|mw-indicators|authority-control|noprint|metadata|"
    r"infobox|hatnote|mw-empty-elt|shortdescription)",
    re.IGNORECASE,
)


class UnsafeURLError(ValueError):
    """Raised when a URL targets a private/loopback/link-local address."""


class URLContentError(ValueError):
    """Raised when a URL returns empty or unusable content."""


def _id_for_source(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def _is_blocked_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_host(host: str) -> Iterable[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    return {str(info[4][0]) for info in infos}


def validate_url(url: str) -> str:
    """Reject SSRF-prone URLs. Returns canonical URL on success."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise UnsafeURLError(f"Disallowed scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL missing host")
    # Reject explicit IP literals that are private/loopback.
    try:
        ip_literal = ipaddress.ip_address(host)
        if _is_blocked_ip(str(ip_literal)):
            raise UnsafeURLError(f"Blocked IP literal: {host}")
    except ValueError:
        pass  # not an IP literal; resolve below

    addresses = _resolve_host(host)
    if not addresses:
        raise UnsafeURLError(f"Unable to resolve host: {host}")
    for addr in addresses:
        if _is_blocked_ip(addr):
            raise UnsafeURLError(f"Host {host!r} resolves to blocked address {addr}")
    return url


# ----------------------------------------------------------------- file loaders


def load_text_file(path: Path) -> Document:
    """Load UTF-8 text file."""
    text = path.read_text(encoding="utf-8")
    return Document(
        doc_id=f"doc_{_id_for_source(str(path.resolve()))}",
        source=str(path.resolve()),
        text=text,
        metadata={"type": "text"},
    )


def load_pdf_file(path: Path) -> Document:
    """Load PDF safely."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    return Document(
        doc_id=f"doc_{_id_for_source(str(path.resolve()))}",
        source=str(path.resolve()),
        text=text,
        metadata={"type": "pdf"},
    )


# ------------------------------------------------------------ HTML extraction


def _remove_noise(soup: BeautifulSoup | Tag) -> None:
    """Strip navigation, references, sidebars, scripts, tables in-place."""
    # Collect all tags to remove first, then decompose — avoids mutating the
    # tree while iterating (which causes NoneType crashes).
    to_remove: list[Tag] = []

    for tag_name in _NOISE_TAGS:
        to_remove.extend(soup.find_all(tag_name))

    for tag in soup.find_all(True):
        if not isinstance(tag, Tag):
            continue
        classes = " ".join(tag.get("class", []))  # type: ignore[arg-type]
        tag_id = tag.get("id", "") or ""
        if _NOISE_CLASSES.search(f"{classes} {tag_id}"):
            to_remove.append(tag)

    for tag in to_remove:
        with contextlib.suppress(Exception):
            tag.decompose()


def _extract_article_text(soup: BeautifulSoup) -> str:
    """Extract clean paragraph text from an HTML page.

    Strategy (ordered by specificity):
    1. Wikipedia: ``div.mw-parser-output`` → all ``<p>`` tags.
    2. Generic ``<article>`` or ``<main>`` element → all ``<p>`` tags.
    3. Fallback: all ``<p>`` tags in the page body.
    4. Last resort: ``soup.get_text()`` (full page text).

    Important: we locate the content container **before** stripping noise,
    because ``_remove_noise`` decomposes tags in-place and can destroy
    content if applied to the full page first.
    """
    # --- Wikipedia-specific: grab container BEFORE noise removal ---
    content_div = soup.find("div", class_="mw-parser-output")
    if content_div and isinstance(content_div, Tag):
        _remove_noise(content_div)
        paragraphs = content_div.find_all("p")
        text = "\n".join(p.get_text(" ", strip=True) for p in paragraphs)
        if len(text.strip()) >= MIN_CONTENT_LENGTH:
            logger.info("url_extract.strategy=wikipedia paragraphs=%d chars=%d", len(paragraphs), len(text))
            return _clean_text(text)

    # --- Generic article / main ---
    for selector in ("article", "main", '[role="main"]'):
        container = soup.find(selector)
        if container and isinstance(container, Tag):
            _remove_noise(container)
            paragraphs = container.find_all("p")
            text = "\n".join(p.get_text(" ", strip=True) for p in paragraphs)
            if len(text.strip()) >= MIN_CONTENT_LENGTH:
                logger.info(
                    "url_extract.strategy=<%s> paragraphs=%d chars=%d",
                    selector,
                    len(paragraphs),
                    len(text),
                )
                return _clean_text(text)

    # --- All <p> tags (strip noise from full page first) ---
    _remove_noise(soup)
    all_p = soup.find_all("p")
    text = "\n".join(p.get_text(" ", strip=True) for p in all_p)
    if len(text.strip()) >= MIN_CONTENT_LENGTH:
        logger.info("url_extract.strategy=all_p paragraphs=%d chars=%d", len(all_p), len(text))
        return _clean_text(text)

    # --- Last resort ---
    text = soup.get_text(" ", strip=True)
    logger.info("url_extract.strategy=full_page chars=%d", len(text))
    return _clean_text(text)


def _clean_text(text: str) -> str:
    """Collapse whitespace, strip citation brackets like [1][2][edit]."""
    text = re.sub(r"\[(?:\d+|edit|citation needed)\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ----------------------------------------------------------------- URL loader


def load_url(url: str, timeout: float = 15.0) -> Document:
    """Load URL as clean text. Validates against SSRF before fetching.

    Raises ``URLContentError`` when the page yields no usable content so
    callers can surface a clear message instead of silently returning 0 docs.
    """
    safe_url = validate_url(url)

    # Log the URL being fetched for operational traceability.
    logger.info("url_fetch.start url=%s", safe_url)

    with httpx.Client(
        timeout=timeout,
        follow_redirects=False,
        headers={"User-Agent": _USER_AGENT, "Accept": "text/html,*/*"},
    ) as client:
        response = client.get(safe_url)
        # Manual redirect handling so each hop is re-validated.
        hops = 0
        while response.is_redirect and hops < 5:
            location = response.headers.get("location", "")
            if not location:
                break
            safe_url = validate_url(str(httpx.URL(safe_url).join(location)))
            response = client.get(safe_url)
            hops += 1

        # Log HTTP status and body size before text extraction.
        logger.info(
            "url_fetch.done url=%s status=%d bytes=%d hops=%d",
            safe_url,
            response.status_code,
            len(response.content),
            hops,
        )
        response.raise_for_status()

        if len(response.content) > MAX_URL_BYTES:
            raise UnsafeURLError(f"Response too large: {len(response.content)} bytes")
        body = response.text

    # Extract clean article text instead of raw page text.
    soup = BeautifulSoup(body, "html.parser")
    text = _extract_article_text(soup)

    # Log extracted text length for ingestion diagnostics.
    logger.info("url_extract.result url=%s text_chars=%d", url, len(text))

    # Validate minimum content before proceeding.
    if len(text) < MIN_CONTENT_LENGTH:
        msg = (
            f"Failed to extract content from URL: extracted only {len(text)} chars "
            f"(minimum {MIN_CONTENT_LENGTH}). The page may require JavaScript or "
            f"has no readable content."
        )
        logger.warning("url_extract.too_short url=%s chars=%d", url, len(text))
        raise URLContentError(msg)

    return Document(
        doc_id=f"doc_{_id_for_source(url)}",
        source=url,
        text=text,
        metadata={"type": "url"},
    )


# --------------------------------------------------------------- orchestration


def load_documents(
    file_paths: list[str],
    urls: list[str],
    warnings: list[str] | None = None,
) -> list[Document]:
    """Load all docs and skip malformed inputs safely.

    If a ``warnings`` list is passed, human-readable failure reasons are
    appended so the API can surface them to the frontend.
    """
    docs: list[Document] = []
    _warnings: list[str] = warnings if warnings is not None else []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            logger.warning("ingestion.skip_missing_file path=%s", file_path)
            _warnings.append(f"File not found: {file_path}")
            continue
        try:
            if path.suffix.lower() == ".pdf":
                docs.append(load_pdf_file(path))
            else:
                docs.append(load_text_file(path))
        except Exception as exc:
            logger.warning("ingestion.file_load_failed path=%s error=%s", file_path, exc)
            _warnings.append(f"Failed to load file {file_path}: {exc}")
            continue
    for url in urls:
        try:
            docs.append(load_url(url))
        except UnsafeURLError as exc:
            logger.warning("ingestion.url_blocked url=%s reason=%s", url, exc)
            _warnings.append(f"URL blocked (security): {url} — {exc}")
            continue
        except URLContentError as exc:
            logger.warning("ingestion.url_content_empty url=%s reason=%s", url, exc)
            _warnings.append(f"Failed to extract content from URL: {url}")
            continue
        except Exception as exc:
            logger.warning("ingestion.url_load_failed url=%s error=%s", url, exc)
            _warnings.append(f"URL fetch failed: {url} — {exc}")
            continue
    loaded = [doc for doc in docs if doc.text.strip()]
    logger.info(
        "ingestion.load_complete total_docs=%d (files=%d urls=%d)",
        len(loaded),
        len(file_paths),
        len(urls),
    )
    return loaded
