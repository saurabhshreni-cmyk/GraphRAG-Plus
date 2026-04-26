"""Document loading utilities."""

from __future__ import annotations

import hashlib
import ipaddress
import socket
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from graphrag_plus.app.ingestion.models import Document
from graphrag_plus.app.utils.logging_utils import get_logger

logger = get_logger(__name__)

ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
MAX_URL_BYTES = 10 * 1024 * 1024  # 10 MB


class UnsafeURLError(ValueError):
    """Raised when a URL targets a private/loopback/link-local address."""


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


def load_url(url: str, timeout: float = 10.0) -> Document:
    """Load URL as plain text. Validates against SSRF before fetching."""
    safe_url = validate_url(url)
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
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
        response.raise_for_status()
        if len(response.content) > MAX_URL_BYTES:
            raise UnsafeURLError(f"Response too large: {len(response.content)} bytes")
        body = response.text
    soup = BeautifulSoup(body, "html.parser")
    text = soup.get_text(" ", strip=True)
    return Document(
        doc_id=f"doc_{_id_for_source(url)}",
        source=url,
        text=text,
        metadata={"type": "url"},
    )


def load_documents(file_paths: list[str], urls: list[str]) -> list[Document]:
    """Load all docs and skip malformed inputs safely."""
    docs: list[Document] = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            logger.warning("ingestion.skip_missing_file path=%s", file_path)
            continue
        try:
            if path.suffix.lower() == ".pdf":
                docs.append(load_pdf_file(path))
            else:
                docs.append(load_text_file(path))
        except Exception as exc:
            logger.warning("ingestion.file_load_failed path=%s error=%s", file_path, exc)
            continue
    for url in urls:
        try:
            docs.append(load_url(url))
        except UnsafeURLError as exc:
            logger.warning("ingestion.url_blocked url=%s reason=%s", url, exc)
            continue
        except Exception as exc:
            logger.warning("ingestion.url_load_failed url=%s error=%s", url, exc)
            continue
    return [doc for doc in docs if doc.text.strip()]
