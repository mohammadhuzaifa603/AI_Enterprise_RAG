"""Private document storage with optional Fernet encryption.

For a real deployment, supply STORAGE_ENCRYPTION_KEY from a secret/KMS
rather than committing it to disk or source control.
"""
from __future__ import annotations
import hashlib
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import base64
import hashlib as _hashlib
from cryptography.fernet import Fernet
from app.config import settings


def _fernet() -> Fernet | None:
    key = settings.storage_encryption_key
    if not key:
        if settings.storage_encryption_required:
            raise RuntimeError("STORAGE_ENCRYPTION_KEY is required for encrypted document storage.")
        return None
    try:
        return Fernet(key.encode())
    except Exception as exc:
        raise RuntimeError("STORAGE_ENCRYPTION_KEY must be a valid Fernet key.") from exc


def save_document(data: bytes, filename: str) -> tuple[str, str, bool]:
    digest = hashlib.sha256(data).hexdigest()
    f = _fernet()
    suffix = ".enc" if f else ".pdf"
    target = settings.upload_dir / f"{digest[:24]}{suffix}"
    if f:
        target.write_bytes(f.encrypt(data))
    else:
        target.write_bytes(data)
    return str(target), digest, bool(f)

@contextmanager
def materialize(path: str, encrypted: bool) -> Iterator[str]:
    p = Path(path)
    if not encrypted:
        yield str(p)
        return
    f = _fernet()
    if f is None:
        raise RuntimeError("Encrypted document cannot be opened without STORAGE_ENCRYPTION_KEY.")
    plaintext = f.decrypt(p.read_bytes())
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(plaintext)
        temp_path = tmp.name
    try:
        yield temp_path
    finally:
        Path(temp_path).unlink(missing_ok=True)
