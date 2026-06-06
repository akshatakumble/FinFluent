"""Fernet encryption for user-uploaded financial files at rest."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from cryptography.fernet import Fernet

ENCRYPTED_SUFFIX = ".csv.enc"


def generate_key() -> bytes:
    return Fernet.generate_key()


class SecureFileStore:
    """Encrypt uploads; decrypt only when an agent needs to read CSV data."""

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key or generate_key()
        self._fernet = Fernet(self._key)

    @property
    def key(self) -> bytes:
        return self._key

    def is_encrypted_path(self, path: str) -> bool:
        return str(path).endswith(ENCRYPTED_SUFFIX)

    def encrypt_bytes(self, data: bytes, suffix: str = ".csv") -> str:
        """Write encrypted blob to disk; returns path ending in .csv.enc."""
        encrypted = self._fernet.encrypt(data)
        handle = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f"{suffix}{ENCRYPTED_SUFFIX}",
        )
        handle.write(encrypted)
        handle.flush()
        handle.close()
        return handle.name

    def decrypt_bytes(self, encrypted_path: str) -> bytes:
        return self._fernet.decrypt(Path(encrypted_path).read_bytes())

    @contextmanager
    def decrypted_csv_path(self, encrypted_path: str):
        """Yield a short-lived plaintext CSV path for pandas/sklearn."""
        plain = self.decrypt_bytes(encrypted_path)
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb")
        try:
            handle.write(plain)
            handle.flush()
            handle.close()
            yield handle.name
        finally:
            secure_delete(handle.name)

    def secure_delete(self, path: str | None) -> None:
        if path and os.path.isfile(path):
            os.remove(path)


def secure_delete(path: str | None) -> None:
    if path and os.path.isfile(path):
        os.remove(path)


def _resolve_secure_store(secure_store: SecureFileStore | None) -> SecureFileStore | None:
    if secure_store is not None:
        return secure_store
    try:
        import streamlit as st

        return st.session_state.get("secure_store")
    except Exception:
        return None


@contextmanager
def open_csv_path(path: str, secure_store: SecureFileStore | None = None):
    """
    Open a CSV path for reading. Decrypts .csv.enc files when secure_store is provided.
    Sample data under data/ is read directly without encryption.
    """
    if path.endswith(ENCRYPTED_SUFFIX):
        store = _resolve_secure_store(secure_store)
        if store is None:
            raise ValueError(
                "Cannot read encrypted upload without a decryption key. "
                "Re-upload your CSV or refresh the app."
            )
        with store.decrypted_csv_path(path) as plain_path:
            yield plain_path
    else:
        yield path
