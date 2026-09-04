import os
from base64 import urlsafe_b64decode
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"KSH1"
NONCE_SIZE = 12


class VaultConfigurationError(RuntimeError):
    pass


class EncryptedVault:
    def __init__(self, root: Path, key_base64: str | None) -> None:
        self.root = root
        self._key = self._decode_key(key_base64)

    @staticmethod
    def _decode_key(value: str | None) -> bytes:
        if not value:
            raise VaultConfigurationError("vault master key is not configured")
        try:
            key = urlsafe_b64decode(value.encode("ascii"))
        except (ValueError, UnicodeError) as exc:
            raise VaultConfigurationError("vault master key is not valid base64") from exc
        if len(key) != 32:
            raise VaultConfigurationError("vault master key must decode to exactly 32 bytes")
        return key

    def encrypt(self, document_id: str, evidence_hash: str, plaintext: bytes) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        nonce = os.urandom(NONCE_SIZE)
        aad = self._aad(document_id, evidence_hash)
        ciphertext = AESGCM(self._key).encrypt(nonce, plaintext, aad)

        destination = self.root / f"{document_id}.ksh"
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(MAGIC + nonce + ciphertext)
        temporary.chmod(0o600)
        os.replace(temporary, destination)
        return destination

    def decrypt(self, document_id: str, evidence_hash: str, path: Path) -> bytes:
        payload = path.read_bytes()
        if not payload.startswith(MAGIC) or len(payload) <= len(MAGIC) + NONCE_SIZE:
            raise ValueError("invalid encrypted vault object")
        nonce_start = len(MAGIC)
        nonce_end = nonce_start + NONCE_SIZE
        return AESGCM(self._key).decrypt(
            payload[nonce_start:nonce_end],
            payload[nonce_end:],
            self._aad(document_id, evidence_hash),
        )

    @staticmethod
    def _aad(document_id: str, evidence_hash: str) -> bytes:
        return f"koshshield:{document_id}:{evidence_hash}".encode()
