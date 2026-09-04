import base64
from pathlib import Path

from koshshield.security.vault import EncryptedVault


def test_vault_encrypts_and_decrypts_without_plaintext(tmp_path: Path) -> None:
    key = base64.urlsafe_b64encode(b"v" * 32).decode()
    vault = EncryptedVault(tmp_path, key)
    plaintext = b"Aadhaar 1111 2222 3333"

    encrypted_path = vault.encrypt("document-1", "evidence-hash", plaintext)

    assert plaintext not in encrypted_path.read_bytes()
    assert vault.decrypt("document-1", "evidence-hash", encrypted_path) == plaintext
