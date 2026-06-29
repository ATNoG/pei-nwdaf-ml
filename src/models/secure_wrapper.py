import torch
import torch.nn as nn
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat


class SecureWrapper(nn.Module):
    """Binds an X25519 keypair to any nn.Module.

    Private key is a plain attribute — persists through mlflow.pytorch.log_model
    (full pickle). Only the holder of the saved model artifact can decrypt.

    Ciphertext layout expected by decrypt():
        ephemeral_pub (32 bytes) | nonce (12 bytes) | AES-256-GCM ciphertext
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.priv_key: bytes = X25519PrivateKey.generate().private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)

    @property
    def public_key(self) -> bytes:
        return X25519PrivateKey.from_private_bytes(self.priv_key).public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )

    def decrypt(self, ciphertext: bytes) -> bytes:
        return ecies_decrypt(ciphertext, self.priv_key)


def ecies_decrypt(ciphertext: bytes, priv_key_bytes: bytes) -> bytes:
    """Standalone ECIES decrypt. Same algorithm as SecureWrapper.decrypt."""
    private_key = X25519PrivateKey.from_private_bytes(priv_key_bytes)
    ephemeral_pub = X25519PublicKey.from_public_bytes(ciphertext[:32])
    shared_secret = private_key.exchange(ephemeral_pub)
    aes_key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"model-decrypt"
    ).derive(shared_secret)
    return AESGCM(aes_key).decrypt(ciphertext[32:44], ciphertext[44:], None)
