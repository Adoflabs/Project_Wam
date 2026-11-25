"""Deterministic UTF-8 encoding for model input sequences."""

BYTE_PAD_ID = 0
BYTE_VOCAB_SIZE = 257


def encode_bytes(text: str | None) -> list[int]:
    """Returns UTF-8 byte IDs with zero reserved for padding."""
    if text is None:
        return []

    return [byte + 1 for byte in text.encode('utf-8')]
