"""Error berkategori untuk NMU. TIDAK ADA UNK — setiap kegagalan mapping
diklasifikasikan eksplisit (sesuai UNICODE_POLICY.md & PRODUCTION.md)."""

from __future__ import annotations


class NMUError(Exception):
    """Basis error NMU dengan kode kategori."""

    code = "NMU_ERROR"

    def __init__(self, message: str, *, char: str | None = None, codepoint: int | None = None):
        self.char = char
        self.codepoint = codepoint
        detail = ""
        if char is not None:
            cp = codepoint if codepoint is not None else ord(char)
            detail = f" char={char!r} U+{cp:04X}"
        super().__init__(f"[{self.code}] {message}{detail}")


class InvalidInputError(NMUError):
    """Byte rusak, unassigned, noncharacter, surrogate telanjang, control char tak diizinkan."""

    code = "INVALID_INPUT"


class PolicyViolationError(NMUError):
    """Unit valid Unicode tetapi di luar ruang NMU yang diakui policy (curated mode)."""

    code = "POLICY_VIOLATION"


class RegistryMismatchError(NMUError):
    """Hash registry tidak cocok antar lingkungan (data/training/serving)."""

    code = "REGISTRY_MISMATCH"
