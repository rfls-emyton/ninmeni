"""NMU framework — codec statis (1 char = 1 ID, tanpa UNK).

Model yang dibangun di atas framework ini: **Veyra**.
"""

from .errors import (
    NMUError,
    InvalidInputError,
    PolicyViolationError,
    RegistryMismatchError,
)
from .codec import NMUCodec

__all__ = [
    "NMUCodec",
    "NMUError",
    "InvalidInputError",
    "PolicyViolationError",
    "RegistryMismatchError",
]
