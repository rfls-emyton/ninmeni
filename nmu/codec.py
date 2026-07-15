"""NMUCodec — pemetaan statis 1 karakter = 1 ID, deterministik, tanpa UNK.

Invarian: decode(encode(x)) == NFC(normalize_newlines(x)) untuk semua x dalam ruang NMU.
Kegagalan mapping -> error berkategori (lihat errors.py), BUKAN UNK.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

from .errors import InvalidInputError, PolicyViolationError, RegistryMismatchError

# Control char yang DIIZINKAN sebagai unit (punya ID sendiri). CR dinormalisasi -> LF
# di normalize(), jadi tidak perlu ID sendiri.
_ALLOWED_CONTROL = {0x09, 0x0A}  # TAB, LF

# G19b (SPEC-SENTINEL-v1, tinjauan internal 2026-07-07): specials 0..15 adalah
# kontrak arsitektural beku permanen (BUKAN ukuran_ruang — ruang karakter registry hanya boleh tumbuh append-only ANTAR-GENERASI model; di dalam masa hidup satu model terlatih, ruangnya beku). Assert konstanta ini di setiap NMUCodec
# construction — fail-fast kalau registry mencoba mengubah sentinel IDs.
_SPECIALS_FROZEN = {
    "PAD": 0, "BOS": 1, "EOS": 2, "MASK": 3,
    "SENTINEL_4": 4, "SENTINEL_5": 5, "SENTINEL_6": 6, "SENTINEL_7": 7,
    "SENTINEL_8": 8, "SENTINEL_9": 9, "SENTINEL_10": 10, "SENTINEL_11": 11,
    "SENTINEL_12": 12, "SENTINEL_13": 13, "SENTINEL_14": 14, "SENTINEL_15": 15,
}


class NMUCodec:
    def __init__(self, specials: dict[str, int], codepoints: list[int],
                 version: str, unicode_version: str, registry_hash: str):
        # G19b: specials 0..15 frozen constant. Assert di construction supaya
        # SETIAP codec (via load() atau direct construct) lulus gate arsitektural.
        for name, expected_id in _SPECIALS_FROZEN.items():
            if name not in specials:
                raise RegistryMismatchError(
                    f"[G19b] specials pelanggaran: {name} hilang dari registry. "
                    f"Specials 0..15 = kontrak arsitektural beku permanen; semua "
                    f"16 nama WAJIB ada dengan ID sesuai konstanta."
                )
            if specials[name] != expected_id:
                raise RegistryMismatchError(
                    f"[G19b] specials pelanggaran: {name} expected id={expected_id}, "
                    f"actual={specials[name]}. Specials 0..15 = frozen constant."
                )

        self.version = version
        self.unicode_version = unicode_version
        self.registry_hash = registry_hash
        self.specials = dict(specials)

        # id -> char (specials decode ke '' agar tak muncul saat round-trip teks murni)
        max_special_id = max(specials.values())
        size = max_special_id + 1 + len(codepoints)
        self._id_to_char: list[str] = [""] * size
        self._char_to_id: dict[int, int] = {}  # codepoint -> id

        base = max_special_id + 1
        for idx, cp in enumerate(codepoints):
            cid = base + idx
            self._id_to_char[cid] = chr(cp)
            self._char_to_id[cp] = cid

        self.ukuran_ruang = size  # ukuran ruang karakter registry (jumlah ID)
        self.pad_id = specials["PAD"]
        self.bos_id = specials["BOS"]
        self.eos_id = specials["EOS"]

    # ---------- konstruksi ----------
    @classmethod
    def load(cls, path: str | Path) -> "NMUCodec":
        path = Path(path)
        raw = path.read_bytes()
        registry_hash = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw.decode("utf-8"))

        # Verifikasi sidecar hash bila ada (blocking di produksi).
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.exists():
            expected = sidecar.read_text(encoding="utf-8").split()[0].strip()
            if expected != registry_hash:
                raise RegistryMismatchError(
                    f"hash registry {registry_hash[:12]} != sidecar {expected[:12]}"
                )
        return cls(
            specials=data["specials"],
            codepoints=data["codepoints"],
            version=data["version"],
            unicode_version=data["unicode_version"],
            registry_hash=registry_hash,
        )

    # ---------- normalisasi ----------
    @staticmethod
    def normalize(text: str) -> str:
        # Line-ending -> LF (sekali), lalu NFC. JANGAN sentuh pemisah angka locale.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return unicodedata.normalize("NFC", text)

    # ---------- encode / decode ----------
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        text = self.normalize(text)
        out: list[int] = [self.bos_id] if add_bos else []
        for ch in text:  # iterasi per code point (Python str = code points)
            cp = ord(ch)
            cid = self._char_to_id.get(cp)
            if cid is not None:
                out.append(cid)
                continue
            # Klasifikasikan kegagalan — TIDAK ADA UNK.
            cat = unicodedata.category(ch)
            if cat == "Cs":
                raise InvalidInputError("surrogate telanjang", char=ch)
            if cat == "Cn":
                raise InvalidInputError("code point unassigned", char=ch)
            if cat == "Cc" and cp not in _ALLOWED_CONTROL:
                raise InvalidInputError("control char tak diizinkan", char=ch)
            # Valid Unicode tapi di luar ruang NMU terkurasi.
            raise PolicyViolationError("unit di luar ruang NMU (curated)", char=ch)
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids: list[int]) -> str:
        n = self.ukuran_ruang
        parts = []
        for i in ids:
            if i < 0 or i >= n:
                raise InvalidInputError(f"id {i} di luar [0,{n})")
            parts.append(self._id_to_char[i])
        return "".join(parts)

    # ---------- util ----------
    def round_trip_ok(self, text: str) -> bool:
        return self.decode(self.encode(text)) == self.normalize(text)
