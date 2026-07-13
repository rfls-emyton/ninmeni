"""build_dataset.py — pembangun dataset NMU produksi (1 builder: ingest+encode+split+metadata).

Memperbaiki celah pipeline lama:
- BATAS DOKUMEN TERJAGA: tiap record JSONL = 1 dokumen (BOS+ids+EOS), bukan 1 file raksasa.
- FROZEN SPLIT train/val/test (deterministik via hash konten; reproducible lintas run).
- METADATA SIDECAR: domain/source/trust/safety/level disimpan sejajar per-dokumen.
- DEKONTAMINASI: opsi buang dokumen yang hash-nya ada di set benchmark (--decon-hashes).
- TANPA UNK: record gagal encode -> dikarantina (dicatat), tidak diganti unit cadangan apa pun.

Output: data/shards_id_v2/{train,val,test}/shard_*.{ids,lens}.npy + *.meta.jsonl + manifest.json,
        plus data/shards_id_v2/datacard.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from nmu import NMUCodec, NMUError

FIELD_PRIORITY = ["teks", "sentence", "surface", "text", "content", "body"]


def extract_text(rec: dict, fields):
    for k in fields:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def extract_meta(rec: dict) -> dict:
    src = rec.get("source") or {}
    safety = rec.get("public_safety") or {}
    return {
        "domain": rec.get("domain"),
        "source_type": (src.get("source_type") if isinstance(src, dict) else None) or rec.get("source_type"),
        "trust": src.get("trust_level") if isinstance(src, dict) else None,
        "level": rec.get("curriculum_level"),
        "sensitive": bool(safety.get("sensitive")) if isinstance(safety, dict) else False,
        "label": rec.get("label"),
    }


def iter_jsonl(paths):
    # Pertahankan URUTAN argumen --inputs (sumber prioritas dulu); sort hanya di dalam tiap dir.
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
        elif p.endswith(".jsonl"):
            files.append(p)
    return files


def split_of(text_hash: str) -> str:
    # Deterministik & frozen: bucket dari 2 byte hash. val~2%, test~2%, sisanya train.
    b = int(text_hash[:4], 16) % 50
    if b == 0:
        return "val"
    if b == 1:
        return "test"
    return "train"


class ShardWriter:
    def __init__(self, out: Path, split: str, shard_units: int):
        self.dir = out / split
        self.dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_units = shard_units
        self.ids: list[int] = []
        self.lens: list[int] = []
        self.meta: list[dict] = []
        self.idx = 0
        self.docs = 0
        self.units = 0

    def add(self, ids: list[int], meta: dict):
        self.ids.extend(ids)
        self.lens.append(len(ids))
        self.meta.append(meta)
        self.docs += 1
        self.units += len(ids)
        if len(self.ids) >= self.shard_units:
            self.flush()

    def flush(self):
        if not self.ids:
            return
        np.save(self.dir / f"shard_{self.idx:05d}.ids.npy", np.asarray(self.ids, dtype=np.uint16))
        np.save(self.dir / f"shard_{self.idx:05d}.lens.npy", np.asarray(self.lens, dtype=np.uint32))
        with open(self.dir / f"shard_{self.idx:05d}.meta.jsonl", "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        self.idx += 1
        self.ids, self.lens, self.meta = [], [], []


def run(inputs, out_dir, fields, max_chars, shard_units, decon_hashes):
    codec = NMUCodec.load("registry/nmu_v1.json")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writers = {s: ShardWriter(out, s, shard_units) for s in ("train", "val", "test")}
    seen: set[str] = set()
    decon: set[str] = set()
    if decon_hashes and os.path.exists(decon_hashes):
        decon = {l.strip() for l in open(decon_hashes, encoding="utf-8") if l.strip()}

    stats = {"records": 0, "ok": 0, "dup": 0, "decon": 0, "quarantine": 0, "chars": 0}
    per_source = {}
    stop = False
    for f in iter_jsonl(inputs):
        if stop:
            break
        src = Path(f).parts[-2] if len(Path(f).parts) > 1 else Path(f).stem
        try:
            fh = open(f, encoding="utf-8")
        except OSError as e:
            print(f"[skip] tak bisa buka {f}: {e}")
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                txt = extract_text(rec, fields)
                if not txt:
                    continue
                stats["records"] += 1
                norm = codec.normalize(txt)
                h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
                if h in seen:
                    stats["dup"] += 1
                    continue
                if h in decon:
                    stats["decon"] += 1
                    continue
                try:
                    ids = codec.encode(norm, add_bos=True, add_eos=True)
                except NMUError:
                    # Fallback per-kalimat: simpan kalimat valid, buang hanya yang bermasalah
                    # (retensi data jauh lebih tinggi pada teks beragam; tanpa UNK).
                    good = []
                    for piece in re.split(r"(?<=[.!?])\s+|\n+", norm):
                        piece = piece.strip()
                        if not piece:
                            continue
                        try:
                            good.append(codec.encode(piece))
                        except NMUError:
                            stats["sent_quarantine"] = stats.get("sent_quarantine", 0) + 1
                    if not good:
                        stats["quarantine"] += 1
                        continue
                    ids = [codec.bos_id]
                    for g in good:
                        ids.extend(g)
                    ids.append(codec.eos_id)
                seen.add(h)
                meta = extract_meta(rec)
                meta["src_file"] = src
                writers[split_of(h)].add(ids, meta)
                stats["ok"] += 1
                stats["chars"] += len(norm)
                per_source[src] = per_source.get(src, 0) + 1
                if stats["chars"] >= max_chars:
                    stop = True
                    break

    for w in writers.values():
        w.flush()

    datacard = {
        "version": codec.version, "registry_hash": codec.registry_hash,
        "unicode_version": codec.unicode_version, "vocab_size": codec.vocab_size,
        "granularity": "per-record (kalimat)", "split_rule": "hash%50: 0=val,1=test,else train (frozen)",
        "stats": stats, "per_source": per_source,
        "splits": {s: {"docs": w.docs, "units": w.units, "shards": w.idx} for s, w in writers.items()},
        "decontamination": {"hashes_loaded": len(decon), "removed": stats["decon"]},
    }
    (out / "datacard.json").write_text(json.dumps(datacard, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] dataset v2 -> {out}")
    print(f"     records={stats['records']} ok={stats['ok']} dup={stats['dup']} "
          f"quarantine={stats['quarantine']} decon={stats['decon']} chars={stats['chars']:,}")
    for s, w in writers.items():
        print(f"     {s:5}: docs={w.docs:,} units={w.units:,} shards={w.idx}")
    print(f"     per_source={per_source}")


def main():
    DATA = r"R:\LLM\DATA"
    default_inputs = [
        os.path.join(DATA, "corpus_clean", "asana_realization_corpus_v2.jsonl"),
        os.path.join(DATA, "corpus_clean", "asana_role_speech_pairs.jsonl"),
        os.path.join(DATA, "corpus_clean", "asana_clean_corpus.jsonl"),
        os.path.join(DATA, "curriculum_wikipedia_full"),
    ]
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=default_inputs)
    ap.add_argument("--out", default="data/shards_id_v2")
    ap.add_argument("--fields", nargs="+", default=FIELD_PRIORITY)
    ap.add_argument("--max-chars", type=int, default=40_000_000)
    ap.add_argument("--shard-units", type=int, default=2_000_000)
    ap.add_argument("--decon-hashes", default=None, help="file berisi sha1 teks benchmark (1 per baris) untuk dibuang")
    a = ap.parse_args()
    run(a.inputs, a.out, a.fields, a.max_chars, a.shard_units, a.decon_hashes)


if __name__ == "__main__":
    main()
