"""native_utils.py — utility netral untuk training Veyra-native.

Berisi: ShardSampler (mmap streaming), get_optimizer (adam8bit + fallback AdamW),
save_ckpt (atomic, Windows-lock-safe). TIDAK bergantung pada arsitektur tertentu —
input/output shape-agnostic; hanya bicara tensor, codec, dan path.
"""

from __future__ import annotations

import glob
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch


def config_hash(cfg_dict: dict) -> str:
    """Hash schema config (semua key+value) -> deteksi stale ckpt setelah skema berevolusi."""
    return hashlib.md5(str(sorted(cfg_dict.items())).encode()).hexdigest()


def get_optimizer(model, lr, kind="adam8bit"):
    """Adam 8-bit (bitsandbytes) -> hemat ~9GB vs AdamW fp32; fallback AdamW jika tak tersedia."""
    if kind == "adam8bit":
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.Adam8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
            return opt, "adam8bit"
        except Exception as e:
            print(f"[warn] bitsandbytes tak tersedia ({e}); fallback AdamW (butuh VRAM lebih).", flush=True)
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1), "adamw"


class ShardSampler:
    """Sampler streaming via mmap — sampel window acak dari shard tanpa load ke RAM.
    Wajib untuk dataset besar (mis. 11,5B unit = 92GB bila di-RAM).
    Shape-agnostic: hanya keluarkan tensor [n, seq_len] long.
    """

    def __init__(self, shards_dir, seq_len, seed=0):
        self.files = sorted(glob.glob(str(Path(shards_dir) / "*.ids.npy")))
        if not self.files:
            raise SystemExit(f"tak ada shard *.ids.npy di {shards_dir}")
        self.arrs = [np.load(f, mmap_mode="r") for f in self.files]
        self.seq_len = seq_len
        valid = np.array([max(0, len(a) - seq_len) for a in self.arrs], dtype=np.float64)
        if valid.sum() <= 0:
            raise SystemExit(f"tak ada shard yang panjangnya cukup utk seq_len={seq_len}")
        self.prob = valid / valid.sum()
        self.rng = np.random.RandomState(seed)
        self.total_units = int(sum(len(a) for a in self.arrs))

    def batch(self, n, dev):
        out = np.empty((n, self.seq_len), dtype=np.int64)
        shs = self.rng.choice(len(self.arrs), size=n, p=self.prob)
        for i, s in enumerate(shs):
            a = self.arrs[s]
            start = self.rng.randint(0, len(a) - self.seq_len + 1)
            out[i] = a[start:start + self.seq_len]
        return torch.from_numpy(out).to(dev, non_blocking=True)


class MultiPoolShardSampler:
    """Sampler multi-pool paket-kurikulum Gelombang-2 (GAP-2/3/4) — additive, tidak
    mengubah ShardSampler existing.

    pools: list of {"dir": str, "weight": float, "mode": "length"|"record_w"}
      - mode "length"  : sampel window ~ panjang shard (perilaku existing;
                          distribusi natural = otoritas substrat).
      - mode "record_w": sampel RECORD ~ w (sidecar shard_XXXXX.recs.npy
                          [start,len] + .recw.npy float32), lalu window di
                          dalam/berawal dari record. Ini wiring Interpretasi B:
                          sum(w) per entitas = 1 -> ekspektasi sampel per
                          entitas setara, tidak ditelan panjang teks.
    weight antar-pool = proporsi sampel per batch (dinormalisasi).
    Shape-agnostic: keluaran [n, seq_len] long, kompatibel penuh dengan
    trainer existing (atribut .files + .total_units disediakan).
    """

    def __init__(self, pools, seq_len, seed=0):
        self.seq_len = seq_len
        self.rng = np.random.RandomState(seed)
        self.pools = []
        self.files = []
        self.total_units = 0
        for p in pools:
            files = sorted(glob.glob(str(Path(p["dir"]) / "*.ids.npy")))
            if not files:
                raise SystemExit(f"tak ada shard *.ids.npy di {p['dir']}")
            arrs = [np.load(f, mmap_mode="r") for f in files]
            entry = {"mode": p.get("mode", "length"), "arrs": arrs,
                     "weight": float(p["weight"]), "dir": p["dir"]}
            if entry["mode"] == "record_w":
                recs, recw = [], []
                for f in files:
                    base = f[: -len(".ids.npy")]
                    recs.append(np.load(base + ".recs.npy"))
                    recw.append(np.load(base + ".recw.npy").astype(np.float64))
                entry["recs"] = recs
                sw = np.array([w.sum() for w in recw], dtype=np.float64)
                entry["shard_prob"] = sw / sw.sum()
                entry["rec_prob"] = [w / w.sum() for w in recw]
            else:
                valid = np.array([max(0, len(a) - seq_len) for a in arrs], dtype=np.float64)
                if valid.sum() <= 0:
                    raise SystemExit(f"shard terlalu pendek utk seq_len={seq_len} di {p['dir']}")
                entry["shard_prob"] = valid / valid.sum()
            self.pools.append(entry)
            self.files.extend(files)
            self.total_units += int(sum(len(a) for a in arrs))
        w = np.array([p["weight"] for p in self.pools], dtype=np.float64)
        self.pool_prob = w / w.sum()

    def _sample_one(self, pool):
        s = self.rng.choice(len(pool["arrs"]), p=pool["shard_prob"])
        a = pool["arrs"][s]
        if pool["mode"] == "record_w":
            r = self.rng.choice(len(pool["rec_prob"][s]), p=pool["rec_prob"][s])
            start, rlen = pool["recs"][s][r]
            # window berawal di dalam record; record pendek -> lanjut ke stream
            # berikutnya (lintasan konteks natural), clamp ke batas shard.
            hi = max(1, rlen - self.seq_len + 1)
            st = int(start + self.rng.randint(0, hi))
            st = min(st, len(a) - self.seq_len)
            st = max(st, 0)
            return a[st: st + self.seq_len]
        start = self.rng.randint(0, len(a) - self.seq_len + 1)
        return a[start: start + self.seq_len]

    def batch(self, n, dev):
        out = np.empty((n, self.seq_len), dtype=np.int64)
        ps = self.rng.choice(len(self.pools), size=n, p=self.pool_prob)
        for i, pi in enumerate(ps):
            out[i] = self._sample_one(self.pools[pi])
        return torch.from_numpy(out).to(dev, non_blocking=True)


def save_ckpt(path: Path, model, opt, step, cfg_dict, codec, total_steps, opt_kind):
    """Checkpoint atomic (tmp -> rename), simpan registry_hash (pin dataset<->model).
    Windows-lock-safe: retry os.replace s.d. 20× kalau ckpt.pt dikunci pembaca lain;
    kalau tetap gagal, lewati simpan (JANGAN crash training).

    Patch perbaikan internal — setelah os.replace ckpt.pt sukses, tulis sidecar step.txt
    atomic (tmp->rename) berisi step angka literal. Daemon baca step.txt O(1)
    tanpa torch.load full blob (1-2 GB) tiap 30 detik -> tutup race window
    baca-tulis ckpt.
    Patch perbaikan internal — broaden except PermissionError -> except OSError
    (cover WinError 5/32/33 varian yang tidak PermissionError subclass).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "config": cfg_dict,
        "config_hash": config_hash(cfg_dict),    # D3: deteksi schema drift saat resume
        "registry_hash": codec.registry_hash,
        "vocab_version": codec.version,
        "total_steps": total_steps,
        "opt_kind": opt_kind,
        "rng": torch.get_rng_state(),
    }
    tmp = path / "ckpt.pt.tmp"
    torch.save(blob, tmp)
    final = path / "ckpt.pt"
    replaced = False
    for _ in range(20):
        try:
            os.replace(tmp, final)
            replaced = True
            break
        except OSError:
            time.sleep(0.5)
    if not replaced:
        print(f"[warn] save_ckpt step {step}: ckpt.pt terkunci, lewati simpan (akan disimpan lagi nanti).", flush=True)
        return
    step_tmp = path / "step.txt.tmp"
    step_final = path / "step.txt"
    try:
        with open(step_tmp, "w", encoding="ascii") as f:
            f.write(str(step))
        for _ in range(20):
            try:
                os.replace(step_tmp, step_final)
                break
            except OSError:
                time.sleep(0.1)
    except OSError as e:
        print(f"[warn] sidecar step.txt @ step {step} gagal ditulis: {e}", flush=True)
