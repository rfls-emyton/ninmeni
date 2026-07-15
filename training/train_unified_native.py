"""train_unified_native.py — UNIFIED pretrain + SFT dalam 1 loop, Veyra-NATIVE.

1 loop: tiap step pilih batch berdasarkan probability r:
  - r dari ramp ratio0 -> ratio1 sepanjang training (awal mostly pretrain, akhir bertahap)
  - pretrain batch (seq=cfg.pretrain_seq_len): loss = model.loss(x)            (SHIFTED CE)
  - SFT batch (max_len=cfg.sft_max_len)     : loss = sft_loss(model, xb, yb)  (SHIFTED CE + IGNORE_INDEX mask)

Semua gate native dipatuhi:
  - G6: loss kedua jalur shifted (model.loss & sft_loss identik formula via G10).
  - G8/G9: SFT mask label selaras shift; sentinel handling konsisten.
  - G11: tak ada pattern scaffold di path unified.
  - G12: PAD attention mask auto-aktif kalau SFT batch punya PAD.

Resumable + time-guard + config_hash + opt_kind check (defensive D1-D4 aktif).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nmu import NMUCodec                                                          # noqa: E402
from model.reference_model import VeyraNativeModel, VeyraNativeConfig, count_params  # ganti dgn arsitektur Anda — docs/antarmuka-model.md  # noqa: E402
from training.native_utils import ShardSampler, MultiPoolShardSampler, get_optimizer, save_ckpt, config_hash  # noqa: E402
from training.sft_native import build_example, sft_loss, IGNORE                   # noqa: E402


def load_sft_data(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def make_sft_batch(codec, USER, ASST, THINK, ENDTHINK, data, rng, microbatch, max_len, dev):
    xs, ys = [], []
    tries = 0
    while len(xs) < microbatch and tries < microbatch * 6:
        d = data[rng.randrange(len(data))]; tries += 1
        ex = build_example(codec, USER, ASST, THINK, ENDTHINK, d["turns"], max_len)
        if ex:
            xs.append(ex[0]); ys.append(ex[1])
    if not xs:
        return None, None
    return (torch.tensor(xs, dtype=torch.long, device=dev),
            torch.tensor(ys, dtype=torch.long, device=dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", required=True, help="dir shard pretrain (*.ids.npy)")
    ap.add_argument("--sft", required=True, help="path SFT jsonl (format turns)")
    ap.add_argument("--registry", default="registry/nmu_v1.json")
    ap.add_argument("--out", default="checkpoints/nmu-veyra-native-unified")
    ap.add_argument("--total-steps", type=int, default=200000)
    ap.add_argument("--time-budget", type=int, default=None)
    ap.add_argument("--pretrain-mix", default=None,
                    help="paket-kurikulum W2: path json pools multi-arm "
                         "[{dir,weight,mode}]; bila diisi, --shards diabaikan "
                         "KECUALI muncul sebagai pool di json ini")
    a = ap.parse_args()

    cfg_y = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    mc, tc = cfg_y["model"], cfg_y["train"]
    time_budget = a.time_budget or tc.get("time_budget_sec", 86400)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    codec = NMUCodec.load(a.registry)
    USER, ASST = codec.specials["SENTINEL_4"], codec.specials["SENTINEL_5"]
    THINK, ENDTHINK = codec.specials["SENTINEL_6"], codec.specials["SENTINEL_7"]

    valid = {f.name for f in fields(VeyraNativeConfig)}
    # Kunci config arsitektur/istilah asing (legacy) — DITOLAK, harus literal agar tertangkap.
    forbidden = {"patch_size", "d_global", "d_local", "n_global_layers", "n_local_layers",
                 "window", "max_seq", "max_seq_len", "tie_embeddings", "max_patches",
                 "vocab_size"}
    bad = [k for k in mc if k in forbidden]
    if bad:
        print(f"[warn] config field dilarang (legacy): {bad} — DIBUANG.", flush=True)
    mc_clean = {k: v for k, v in mc.items() if k in valid}
    cfg = VeyraNativeConfig(ukuran_ruang=codec.ukuran_ruang, pad_id=codec.pad_id, **mc_clean)
    model = VeyraNativeModel(cfg).to(dev)
    opt, opt_kind = get_optimizer(model, tc["lr"], tc.get("optimizer", "adam8bit"))
    print(f"Veyra-NATIVE UNIFIED {count_params(model)/1e6:.1f}M  opt={opt_kind}  device={dev}  "
          f"grad_checkpoint={cfg.grad_checkpoint}", flush=True)

    out = Path(a.out)
    start_step = 0
    ckpt = out / "ckpt.pt"
    if ckpt.exists():                                                   # RESUME
        blob = torch.load(ckpt, map_location=dev, weights_only=False)
        if blob["registry_hash"] != codec.registry_hash:
            raise SystemExit("[FAIL] REGISTRY_MISMATCH ckpt vs registry.")
        if blob["config"] != cfg.__dict__:
            diff = {k: (blob["config"].get(k), cfg.__dict__.get(k))
                    for k in set(blob["config"]) | set(cfg.__dict__)
                    if blob["config"].get(k) != cfg.__dict__.get(k)}
            raise SystemExit(f"[FAIL] CONFIG_MISMATCH: {diff}")
        expected_hash = config_hash(cfg.__dict__)
        if blob.get("config_hash") and blob["config_hash"] != expected_hash:
            raise SystemExit(f"[FAIL] CONFIG_HASH_MISMATCH ckpt vs current.")
        model.load_state_dict(blob["model"])
        if blob.get("opt_kind") and blob["opt_kind"] != opt_kind:
            raise SystemExit(f"[FAIL] OPT_KIND_MISMATCH: {blob['opt_kind']} vs {opt_kind}.")
        opt.load_state_dict(blob["optimizer"])
        start_step = blob["step"]
        torch.set_rng_state(blob["rng"].cpu())
        print(f"[RESUME UNIFIED] dari step {start_step}", flush=True)
    else:
        print("[FRESH UNIFIED] mulai dari nol.", flush=True)

    pretrain_seq = int(tc["pretrain_seq_len"])
    sft_max = int(tc["sft_max_len"])
    mb_pre = int(tc["microbatch_pretrain"])
    mb_sft = int(tc["microbatch_sft"])
    grad_accum = int(tc["grad_accum"])
    ratio0 = float(tc["ratio0"]); ratio1 = float(tc["ratio1"])
    warmup = tc["warmup"]

    if a.pretrain_mix:
        pools = json.loads(Path(a.pretrain_mix).read_text(encoding="utf-8"))["pools"]
        sampler = MultiPoolShardSampler(pools, pretrain_seq, seed=start_step)
        print(f"[wave2] multi-pool aktif: "
              + ", ".join(f"{Path(p['dir']).name}={p['weight']}" for p in pools), flush=True)
    else:
        sampler = ShardSampler(a.shards, pretrain_seq, seed=start_step)
    sft_data = load_sft_data(a.sft)
    print(f"shards={len(sampler.files)} pretrain_units={sampler.total_units/1e9:.2f}B "
          f"seq_pre={pretrain_seq} mb_pre={mb_pre} | sft={len(sft_data)} sft_max={sft_max} "
          f"mb_sft={mb_sft} | grad_accum={grad_accum} ratio {ratio0}->{ratio1}", flush=True)

    total = a.total_steps
    def lr_at(s):
        if s < warmup:
            return tc["lr"] * s / max(1, warmup)
        p = (s - warmup) / max(1, total - warmup)
        return 0.1 * tc["lr"] + 0.5 * 0.9 * tc["lr"] * (1 + math.cos(math.pi * p))

    def ratio_at(s):
        return ratio0 + (ratio1 - ratio0) * min(1.0, s / max(1, total))

    amp = dev == "cuda"
    ln2 = math.log(2)
    model.train()
    t0 = time.time()
    step = start_step
    rng = random.Random(start_step * 7919 + 13)
    nP = nS = 0
    while step < total:
        if time.time() - t0 > time_budget:
            print(f"[TIME-GUARD] {time_budget}s tercapai pada step {step} — simpan & keluar.", flush=True)
            save_ckpt(out, model, opt, step, cfg.__dict__, codec, total, opt_kind)
            return
        step += 1
        r = ratio_at(step)
        is_sft = rng.random() < r
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        micro = 0.0
        n_ok_micro = 0                                                # Perbaikan internal
        for _ in range(grad_accum):
            if is_sft:
                xb, yb = make_sft_batch(codec, USER, ASST, THINK, ENDTHINK, sft_data,
                                         rng, mb_sft, sft_max, dev)
                if xb is None: continue
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = sft_loss(model, xb, yb) / grad_accum
            else:
                xb = sampler.batch(mb_pre, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = model.loss(xb) / grad_accum
            loss.backward()
            micro += loss.item()
            n_ok_micro += 1                                           # Perbaikan internal F8
        # Perbaikan internal — kalau semua micro-step continue (jarang tapi mungkin
        # untuk SFT batch=None seluruhnya), skip opt.step() supaya Adam m/v tidak
        # drift dari grad=0 spurious update. Existing finite check tidak catch
        # kasus ini karena micro=0.0 IS finite dan gn=0.0 IS finite.
        if n_ok_micro == 0:
            continue
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not (math.isfinite(micro) and torch.isfinite(gn)):
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        # REVERT perbaikan internal zeroing (SPEC-F6-NATIVE-v1, sesi review 2026-07-07):
        # Zeroing melawan state terpelajar (63K step supresi logit PAD; PAD row
        # norm 2.256 = fungsional-benign, bukan drift bug). Ganti dengan eksklusi
        # PAD dari ruang prediksi via mask -inf di forward() head + _surprisal_from()
        # (lihat model/reference_model.py) — memformalkan state terpelajar jadi
        # invarian arsitektural permanen + hentikan drift secara struktural
        # (gradien denominator ke PAD row berhenti total by construction).
        # Kontinuitas terverifikasi CPU pra-deploy: |delta loss shifted| = 5.96e-08 nats.
        if is_sft: nS += 1
        else: nP += 1
        if step % 25 == 0:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
            tag = "S" if is_sft else "P"
            bpc_or_loss = micro / ln2 if not is_sft else micro
            label = "bpc" if not is_sft else "loss"
            print(f"step {step}/{total} [{tag}] {label} {bpc_or_loss:.3f} gnorm {gn.item():.2f} "
                  f"r {r:.2f} lr {lr_at(step):.2e} P/S {nP}/{nS} "
                  f"{el/max(1,step-start_step):.1f}s/step vram {mem:.1f}GB", flush=True)
        if step % tc["ckpt_every"] == 0:
            save_ckpt(out, model, opt, step, cfg.__dict__, codec, total, opt_kind)

    save_ckpt(out, model, opt, step, cfg.__dict__, codec, total, opt_kind)
    print(f"[DONE UNIFIED] total {total} -> {out}/ckpt.pt  | final P/S = {nP}/{nS}", flush=True)


if __name__ == "__main__":
    main()
