"""sft_native.py — SFT (pelatihan terarah pembentuk protokol dialog) di atas arsitektur Veyra-NATIVE.

KONTRAK:
  1. Loss WAJIB causal-LM SHIFTED — sama seperti pretrain native. Tanpa shift,
     head bisa identity-cheat (B1 audit-2 yg muncul lagi di SFT scaffold).
  2. Masking labels selaras dgn target shifted — labels dibangun positional
     paralel dgn input, lalu di-shift bersama input di loss. IGNORE di posisi i
     berarti TARGET di posisi i (= input[i]) tidak dilatih. Karena shift, logits[t-1]
     yang seharusnya memprediksi input[t] otomatis di-skip kalau labels[t]=IGNORE.
  3. Sentinel handling:
       USER, ASST  → label IGNORE (sentinel pemisah, tak dilatih predict)
       THINK, ENDTHINK → label = sentinel-id (TRAINED: model harus emit sendiri)
       User content, BOS, PAD → IGNORE
       Assistant content + EOS → TRAINED
  4. Gate by-entropi (NMUCell) berjalan APA ADANYA. Tidak ada intervensi khusus
     pada surprisal di sentinel positions. Konsisten prinsip 3.
  5. Tidak ada segmentasi perantara, tidak ada jendela geser, tidak ada batas
     keras panjang sekuens. max_len SFT adalah cap PADDING (contoh > max_len
     dibuang), BUKAN constraint kelipatan apa pun.
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
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nmu import NMUCodec                                          # noqa: E402
from model.reference_model import VeyraNativeModel, VeyraNativeConfig, count_params  # noqa: E402 — ganti dgn arsitektur Anda
from training.native_utils import get_optimizer, save_ckpt, config_hash  # noqa: E402

IGNORE = -100                                                      # standard PyTorch CE ignore_index
_VALID_ROLES = {"user", "assistant"}                               # D2: hanya 2 role yg diizinkan

# ============================================================
# SPEC-subst-sentinel-NATIVE-v1 (tinjauan internal 2026-07-08) — tool placeholder → sentinel single-ID
# ============================================================
# Design doc design/v6_tool_code_protocol.md §2.3.3 L189-191 menjanjikan substitusi
# placeholder tekstual → sentinel single-ID di dataset akhir. subst-sentinel audit menemukan
# substitusi TIDAK diimplementasi — tool corpus 3,886 sample (24% v6_final) memakai
# teks literal '<S8>' yang encode ke 4 char IDs [46,69,42,48] bukan single ID 8.
#
# Fix (SPEC-subst-sentinel-NATIVE-v1):
# - Substitusi HANYA di ASSISTANT content (termasuk think). USER content teks apa
#   adanya sebagai anti-spoofing — user tidak boleh bisa "menghasilkan sentinel"
#   via teks yang mereka tulis.
# - Sentinel tool = TRAINED (labels = sentinel_id) mengikuti pola THINK/ENDTHINK.
# - JSONL tetap human-readable — satu titik kebenaran di loader ini, bukan
#   rewrite dataset.
import re as _re

_TOOL_PLACEHOLDER_MAP = {
    "8": "SENTINEL_8",   # TOOL_CALL open
    "9": "SENTINEL_9",   # TOOL_CALL close (end of call)
    "10": "SENTINEL_10", # TOOL_RESULT open
    "11": "SENTINEL_11", # TOOL_RESULT close
}
_TOOL_PATTERN = _re.compile(r"<S(8|9|10|11)>")


def _encode_assistant_content(content: str, codec):
    """Encode assistant content dengan substitusi <S8..S11> → SENTINEL_8..11 single-ID.

    Substitusi HANYA berlaku untuk assistant content (build_example L79-87). User content
    (L76-78) tetap `codec.encode(text)` mentah — anti-spoofing per SPEC-subst-sentinel-NATIVE-v1.

    Returns:
        (ids, labels) — parallel lists. Semua labels = TRAINED (=ids untuk chars,
        =sentinel_id untuk placeholder yang ter-substitusi). Model belajar emit
        sentinel ID-tunggal, mengikuti pola THINK/ENDTHINK di build_example L84-86.
    """
    ids = []
    labels = []
    pos = 0
    for m in _TOOL_PATTERN.finditer(content):
        # Chunk teks sebelum placeholder
        if m.start() > pos:
            chunk = codec.encode(content[pos:m.start()])
            ids.extend(chunk)
            labels.extend(chunk)  # TRAINED regular content
        # Substitusi placeholder → sentinel single-ID
        sentinel_name = _TOOL_PLACEHOLDER_MAP[m.group(1)]
        sentinel_id = codec.specials[sentinel_name]
        ids.append(sentinel_id)
        labels.append(sentinel_id)  # TRAINED sentinel emit
        pos = m.end()
    # Trailing text
    if pos < len(content):
        chunk = codec.encode(content[pos:])
        ids.extend(chunk)
        labels.extend(chunk)
    return ids, labels


# ============================================================
# build_example — label positional yang otomatis selaras saat shift.
# ============================================================
def build_example(codec, USER, ASST, THINK, ENDTHINK, turns, max_len):
    """Bangun (seq, labels) length=max_len, padded dgn PAD/IGNORE.

    Posisi (sebagai TARGET, dipakai oleh logits[t-1] setelah shift):
      [BOS]           IGNORE  (BOS tak dilatih)
      [USER]          IGNORE  (sentinel tak dilatih)
      user content    IGNORE  (prompt tak dilatih)
      [ASST]          IGNORE  (batas asst diberi saat inference)
      [THINK]         THINK   (TRAINED: model emit sendiri)
      think content   = ids   (TRAINED: reasoning content)
      [/THINK]        ENDTHINK (TRAINED: model tutup sendiri)
      asst content    = ids   (TRAINED: jawaban final)
      [EOS]           EOS     (TRAINED: model belajar berhenti)
      [PAD] sisa      IGNORE  (tail padding tak dilatih)

    Return None jika seq > max_len (drop, jangan potong di tengah morfem).
    """
    seq, labels = [codec.bos_id], [IGNORE]
    for t in turns:
        role = t["role"]
        if role not in _VALID_ROLES:                               # D2: fail-fast utk role tak dikenal
            raise ValueError(
                f"build_example: role {role!r} tidak dikenal. "
                f"Harus salah satu dari {_VALID_ROLES} — turn lain tak boleh diam-diam "
                f"diperlakukan sbg assistant (sentinel handling violation)."
            )
        if role == "user":
            # SPEC-subst-sentinel: user content TETAP raw (anti-spoofing) — <S8> jika muncul di user
            # akan tetap encode ke 4 char IDs, bukan sentinel single-ID. Ini SENGAJA:
            # user tidak boleh bisa "menghasilkan sentinel" via teks yang mereka tulis.
            ids = codec.encode(t["content"])
            seq.append(USER); labels.append(IGNORE)
            seq += ids; labels += [IGNORE] * len(ids)
        else:                                                      # role == "assistant" (terverifikasi)
            seq.append(ASST); labels.append(IGNORE)
            think = t.get("think")
            if think and THINK is not None and ENDTHINK is not None:
                # SPEC-subst-sentinel: think content = assistant content, apply substitusi
                tid, tlabels = _encode_assistant_content(think, codec)
                seq.append(THINK); labels.append(THINK)           # TRAINED: buka think
                seq += tid; labels += tlabels                       # TRAINED: reasoning (sentinel-substituted)
                seq.append(ENDTHINK); labels.append(ENDTHINK)     # TRAINED: tutup think
            # SPEC-subst-sentinel-NATIVE-v1: assistant content substitusi <S8..S11> → SENTINEL_8..11
            aids, alabels = _encode_assistant_content(t["content"], codec)
            seq += aids; labels += alabels                          # TRAINED: jawaban (sentinel-substituted)
    seq.append(codec.eos_id); labels.append(codec.eos_id)           # TRAINED: EOS
    if len(seq) > max_len:
        return None
    pad = max_len - len(seq)
    return seq + [codec.pad_id] * pad, labels + [IGNORE] * pad


def load_turns(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def pack_multi_doc(codec, USER, ASST, THINK, ENDTHINK, docs, max_len, *, sep_pads=2):
    """Multi-doc packing: gabungkan beberapa turns ke satu sequence dgn PAD pemisah.

    Format: [doc1_seq] PAD*sep_pads [doc2_seq] PAD*sep_pads [doc3_seq] ... PAD-tail
    Untuk SETIAP doc: jalankan build_example logic (label assistant-only).
    PAD pemisah di-mask attention oleh VeyraNativeModel._build_attn_mask.

    Return (seq, labels) dgn shape (max_len,). None kalau gabungan > max_len.
    """
    seq, labels = [], []
    for doc_idx, doc_turns in enumerate(docs):
        if doc_idx > 0:
            seq += [codec.pad_id] * sep_pads
            labels += [IGNORE] * sep_pads
        # build per dokumen (re-use logic build_example tanpa tail pad)
        seq.append(codec.bos_id); labels.append(IGNORE)
        for t in doc_turns:
            role = t["role"]
            if role not in _VALID_ROLES:
                raise ValueError(f"pack_multi_doc: role {role!r} tak dikenal")
            if role == "user":
                # SPEC-subst-sentinel: user content raw (anti-spoofing)
                ids = codec.encode(t["content"])
                seq.append(USER); labels.append(IGNORE)
                seq += ids; labels += [IGNORE] * len(ids)
            else:
                seq.append(ASST); labels.append(IGNORE)
                think = t.get("think")
                if think and THINK is not None and ENDTHINK is not None:
                    tid, tlabels = _encode_assistant_content(think, codec)
                    seq.append(THINK); labels.append(THINK)
                    seq += tid; labels += tlabels
                    seq.append(ENDTHINK); labels.append(ENDTHINK)
                aids, alabels = _encode_assistant_content(t["content"], codec)
                seq += aids; labels += alabels
        seq.append(codec.eos_id); labels.append(codec.eos_id)
    if len(seq) > max_len:
        return None
    pad = max_len - len(seq)
    return seq + [codec.pad_id] * pad, labels + [IGNORE] * pad


def sft_batch(codec, USER, ASST, THINK, ENDTHINK, data, idxs, max_len, dev):
    xs, ys = [], []
    for i in idxs:
        ex = build_example(codec, USER, ASST, THINK, ENDTHINK, data[i]["turns"], max_len)
        if ex:
            xs.append(ex[0]); ys.append(ex[1])
    if not xs:
        return None, None
    return (torch.tensor(xs, dtype=torch.long, device=dev),
            torch.tensor(ys, dtype=torch.long, device=dev))


# ============================================================
# sft_loss — causal-LM SHIFTED CE dgn IGNORE_INDEX masking.
#   labels dibangun positional paralel seq → shift bersama input.
#   IGNORE di posisi i otomatis menonaktifkan loss utk prediksi target i.
# ============================================================
def sft_loss(model, input_ids, labels):
    """Loss SFT: CE shifted dgn IGNORE_INDEX masking selaras.

    logits[B,T,V] = model(input_ids[B,T])
    Predict next-char: logits[:, :-1, :] vs target = labels[:, 1:]
    labels[i] = IGNORE  -> tidak dilatih (mask di target shifted = labels[t+1])
    labels[i] = ID      -> dilatih memprediksi ID = input_ids[i]
    """
    logits = model(input_ids)                                      # [B,T,V]
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, model.cfg.ukuran_ruang),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE,
    )


# ============================================================
# Main training loop — resumable + time-guard utk Kaggle T4.
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_native.yaml")
    ap.add_argument("--base-ckpt", required=True, help="pretrain ckpt utk init bobot SFT")
    ap.add_argument("--train", required=True, help="path SFT train jsonl (format turns)")
    ap.add_argument("--val", default=None, help="path SFT val jsonl (opsional)")
    ap.add_argument("--registry", default="registry/nmu_v1.json")
    ap.add_argument("--out", default="checkpoints/nmu-veyra-native-sft")
    ap.add_argument("--total-steps", type=int, default=20000)
    ap.add_argument("--time-budget", type=int, default=None)
    a = ap.parse_args()

    cfg_y = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    tc = cfg_y["train"]
    time_budget = a.time_budget or tc.get("time_budget_sec", 30600)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    codec = NMUCodec.load(a.registry)
    USER = codec.specials["SENTINEL_4"]
    ASST = codec.specials["SENTINEL_5"]
    THINK = codec.specials["SENTINEL_6"]
    ENDTHINK = codec.specials["SENTINEL_7"]
    print(f"sentinels: USER={USER} ASST={ASST} THINK={THINK} ENDTHINK={ENDTHINK}", flush=True)

    # Load pretrain ckpt -> ambil config + state_dict
    base_blob = torch.load(a.base_ckpt, map_location=dev, weights_only=False)
    if base_blob["registry_hash"] != codec.registry_hash:
        raise SystemExit("[FAIL] REGISTRY_MISMATCH base-ckpt vs registry.")
    base_cfg = base_blob["config"]
    valid_fields = {f.name for f in fields(VeyraNativeConfig)}
    # Kunci config arsitektur/istilah asing (legacy) — DITOLAK, harus literal agar tertangkap.
    forbidden = {"patch_size", "d_global", "d_local", "n_global_layers", "n_local_layers",
                 "window", "max_seq", "max_seq_len", "tie_embeddings", "max_patches",
                 "vocab_size"}
    bad = [k for k in base_cfg if k in forbidden]
    if bad:
        raise SystemExit(f"[FAIL] base-ckpt mengandung field scaffold: {bad}. "
                         f"Bukan ckpt veyra-native.")
    cfg_clean = {k: v for k, v in base_cfg.items() if k in valid_fields}
    cfg = VeyraNativeConfig(**cfg_clean)
    model = VeyraNativeModel(cfg).to(dev)
    model.load_state_dict(base_blob["model"])
    print(f"Veyra-NATIVE {count_params(model)/1e6:.1f}M  loaded pretrain ckpt step={base_blob['step']}",
          flush=True)

    opt, opt_kind = get_optimizer(model, tc["lr"], tc.get("optimizer", "adam8bit"))

    out = Path(a.out)
    start_step = 0
    ckpt = out / "ckpt.pt"
    if ckpt.exists():                                               # RESUME SFT
        blob = torch.load(ckpt, map_location=dev, weights_only=False)
        if blob["registry_hash"] != codec.registry_hash:
            raise SystemExit("[FAIL] REGISTRY_MISMATCH SFT ckpt vs registry.")
        # D1: cek keras config konsisten (cegah shape mismatch / silent wrong weights)
        if blob["config"] != cfg.__dict__:
            diff = {k: (blob["config"].get(k), cfg.__dict__.get(k))
                    for k in set(blob["config"]) | set(cfg.__dict__)
                    if blob["config"].get(k) != cfg.__dict__.get(k)}
            raise SystemExit(f"[FAIL] CONFIG_MISMATCH SFT ckpt vs base-ckpt arsitektur: {diff}")
        # D3: cek schema/version hash (deteksi VeyraNativeConfig stale)
        expected_hash = config_hash(cfg.__dict__)
        if blob.get("config_hash") and blob["config_hash"] != expected_hash:
            raise SystemExit(
                f"[FAIL] CONFIG_HASH_MISMATCH: ckpt={blob['config_hash'][:8]} "
                f"current={expected_hash[:8]}. Schema VeyraNativeConfig mungkin berubah; "
                f"retrain atau migrasi ckpt manual."
            )
        model.load_state_dict(blob["model"])
        # D4: cek opt_kind konsisten sebelum load opt.state_dict
        if blob.get("opt_kind") and blob["opt_kind"] != opt_kind:
            raise SystemExit(
                f"[FAIL] OPT_KIND_MISMATCH: ckpt={blob['opt_kind']} current={opt_kind}. "
                f"adam8bit vs adamw punya state_dict yang berbeda; ganti config optimizer "
                f"kembali atau restart dari base-ckpt."
            )
        opt.load_state_dict(blob["optimizer"])
        start_step = blob["step"]
        torch.set_rng_state(blob["rng"].cpu())
        print(f"[RESUME SFT] dari step {start_step} (config OK, hash OK, opt_kind OK)", flush=True)
    else:
        print("[FRESH SFT] mulai dari pretrain ckpt + opt reset.", flush=True)

    train_data = load_turns(a.train)
    val_data = load_turns(a.val) if a.val else None
    print(f"SFT data: train={len(train_data)} val={len(val_data) if val_data else 0}", flush=True)

    total = a.total_steps
    warmup = tc["warmup"]
    max_len = int(tc["max_len"])
    microbatch = int(tc["microbatch"])
    grad_accum = int(tc["grad_accum"])

    def lr_at(s):
        if s < warmup:
            return tc["lr"] * s / max(1, warmup)
        p = (s - warmup) / max(1, total - warmup)
        return 0.1 * tc["lr"] + 0.5 * 0.9 * tc["lr"] * (1 + math.cos(math.pi * p))

    amp = dev == "cuda"
    model.train()
    t0 = time.time()
    step = start_step
    rng = random.Random(start_step * 7919 + 13)
    while step < total:
        if time.time() - t0 > time_budget:
            print(f"[TIME-GUARD] {time_budget}s tercapai pada step {step} — simpan & keluar.", flush=True)
            save_ckpt(out, model, opt, step, cfg.__dict__, codec, total, opt_kind)
            return
        step += 1
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        micro = 0.0
        for _ in range(grad_accum):
            idxs = [rng.randrange(len(train_data)) for _ in range(microbatch * 4)]   # oversample utk filter drop
            xb, yb = sft_batch(codec, USER, ASST, THINK, ENDTHINK, train_data, idxs, max_len, dev)
            if xb is None or xb.shape[0] < microbatch:
                continue
            xb, yb = xb[:microbatch], yb[:microbatch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = sft_loss(model, xb, yb) / grad_accum
            loss.backward()
            micro += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not (math.isfinite(micro) and torch.isfinite(gn)):
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        if step % 25 == 0:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
            print(f"step {step}/{total}  loss {micro:.3f}  gnorm {gn.item():.2f}  "
                  f"lr {lr_at(step):.2e}  {el/max(1,step-start_step):.1f}s/step  vram {mem:.1f}GB  "
                  f"elapsed {el/3600:.2f}h", flush=True)
        if step % tc["ckpt_every"] == 0:
            save_ckpt(out, model, opt, step, cfg.__dict__, codec, total, opt_kind)

    save_ckpt(out, model, opt, step, cfg.__dict__, codec, total, opt_kind)
    print(f"[DONE SFT] total_steps {total} -> {out}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
