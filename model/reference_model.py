"""reference_model.py — MODEL REFERENSI untuk pipeline NINMENI.

PENTING: ini BUKAN arsitektur Veyra. Ini decoder-only transformer karakter-level
yang sengaja dibuat polos dan mudah dibaca, dengan satu tujuan: pipeline di repo
ini bisa dijalankan ujung-ke-ujung, dan Anda punya titik colok yang jelas untuk
arsitektur Anda sendiri.

Kontrak antarmuka yang dituntut trainer (lihat docs/antarmuka-model.md):
  - Config  : dataclass; field mencakup kunci `model:` di configs/*.yaml
              + ukuran_ruang + pad_id (diisi trainer dari registry).
  - Model   : __init__(cfg); menyimpan `self.cfg`;
              forward(x[B,T] long) -> logits [B, T, ukuran_ruang];
              loss(x[B,T]) -> scalar CE SHIFTED (predict karakter berikutnya).
  - count_params(model) -> int.

Prinsip native yang dijaga di sini:
  - Input = deretan ID karakter (1 karakter = 1 ID, registry-driven). Tidak ada
    segmentasi apa pun sebelum model.
  - Loss WAJIB shifted: logits[:, :-1] vs target x[:, 1:]. Loss tanpa shift =
    model belajar menyalin, bukan memprediksi.
  - PAD tidak dilatih dan tidak dilihat attention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ReferenceConfig:
    ukuran_ruang: int = 4450     # ukuran ruang karakter registry (jumlah ID)
    pad_id: int = 0
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4          # diterima demi kompatibilitas config; referensi memakai MHA penuh
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0
    grad_checkpoint: bool = False


def _rope(q, k, theta):
    # Penyandian posisi rotari (RoPE) sederhana (separuh-dimensi berpasangan).
    B, H, T, D = q.shape
    half = D // 2
    freqs = theta ** (-torch.arange(0, half, device=q.device, dtype=torch.float32) / half)
    pos = torch.arange(T, device=q.device, dtype=torch.float32)
    ang = torch.outer(pos, freqs)                      # [T, half]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]

    def rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rot(q), rot(k)


class _Block(nn.Module):
    def __init__(self, cfg: ReferenceConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        hidden = int(cfg.d_model * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model, bias=False),
        )

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        H = self.cfg.n_heads
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(B, T, H, C // H).transpose(1, 2)
        k = k.view(B, T, H, C // H).transpose(1, 2)
        v = v.view(B, T, H, C // H).transpose(1, 2)
        q, k = _rope(q, k, self.cfg.rope_theta)
        attn_mask = None
        if pad_mask is not None:
            # PAD tidak boleh dilihat sebagai key oleh posisi mana pun.
            attn_mask = pad_mask[:, None, None, :].expand(B, 1, T, T)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
        x = x + self.proj(out.transpose(1, 2).reshape(B, T, C))
        x = x + self.mlp(self.norm2(x))
        return x


class ReferenceNativeModel(nn.Module):
    def __init__(self, cfg: ReferenceConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.ukuran_ruang, cfg.d_model, padding_idx=cfg.pad_id)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.ukuran_ruang, bias=False)
        self.head.weight = self.emb.weight  # tied
        # Init kecil (std 0.02) — tanpa ini logits awal membesar dan CE awal
        # jauh di atas ln(ukuran ruang). Uji waras: loss step-0 harus ~ ln(ukuran_ruang).
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        pad_mask = None
        if (x == self.cfg.pad_id).any():
            pad_mask = x != self.cfg.pad_id  # True = boleh dilihat
        h = self.emb(x)
        for blk in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                h = checkpoint(blk, h, pad_mask, use_reentrant=False)
            else:
                h = blk(h, pad_mask)
        return self.head(self.norm_f(h))

    def loss(self, x):
        """CE SHIFTED — objective pretraining native (predict karakter berikutnya)."""
        logits = self.forward(x)
        return F.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.cfg.ukuran_ruang),
            x[:, 1:].reshape(-1),
            ignore_index=self.cfg.pad_id,
        )


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# Alias kompatibilitas — trainer mengimpor nama-nama ini. Saat Anda menulis
# arsitektur sendiri, ekspor nama yang sama dari modul Anda dan arahkan import
# di training/train_unified_native.py ke sana.
VeyraNativeConfig = ReferenceConfig
VeyraNativeModel = ReferenceNativeModel
