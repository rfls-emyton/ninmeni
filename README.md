# NINMENI — Framework Model Bahasa Native Indonesia

**NINMENI** adalah kerangka riset dan rekayasa untuk membangun model bahasa yang
native terhadap bahasa Indonesia, dari lapisan paling dasar.

Hierarki: **framework NINMENI → paradigma NMU → model (contoh: Veyra)**.

- **NMU (Native Meaning Unit)** — paradigma representasi: satu karakter = satu
  identitas tetap, di ruang karakter statis universal. Tanpa segmentasi sub-kata,
  tanpa UNK. Bahasa Indonesia hidup dari imbuhan; dengan membaca per karakter, akar
  kata selalu tampak utuh di setiap turunannya — model menemukan pola imbuhan sendiri.
- **Veyra** — model pertama yang lahir dari kerangka ini (beta).

## Isi repo
```
nmu/         codec NMU (paket paradigma) (encode/decode 1 karakter = 1 ID, registry-driven)
registry/    ruang karakter statis (nmu_v1.json)
model/       model REFERENSI (decoder polos) — titik colok arsitektur Anda; BUKAN arsitektur Veyra
training/    loop pelatihan unified (pretraining + SFT satu loop) + sampler shard
pipeline/    build_shards.py — teks JSONL -> shard ids (mmap-able)
configs/     contoh konfigurasi
docs/        panduan kurikulum data
examples/    contoh format kurikulum (pretraining & SFT)
```

## Mulai cepat
1. Siapkan korpus JSONL `{"teks": "..."}` (lihat `examples/curriculum/`).
2. Shardize: `python pipeline/build_shards.py --inputs korpus.jsonl --out data/shards`
3. Latih: `python -m training.train_unified_native --config configs/contoh_75m.yaml \
   --shards data/shards/train --sft data/sft_train.jsonl --registry registry/nmu_v1.json`

## Status & batas yang jujur
Proyek ini dibuka bertahap. Kode di sini adalah pipeline yang benar-benar dipakai
membangun Veyra 75M (beta) — bukan kode demonstrasi. Yang TIDAK ada di repo ini:
bobot model, arsitektur internal Veyra (model/ berisi referensi polos sebagai titik
colok — lihat docs/antarmuka-model.md), korpus pelatihan, dan catatan riset internal. Evaluasi menyeluruh
menunggu dokumentasi teknis rilis penuh.

## Kontribusi & lisensi
Tata kelola kontribusi dan lisensi final sedang disiapkan (lihat `LICENSE.PLACEHOLDER.md`).
Sebelum itu ditetapkan, repo ini berstatus pratinjau sumber.

## Kontak
Emylton Leunufna — pencipta framework NINMENI & paradigma NMU — `emylleons8@gmail.com`
