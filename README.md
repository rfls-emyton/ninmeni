# NINMENI — Framework Model Bahasa Native Indonesia

**NINMENI** adalah kerangka riset dan rekayasa untuk membangun model bahasa yang
native terhadap bahasa Indonesia, dari lapisan paling dasar.

Hierarki: **framework NINMENI → paradigma NMU → model (contoh: Veyra)**.

- **NMU (ninmeni meaning unit)** — paradigma representasi: satu karakter = satu
  identitas tetap, di ruang karakter statis universal. Tanpa segmentasi sub-kata,
  tanpa UNK. Bahasa Indonesia hidup dari imbuhan; dengan membaca per karakter, akar
  kata selalu tampak utuh di setiap turunannya — model menemukan pola imbuhan sendiri.
- **Veyra** — model yang lahir dari kerangka ini. Generasi pertama: Veyra 75M.
  Generasi kedua: Veyra Nyra 92M, sedang dilatih dari nol sejak Agustus 2026.

> ### Repo ini adalah jalur GENERASI PERTAMA
>
> Kode dan ruang karakter di sini adalah jalur yang dipakai membangun **Veyra 75M**,
> dengan ruang karakter **±4.450** (`registry/nmu_v1.json`). Ia utuh dan berjalan
> untuk generasi itu.
>
> Generasi kedua memakai ruang karakter **10.240** dengan **format berkas registry
> yang berbeda**, sehingga codec di repo ini **tidak dapat memuatnya**. Perbedaannya
> bukan sekadar jumlah: 4.450 ID pertama tetap persis sama, tetapi berkas registry
> generasi kedua memakai skema `entries` — bukan `codepoints` seperti di sini.
>
> Perubahan inti lain pada generasi kedua: besaran yang menakar kerja tiap lapisan
> kini diturunkan sendiri di dalam paradigma NINMENI (**Ξ_EMYLTON**), menggantikan
> rumusan yang sebelumnya berasal dari luar NMU. Perubahan itu ada di arsitektur
> internal Veyra, yang memang tidak pernah dipublikasikan di repo ini (`model/`
> berisi decoder referensi polos — lihat bagian *Status & batas yang jujur*).
>
> Jalur generasi kedua akan diterbitkan setelah pelatihannya matang. Sampai saat itu,
> repo ini tidak diubah agar tetap dapat dipakai apa adanya.

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
membangun **Veyra 75M — generasi pertama** — bukan kode demonstrasi. Yang TIDAK ada
di repo ini: bobot model (untuk generasi mana pun), arsitektur internal Veyra
(model/ berisi referensi polos sebagai titik colok — lihat docs/antarmuka-model.md),
korpus pelatihan, dan catatan riset internal. Evaluasi menyeluruh menunggu
dokumentasi teknis rilis penuh.

Batas yang perlu dinyatakan terus terang: repo ini **tidak** mengikuti generasi kedua.
Ia tidak usang untuk apa yang dikerjakannya, tetapi ia juga bukan versi lama dari
jalur yang sedang berjalan sekarang — keduanya sudah bercabang. Kalau Anda memakai
repo ini, Anda memakai jalur generasi pertama, dan itu memang tetap sah.

## Kontribusi & lisensi
Kode di repo ini dilisensikan di bawah **Apache License 2.0** (lihat `LICENSE`).
Kontribusi dipersilakan melalui issue / pull request.

## Kontak
Emylton Leunufna — pencipta framework NINMENI & paradigma NMU — `emylleons8@gmail.com`
