# Kurikulum Data NINMENI — Panduan Ringkas

Dua jalur data, dua peran yang berbeda dan tidak boleh dicampur:

## 1. Pretraining — membentuk PEMAHAMAN
- Isi: teks aktual (prosa komunikasi nyata) berbahasa Indonesia — artikel, dokumentasi,
  narasi, kode dengan penjelasan. Format: JSONL `{"teks": "..."}` (lihat
  `examples/curriculum/pretraining_contoh.jsonl`).
- Prinsip: substrat belajar dari BAHASA HIDUP. Format meta-linguistik (tabel kamus,
  daftar lema, anotasi kelas kata) bukan corpus — kamus adalah alat kurasi, bukan data.
- Setiap sumber wajib berlisensi jelas dan tercatat (URL + lisensi + hash) di manifest
  Anda sendiri. Jangan gunakan data yang tidak boleh Anda gunakan.
- Konten pengetahuan dunia (fakta, statistik, budaya) masuk ke jalur INI — bukan ke SFT.

## 2. SFT — membentuk PROTOKOL DIALOG ("mulut")
- Isi: contoh percakapan/tugas dalam format turns (lihat `examples/curriculum/sft_contoh.jsonl`).
- Peran: mengajarkan BENTUK berinteraksi (format jawaban, protokol tool, gaya) — bukan
  menjejalkan fakta. Fakta yang hanya ada di SFT cenderung rapuh.
- Penalaran bertahap diajarkan sebagai LINTASAN: enumerasi langkah searah penulisan,
  jawaban = kelanjutan alami dari langkahnya (lihat contoh `reason`).

## Prinsip lintas-jalur
- 1 karakter = 1 ID: registry karakter statis (`registry/nmu_v1.json`) berlaku untuk
  SEMUA data. Teks yang memuat karakter di luar registry tidak dipaksa masuk — catat
  dan putuskan secara sadar (normalisasi terdokumentasi, atau buang utuh).
- Tanpa UNK: kegagalan encode = masalah validasi data, bukan sesuatu yang ditambal.
- Evaluasi model native dibaca dengan kacamata native: model karakter-level punya
  ritme belajar sendiri; jangan menilai dengan ekspektasi paradigma lain.
