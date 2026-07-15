# Antarmuka Model — Kontrak yang Dituntut Pipeline

Repo ini menyertakan `model/reference_model.py` — decoder karakter-level yang
sengaja polos. Ia BUKAN arsitektur Veyra; ia titik colok. Arsitektur Anda sendiri
bisa langsung dipakai trainer selama memenuhi kontrak berikut.

## 1. Config (dataclass)

```python
@dataclass
class MyConfig:
    ukuran_ruang: int # ukuran ruang karakter — diisi trainer dari registry (codec.ukuran_ruang)
    pad_id: int       # diisi trainer dari registry (codec.pad_id)
    # ... field lain = kunci di blok `model:` pada configs/*.yaml
    grad_checkpoint: bool = False   # dibaca trainer untuk log
```

Trainer memvalidasi kunci yaml terhadap `fields(Config)` — kunci yang tidak
dikenal dibuang dengan peringatan. Jadi: field config Anda = kunci yaml Anda.

## 2. Model

```python
class MyModel(nn.Module):
    def __init__(self, cfg: MyConfig): ...
    # WAJIB: simpan config di atribut ini (dipakai sft_loss & checkpointing)
    self.cfg = cfg

    def forward(self, x):        # x: LongTensor [B, T] berisi ID karakter
        return logits            # FloatTensor [B, T, ukuran_ruang]

    def loss(self, x):           # objective pretraining
        # WAJIB SHIFTED: logits[:, :-1] vs x[:, 1:]
        # Loss tanpa shift = model belajar MENYALIN, bukan memprediksi —
        # ini kelas bug yang diam dan mahal. Uji: loss awal harus ~ln(ukuran ruang).
        ...
```

## 3. Aturan native yang tidak boleh dilanggar

- **Input = deretan ID karakter mentah** dari codec (1 karakter = 1 ID).
  Jangan menambahkan segmentasi/penggabungan apa pun sebelum pemetaan ID→vektor
  di dalam model — kalau Anda ingin kompresi, letakkan DI DALAM model sebagai
  bagian arsitektur.
- **Tanpa UNK**: ukuran_ruang datang dari registry; semua ID valid.
- **PAD**: tidak dilatih (ignore_index) dan tidak dilihat attention.
- **Checkpoint**: trainer menyimpan `model.state_dict()` + config dict —
  pastikan arsitektur Anda dapat direkonstruksi dari config-nya saja.

## 4. Menghubungkan ke trainer

Satu baris di `training/train_unified_native.py`:

```python
from model.reference_model import VeyraNativeModel, VeyraNativeConfig, count_params
# ganti menjadi:
from model.arsitektur_anda import VeyraNativeModel, VeyraNativeConfig, count_params
```

(Ekspor alias dengan nama yang sama, atau ubah ketiga nama di trainer.)

## 5. Cara menilai model native (kacamata yang benar)

Model karakter-level punya ritme belajarnya sendiri. Metrik yang wajar dipantau:
bits/char (BPC) pada validasi — bukan membandingkan kurva Anda dengan kurva model
ber-segmentasi sub-kata; keduanya mengukur unit yang berbeda. Perilaku awal seperti
pengulangan pada teks yang digenerasi adalah fase belajar substrat, bukan bug decoding.
