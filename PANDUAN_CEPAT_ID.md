# Panduan Cepat Active Learning GrainSet-Rice

## Ringkasan

Kode menggunakan **ResNet50 existing dari timm dalam repository GrainSet**; tidak menggunakan BYOL. Dataset hanya **rice** dan memakai manifest asli `rice_train.txt`, `rice_val.txt`, `rice_test.txt`.

Jumlah training Rice = **24.767**, validation = **3.095**, test = **3.100**. Total AL = **25 titik evaluasi** mulai R0 (2% = 495) hingga R24 (50% = 12.384), setiap round menambahkan sekitar 495–496 gambar. Eksperimen menguji **Random, Least Confidence, Entropy, Margin, Hybrid Uncertainty + Diversity**. Tiap seed memakai labeled set awal identik untuk semua metode. Model dipelajari ulang dari inisialisasi bobot yang sama pada setiap round.

## Langkah 1: Instalasi

Siapkan PyTorch versi GPU sesuai CUDA/NVIDIA, lalu jalankan:

```bash
pip install -r requirements-al.txt
```

Pastikan gambar telah diekstrak ke folder seperti `/datasets/rice/` dan mempunyai struktur `train/` serta `test/` sesuai manifest. **ZIP unggahan asli hanya memuat source code dan manifest; tidak menyertakan gambar beras.**

## Langkah 2: Cek jadwal

```bash
python src/train_al.py --data-root /datasets/rice --dry-run
```

## Langkah 3: Coba 3 round terlebih dahulu

```bash
python src/train_al.py \
    --data-root /datasets/rice \
    --output runs/active_learning \
    --methods random entropy hybrid \
    --seeds 42 \
    --start-pct 2 --step-pct 2 --max-pct 50 \
    --max-rounds 3 \
    --epochs 50 --patience 10 \
    --backend timm --pretrained
```

Hanya menjalankan R0=2%, R1=4%, R2=6%, sambil mempertahankan jadwal penuh. Gunakan parameter yang sama tanpa `--max-rounds 3` untuk melanjutkan sampai 50%. **Jangan ubah hyperparameter di folder hasil yang sama**, karena akan ditolak untuk menjaga reprodusibilitas.

## Langkah 4: Jalankan semua eksperimen

```bash
DATA_ROOT=/datasets/rice bash scripts/run_al.sh
```

Ini menjalankan 5 metode × 25 round × 3 seed, ditambah 3 baseline full-supervised = **378 kali training**; cukup berat untuk satu GPU, jadi jalankan percobaan singkat lebih dahulu.

## Langkah 5: Grafik dan CSV

```bash
python src/plot_al.py --input runs/active_learning
```

Hasil tersedia di `runs/active_learning/`: `all_runs.csv`, `summary_mean_std.csv`, `aulc_by_seed.csv`, gambar `.png` dan `.pdf` dalam `figures/`, serta checkpoint, prediksi, dan query tiap round.

**Metrik:** Top-1, Top-2, Macro-F1, Balanced Accuracy, Recall kelas IM, MAPE kemurnian berbasis hitungan objek, dan MAE impuritas. MAPE dihitung dari kelompok gambar (*synthetic bags*), bukan secara langsung dari label klasifikasi. MAPE **bukan ukuran kemurnian massa atau kemurnian varietas**. Test digunakan hanya pada 50% label dan full-supervised 100%, sedangkan validation digunakan pada setiap round untuk kurva learning.

Baca `README_ACTIVE_LEARNING.md` untuk seluruh opsi, penjelasan metrik, kebijakan checkpoint, dan catatan metodologi.
