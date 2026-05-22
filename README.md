# 🪪 Deteksi Kualitas Blur KTP Indonesia

Aplikasi web untuk mendeteksi kualitas blur pada foto KTP Indonesia menggunakan **metode Laplacian Variance** dengan deteksi ROI (Region of Interest) otomatis.

## 📋 Alur Kerja

```
Upload Foto KTP → Validasi Format → Deteksi ROI → Pra-pemrosesan → Filter Laplacian → Hitung Variance → Normalisasi → Klasifikasi
```

1. **Input foto KTP** — Pengguna mengunggah foto KTP (format: JPG, PNG, WEBP)
2. **Validasi format** — Sistem memeriksa ekstensi file
3. **Deteksi ROI** — Sistem mendeteksi area KTP pada gambar (3 metode)
4. **Pra-pemrosesan** — ROI di-resize + konversi ke grayscale
5. **Filter Laplacian** — Kernel 3×3 diterapkan pada ROI untuk mendeteksi tepi
6. **Hitung Variance (σ²)** — Variance dari respons Laplacian dihitung
7. **Normalisasi** — Konversi ke skala 0–100%
8. **Klasifikasi** — Tajam, Sedikit Blur, atau Blur

## 🎯 Metode Deteksi ROI

Aplikasi mendukung 3 metode untuk mendeteksi area KTP sebelum analisis blur:

### 1. Deteksi Dokumen (Kontur)
- Menggunakan **Canny edge detection** + **contour detection** dari OpenCV
- Mencari kontur terbesar berbentuk persegi panjang (4 sisi)
- Otomatis crop pada area kontur KTP yang terdeteksi
- Jika gagal, fallback ke Center Crop

### 2. Deteksi Wajah (Haar Cascade)
- Menggunakan **Haar Cascade Classifier** (`haarcascade_frontalface_default.xml`)
- KTP selalu memiliki pas foto — wajah yang blur = KTP blur
- Mengambil bounding box wajah + padding 30% untuk konteks
- Jika wajah tidak terdeteksi, fallback ke Center Crop

### 3. Center Crop (Cara Instan)
- Memotong **40% area tengah** gambar secara statis
- Asumsi: pengguna menempatkan KTP di tengah frame
- Tidak memerlukan machine learning

## 🧮 Formula

```
blur% = 100 − min(σ² / threshold × 100, 100)
```

| Kondisi | Status |
|---------|--------|
| σ² ≥ threshold | ✅ Tajam |
| threshold/2 ≤ σ² < threshold | ⚠️ Sedikit Blur |
| σ² < threshold/2 | ❌ Blur |

## 🚀 Instalasi & Menjalankan

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 Struktur Proyek

```
DeteksiBlurKTP_PCD/
├── app.py              # Aplikasi web Streamlit (UI + visualisasi)
├── blur_detector.py    # Modul deteksi blur + ROI (Laplacian)
├── requirements.txt    # Dependencies Python
└── README.md           # Dokumentasi
```

## 🛠️ Teknologi

- **Python 3.10+**
- **OpenCV** — Pemrosesan citra, filter Laplacian, Haar Cascade, contour detection
- **NumPy** — Komputasi numerik
- **Pillow** — Manipulasi gambar
- **Streamlit** — Antarmuka web

## 📖 Metode Laplacian

Filter Laplacian menghitung turunan kedua dari intensitas piksel. Kernel 3×3 yang digunakan:

```
[ 0,  1,  0]
[ 1, -4,  1]
[ 0,  1,  0]
```

- **Gambar tajam** → banyak perubahan intensitas → variance tinggi
- **Gambar blur** → transisi halus → variance rendah
