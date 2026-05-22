# 🪪 Deteksi Kualitas Blur KTP Indonesia

Aplikasi web untuk mendeteksi kualitas ketajaman foto KTP Indonesia menggunakan analisis **Kepadatan Teks (Morfologi)** dan **Frekuensi Tinggi (FFT)**.

## 📋 Alur Kerja

```
Upload Foto KTP → Crop Area Teks → Bilateral Filter → Adaptive Thresholding → Morphological Opening → Contour & FFT Analysis → Klasifikasi
```

1. **Input Foto KTP** — Pengguna mengunggah foto KTP (format: JPG, PNG, WEBP, BMP).
2. **Seleksi Area Teks (Crop)** — Pengguna secara manual memilih area teks pada KTP (menghindari pas foto dan latar belakang) untuk analisis yang presisi.
3. **Bilateral Filter** — Meredam noise kompresi secara agresif sembari mempertahankan ketajaman tepi huruf.
4. **Adaptive Thresholding** — Melakukan binarisasi untuk memisahkan teks dari latar belakang KTP secara lokal.
5. **Morphological Opening** — Menghapus bintik-bintik noise (kotoran) menggunakan kernel 3x3 agar hanya menyisakan karakter teks yang solid.
6. **Contour Analysis (Text Density)** — Menghitung jumlah kontur karakter yang valid berdasarkan aturan proporsi dan ukuran huruf KTP.
7. **Frequency Analysis (FFT)** — Menggunakan Transformasi Fourier Cepat (Fast Fourier Transform) untuk mengukur rasio energi frekuensi tinggi yang menandakan ketajaman citra.
8. **Klasifikasi** — Skor diakumulasi (50% Morfologi + 50% FFT) untuk menentukan kualitas: Sangat Baik, Baik, Cukup, atau Kurang.

## 🧮 Sistem Penilaian

Skor akhir dihitung dalam rentang **0 hingga 100**, yang dibagi menjadi 4 kategori:

| Skor Akhir | Status Kualitas | Rekomendasi |
|------------|-----------------|-------------|
| **≥ 70**   | ✅ Sangat Baik (Tajam) | Foto KTP sangat jelas, tulisan terbaca sempurna! |
| **45 - 69**| 🟢 Baik (Cukup Jelas) | Foto cukup baik dan layak digunakan. |
| **25 - 44**| ⚠️ Cukup (Agak Blur) | KTP agak buram atau berisik. Disarankan foto ulang. |
| **< 25**   | ❌ Kurang (Blur Parah) | Foto tidak layak! Karakter tulisan rusak atau tidak terdeteksi. |

## 🚀 Instalasi & Menjalankan

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501` (atau port Streamlit yang tersedia).

## 📁 Struktur Proyek

```
DeteksiBlurKTP_PCD/
├── app.py              # Antarmuka web Streamlit (UI + Integrasi Cropper)
├── blur_detector.py    # Logika inti Pemrosesan Citra Digital (Morfologi & FFT)
├── requirements.txt    # Daftar dependensi Python
└── README.md           # Dokumentasi proyek
```

## 🛠️ Teknologi & Pustaka Utama

- **Python 3.10+**
- **OpenCV (`cv2`)** — Filter bilateral, thresholding, operasi morfologi, dan pencarian kontur.
- **NumPy** — Komputasi matriks dan analisis domain frekuensi (FFT).
- **Streamlit** — Framework pembuatan antarmuka web interaktif.
- **streamlit-cropper** — Komponen antarmuka untuk melakukan *cropping* gambar secara langsung di peramban.
- **Pillow** — Manipulasi format citra.
