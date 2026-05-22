import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

from blur_detector import (
    hitung_skor_blur,
    validate_file_extension
)

st.set_page_config(
    page_title="Deteksi Blur KTP — Kepadatan Teks",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
    text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.main-header h1 { color:#fff; font-size:2.2rem; font-weight:800; margin:0 0 .5rem; letter-spacing:-.5px; }
.main-header p { color:#a5b4fc; font-size:1rem; margin:0; }

.metric-card {
    background: linear-gradient(145deg, #1e1e2f, #2a2a40);
    border: 1px solid rgba(165,180,252,0.15); border-radius: 14px;
    padding: 1.5rem; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(99,102,241,0.15); }
.metric-label { color:#94a3b8; font-size:.8rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:.5rem; }
.metric-value { font-size:1.8rem; font-weight:800; margin:0; }
.metric-sub { color:#64748b; font-size:.75rem; margin-top:.3rem; }

.info-box {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border-left: 4px solid #6366f1; border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem; margin: 1rem 0;
    color: #cbd5e1; font-size: .9rem; line-height: 1.6;
}

[data-testid="stFileUploader"] { border:2px dashed rgba(99,102,241,0.3); border-radius:16px; padding:1rem; transition:border-color .3s; }
[data-testid="stFileUploader"]:hover { border-color:rgba(99,102,241,0.6); }

.fancy-divider { height:2px; background:linear-gradient(90deg,transparent,#6366f1,transparent); border:none; margin:2rem 0; border-radius:999px; }
</style>
""", unsafe_allow_html=True)


def render_metric(label, value, sub="", color="white"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <p class="metric-value" style="color: {color};">{value}</p>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>🪪 Deteksi Kualitas Blur KTP</h1>
    <p>Analisis Kepadatan Teks (Morfologi) & Frekuensi (FFT)</p>
</div>""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 📖 Cara Kerja")

    st.markdown("""
    <div class="info-box">
        <strong>Pilih Area Teks:</strong><br>
        Setelah mengunggah gambar, pilih (crop) area yang berisi teks KTP. Hindari jari, background luar, atau pas foto agar analisis lebih akurat.
    </div>""", unsafe_allow_html=True)

    steps = [
        ("1️⃣ Upload Foto", "Unggah foto KTP (JPG, PNG, WEBP, BMP)."),
        ("2️⃣ Crop Teks", "Seleksi kotak area teks."),
        ("3️⃣ Pre-Processing", "Bilateral Filter & Adaptive Thresholding."),
        ("4️⃣ Morfologi", "Operasi Morphological Opening untuk hapus noise."),
        ("5️⃣ Contour & FFT", "Hitung kepadatan teks dinamis dan frekuensi tinggi."),
        ("6️⃣ Klasifikasi", "Sangat Baik / Baik / Cukup / Kurang."),
    ]
    for title, desc in steps:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, #1e1e2f, #262640); border-radius: 14px; padding: 1.5rem; margin: .5rem 0; border: 1px solid rgba(99,102,241,0.1);">
            <div style="color:#a5b4fc; font-weight:700; font-size:.95rem; margin-bottom:.3rem;">{title}</div>
            <div style="color:#94a3b8; font-size:.85rem; line-height:1.5;">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ── Upload ──
st.markdown("### 📤 Upload Foto KTP")
uploaded_file = st.file_uploader(
    "Pilih file gambar KTP", type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Format: JPG, PNG, WEBP, BMP", label_visibility="collapsed",
)

if uploaded_file is not None:
    if not validate_file_extension(uploaded_file.name):
        st.error("❌ **Format file tidak valid!** Hanya JPG, PNG, WEBP, BMP.")
        st.stop()

    pil_image = Image.open(uploaded_file).convert("RGB")
    
    st.markdown("### ✂️ Seleksi Area Teks")
    st.info("💡 **Petunjuk:** Geser kotak di bawah untuk menyeleksi **area teks KTP**. Hindari jari tangan atau background di luar KTP untuk akurasi maksimal.")
    
    # st_cropper returns the cropped PIL Image
    cropped_img = st_cropper(pil_image, realtime_update=True, box_color='#00FF00', aspect_ratio=None)
    
    if st.button("🔍 Analisis Potongan KTP", type="primary"):
        with st.spinner("Menganalisis kepadatan teks dan frekuensi..."):
            # Konversi PIL ke OpenCV Grayscale
            img_cv = np.array(cropped_img)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            
            # Hitung skor blur menggunakan fungsi PCD yang baru
            hasil = hitung_skor_blur(img_gray)
            
            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📊 Hasil Analisis")

            col1, col2, col3 = st.columns(3)
            with col1:
                render_metric("Skor Fokus", f"{hasil['score']} / 100", sub="Total Skor PCD", color=hasil['warna'])
            with col2:
                render_metric("Kualitas", hasil['kualitas'], sub="Kategori", color=hasil['warna'])
            with col3:
                render_metric("Rekomendasi", "Lihat di bawah", sub=hasil['rincian'])

            st.markdown(f"""
            <div style="background-color: {hasil['warna']}20; border-left: 5px solid {hasil['warna']}; padding: 15px; border-radius: 5px; margin-top: 15px;">
                <h4 style="margin: 0; color: {hasil['warna']};">Rekomendasi Sistem:</h4>
                <p style="margin: 5px 0 0 0; color: #cbd5e1; font-size: 1.1rem;">{hasil['rekomendasi']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            
            # Visualisasi
            st.markdown("### 🖼️ Visualisasi Proses Morfologi")
            tab1, tab2, tab3 = st.tabs(["✂️ Crop Citra (Grayscale)", "🔘 Pre-processing (Bilateral Filter)", "🔠 Threshold & Cleaning (Kontur)"])
            
            with tab1:
                st.image(img_gray, caption="Potongan Citra Input", width=600, channels="GRAY")
            with tab2:
                st.image(hasil["img_inner"], caption="Border Slicing & Area Dalam", width=600, channels="GRAY")
            with tab3:
                st.image(hasil["thresh_cleaned"], caption="Hasil Segmentasi Teks (Bintik Dihapus)", width=600, channels="GRAY")

else:
    st.markdown("""
    <div style="background:linear-gradient(145deg,#1e1e2f,#262640);border:2px dashed rgba(99,102,241,0.3);
        border-radius:16px;padding:4rem 2rem;text-align:center;margin-top:1rem;">
        <p style="font-size:3rem;margin:0;">🪪</p>
        <p style="color:#a5b4fc;font-size:1.1rem;font-weight:600;margin:1rem 0 .5rem;">Belum ada foto yang diunggah</p>
        <p style="color:#64748b;font-size:.9rem;margin:0;">Unggah foto KTP untuk memulai analisis</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#475569;font-size:.75rem;padding:1rem 0;border-top:1px solid #1e293b;">
    Deteksi Blur KTP — Kepadatan Teks & Frekuensi &nbsp;•&nbsp; Pengolahan Citra Digital
</div>""", unsafe_allow_html=True)