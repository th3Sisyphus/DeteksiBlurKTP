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
    page_title="Angin Tak Punya KTP",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif !important; 
    color: #e2e8f0;
    background-color: #0b0f19;
}

h1, h2, h3, h4, h5, h6, .metric-value {
    font-family: 'Outfit', sans-serif !important;
}

.main-header {
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 3rem 2rem; 
    border-radius: 24px;
    margin-bottom: 2.5rem;
    text-align: center; 
    box-shadow: 0 20px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 60%);
    animation: drift 15s linear infinite;
    pointer-events: none;
}

@keyframes drift {
    0% { transform: translate(0, 0) rotate(0deg); }
    100% { transform: translate(10%, 10%) rotate(360deg); }
}

.main-header h1 { 
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem; 
    font-weight: 800; 
    margin: 0 0 0.8rem; 
    letter-spacing: -1px;
}
.main-header p { 
    color: #94a3b8; 
    font-size: 1.15rem; 
    font-weight: 300;
    margin: 0; 
}

.metric-card {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1.8rem; 
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
.metric-card:hover { 
    transform: translateY(-8px) scale(1.02); 
    background: rgba(30, 41, 59, 0.6);
    box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
    border-color: rgba(129, 140, 248, 0.3);
}
.metric-label { 
    color: #64748b; 
    font-size: 0.85rem; 
    font-weight: 600; 
    text-transform: uppercase; 
    letter-spacing: 1.5px; 
    margin-bottom: 0.8rem; 
}
.metric-value { 
    font-size: 2.2rem; 
    font-weight: 800; 
    margin: 0.5rem 0; 
    text-shadow: 0 0 20px currentColor;
}
.metric-sub { 
    color: #94a3b8; 
    font-size: 0.8rem; 
    font-weight: 400;
}

.info-box {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border-left: 4px solid #6366f1;
    border-radius: 0 16px 16px 0;
    padding: 1.5rem; 
    margin: 1.5rem 0;
    color: #cbd5e1; 
    font-size: 0.95rem; 
    line-height: 1.7;
    box-shadow: 5px 5px 15px rgba(0,0,0,0.3);
}

[data-testid="stFileUploader"] { 
    border: 2px dashed rgba(99, 102, 241, 0.4); 
    background: rgba(30, 41, 59, 0.3);
    border-radius: 20px;
    padding: 2rem 1rem; 
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover { 
    border-color: #818cf8;
    background: rgba(30, 41, 59, 0.5);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
}

.fancy-divider { 
    height: 1px; 
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.5), transparent); 
    border: none; 
    margin: 3rem 0; 
}

.classic-step {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 1.2rem 1.5rem; 
    margin: 0.8rem 0; 
    border-left: 4px solid #818cf8;
    transition: transform 0.2s ease;
}
.classic-step:hover {
    transform: translateX(5px);
    background: rgba(30, 41, 59, 0.6);
}
.classic-step-title { 
    color: #e0e7ff; 
    font-weight: 600; 
    font-size: 1.05rem; 
    margin-bottom: 0.4rem; 
}
.classic-step-desc { 
    color: #94a3b8; 
    font-size: 0.9rem; 
    line-height: 1.5; 
}

.stButton>button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.6);
}
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
    <h1>Angin Tak Punya KTP</h1>
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
        ("2️⃣ Crop KTP", "Seleksi kotak area KTP."),
        ("3️⃣ Pre-Processing", "Bilateral Filter & Adaptive Thresholding."),
        ("4️⃣ Morfologi", "Operasi Morphological Opening untuk hapus noise."),
        ("5️⃣ Contour & FFT", "Hitung kepadatan teks dinamis dan frekuensi tinggi."),
        ("6️⃣ Klasifikasi", "Sangat Baik / Baik / Cukup / Kurang."),
    ]
    for title, desc in steps:
        st.markdown(f"""
        <div class="classic-step">
            <div class="classic-step-title">{title}</div>
            <div class="classic-step-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ── Manajemen State Layout Dinamis ──
# Menyimpan status apakah tombol analisis sudah ditekan atau belum
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# ── Penentuan Layout ──
# Jika show_results True -> Layar membelah jadi 2 kolom
# Jika show_results False -> Hanya 1 container (lebar penuh)
if st.session_state.show_results:
    col_left, col_right = st.columns([1, 1], gap="large")
else:
    col_left = st.container()
    col_right = None


# ── Render Kolom Kiri (Upload & Crop) ──
with col_left:
    st.markdown("### 📤 Upload Foto KTP")
    uploaded_file = st.file_uploader(
        "Pilih file gambar KTP", type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Format: JPG, PNG, WEBP, BMP", label_visibility="collapsed",
    )
    
    # Reset status layout jika foto dihapus/belum ada
    if uploaded_file is None:
        st.session_state.show_results = False
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); border: 2px dashed rgba(99, 102, 241, 0.4); padding: 3rem 1.5rem; text-align: center; margin-top: 1rem; border-radius: 20px; backdrop-filter: blur(12px);">
            <p style="font-size: 3rem; margin: 0; filter: drop-shadow(0 0 10px rgba(99,102,241,0.5));">🪪</p>
            <p style="color: #e0e7ff; font-size: 1.1rem; font-weight: 600; margin: 1rem 0 0.5rem; letter-spacing: 0.5px;">Belum ada foto yang diunggah</p>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">Silakan unggah foto di atas</p>
        </div>""", unsafe_allow_html=True)
    
    else:
        if not validate_file_extension(uploaded_file.name):
            st.error("❌ **Format file tidak valid!** Hanya JPG, PNG, WEBP, BMP.")
            st.stop()

        pil_image = Image.open(uploaded_file).convert("RGB")
        
        st.markdown("### ✂️ Seleksi Area KTP")
        st.info("💡 **Petunjuk:** Geser kotak di bawah untuk menyeleksi **area KTP**.")
        
        # St_cropper
        cropped_img = st_cropper(pil_image, realtime_update=True, box_color='#00FF00', aspect_ratio=None)
        
        # Tombol Analisis: Mengubah state layout dan memaksa aplikasi memuat ulang (rerun)
        if not st.session_state.show_results:
            if st.button("🔍 Analisis Potongan KTP", type="primary", use_container_width=True):
                st.session_state.show_results = True
                st.rerun()


# ── Render Kolom Kanan (Hasil Analisis & Visualisasi) ──
# Kolom kanan hanya dieksekusi jika layout sudah membelah dan col_right ada
if st.session_state.show_results and col_right is not None:
    with col_right:
        with st.spinner("Menganalisis kepadatan teks dan frekuensi..."):
            
            # Proses gambar (Sistem akan memproses real-time setiap kali kotak cropper di kolom kiri digeser)
            img_cv = np.array(cropped_img)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            
            # Hitung skor blur
            hasil = hitung_skor_blur(img_gray)
            
            st.markdown("### 📊 Hasil Analisis")

            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                render_metric("Skor Fokus", f"{hasil['score']} / 100", sub="Total Skor PCD", color=hasil['warna'])
            with col_metric2:
                render_metric("Kualitas", hasil['kualitas'], sub="Kategori", color=hasil['warna'])

            st.markdown(f"""
            <div style="background-color: {hasil['warna']}20; border-left: 5px solid {hasil['warna']}; padding: 15px; border-radius: 5px; margin-top: 15px; margin-bottom: 30px;">
                <h4 style="margin: 0; color: {hasil['warna']};">Rekomendasi Sistem:</h4>
                <p style="margin: 5px 0 0 0; color: #cbd5e1; font-size: 1.1rem;">{hasil['rekomendasi']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualisasi Tab
            st.markdown("### 🖼️ Visualisasi Proses Morfologi")
            tab1, tab2, tab3, tab4 = st.tabs(["✂️ Crop Citra", "Grayscale Mode", "🔘 Pre-processing", "🔠 Threshold Cleaning"])
            
            with tab1:
                st.image(cropped_img, caption="Crop Citra", use_container_width=True, channels="RGB")
            with tab2:
                st.image(img_gray, caption="Grayscale Mode", use_container_width=True, channels="GRAY")
            with tab3:
                st.image(hasil["img_inner"], caption="Pre-processing (Bilateral Filter)", use_container_width=True, channels="GRAY")
            with tab4:
                st.image(hasil["thresh_cleaned"], caption="Threshold Cleaning", use_container_width=True, channels="GRAY")


# ── Footer ──
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color: #475569; font-size: 0.8rem; font-weight: 500; padding: 1.5rem 0; border-top: 1px solid rgba(255,255,255,0.05); letter-spacing: 1px;">
    Angin Tak Punya KTP — Kepadatan Teks & Frekuensi &nbsp;•&nbsp; Pengolahan Citra Digital
</div>""", unsafe_allow_html=True)