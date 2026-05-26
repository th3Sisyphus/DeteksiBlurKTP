import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

from blur_detector import (
    hitung_skor_blur,
    validate_file_extension,
    auto_warp_ktp
)

st.set_page_config(
    page_title="Angin Tak Punya KTP Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

# ── Custom Cheerful & Bright Dashboard CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Outfit:wght@400;600;800&display=swap');

/* Force Light Mode Background for the whole app */
[data-testid="stAppViewContainer"], .stApp {
    background-color: #F4F7FE !important; /* Soft airy blue/grey */
}

html, body, [class*="css"] { 
    font-family: 'Nunito', sans-serif !important; 
    color: #334155; /* Dark slate for readable text */
}

h1, h2, h3, h4, h5, h6, .metric-value {
    font-family: 'Outfit', sans-serif !important;
}

/* Header Dashboard Ceria */
.db-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 2.5rem 2rem; 
    border-radius: 24px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(112, 144, 176, 0.12); /* Soft sunny shadow */
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    overflow: hidden;
}

/* Dekorasi Lingkaran Ceria di Background Header */
.db-header::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: linear-gradient(135deg, #FF9A9E 0%, #FECFEF 100%);
    border-radius: 50%;
    opacity: 0.4;
}

.db-header::after {
    content: '';
    position: absolute;
    bottom: -80px; right: 100px;
    width: 150px; height: 150px;
    background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
    border-radius: 50%;
    opacity: 0.3;
}

.db-title h1 {
    background: linear-gradient(90deg, #FF416C, #FF4B2B); /* Vibrant Coral to Orange */
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    position: relative;
    z-index: 2;
}

.db-title p {
    color: #64748B;
    margin: 0.3rem 0 0 0;
    font-size: 1.1rem;
    font-weight: 600;
    position: relative;
    z-index: 2;
}

/* Card Container Putih Bersih */
.db-card {
    background: #FFFFFF;
    border: 1px solid #F1F5F9;
    border-radius: 24px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 15px 35px rgba(112, 144, 176, 0.08); /* Bayangan lembut */
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.db-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(112, 144, 176, 0.15);
}

/* Metric Display Segar */
.metric-grid-card {
    background: #F8FAFC;
    border: 2px dashed #E2E8F0;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-grid-card:hover {
    background: #FFFFFF;
    border-color: #A5B4FC;
    box-shadow: 0 10px 25px rgba(99, 102, 241, 0.1);
}

.metric-grid-label {
    color: #94A3B8;
    font-size: 0.95rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-grid-value {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0.5rem 0;
}

/* Section Badges Imut (Pastel) */
.section-badge {
    background: #FFF0F0;
    color: #FF6B6B;
    padding: 0.5rem 1.2rem;
    border-radius: 100px;
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    display: inline-block;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 10px rgba(255, 107, 107, 0.2);
}

/* Custom File Uploader */
[data-testid="stFileUploader"] {
    background: #F8FAFC;
    border: 2px dashed #4FACFE; /* Garis putus-putus biru cerah */
    border-radius: 20px;
    padding: 2rem;
}
[data-testid="stFileUploader"]:hover {
    background: #F0F9FF;
    border-color: #00F2FE;
}

/* Tombol Biru Gradien Cerah */
.stButton>button {
    background: linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%);
    color: white;
    border-radius: 16px;
    padding: 0.8rem 2.5rem;
    font-weight: 800;
    font-size: 1.1rem;
    border: none;
    box-shadow: 0 8px 25px rgba(0, 242, 254, 0.4);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 12px 30px rgba(0, 242, 254, 0.6);
}
</style>
""", unsafe_allow_html=True)


# ── Dashboard Top Header ──
st.markdown("""
<div class="db-header">
    <div class="db-title">
        <h1>✨ Angin Tak Punya KTP Pro</h1>
        <p>Sistem Deteksi Kualitas Optik Dokumen KTP</p>
    </div>
    <div style="text-align: right; color: #4FACFE; font-size: 0.9rem; font-weight: 800; background: #E0F2FE; padding: 0.5rem 1rem; border-radius: 12px; z-index:2;">
        🚀 ENGINE: v2.0-HYBRID<br>STATUS: READY
    </div>
</div>""", unsafe_allow_html=True)


# ── Manajemen State Memori ──
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None
if 'warped_image_pil' not in st.session_state:
    st.session_state.warped_image_pil = None
if 'original_image_pil' not in st.session_state:
    st.session_state.original_image_pil = None
if 'is_warped_success' not in st.session_state:
    st.session_state.is_warped_success = False


# ── PANEL 1: AREA UPLOAD ──
st.markdown('<div class="db-card">', unsafe_allow_html=True)
st.markdown('<span class="section-badge">🌸 FASE 1</span><h3 style="color:#1E293B; margin-top:0;">📤 Unggah Dokumen KTP</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Pilih file gambar KTP", type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Format standar: JPG, PNG, WEBP, BMP", label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)


# Jalankan logika pemrosesan jika gambar tersedia
if uploaded_file is not None:
    if not validate_file_extension(uploaded_file.name):
        st.error("❌ Format file tidak didukung! Gunakan JPG, PNG, WEBP, atau BMP.")
        st.stop()

    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.show_results = False
        
        with st.spinner("🤖 Mendeteksi keajaiban... mencari sudut KTP!"):
            st.session_state.original_image_pil = Image.open(uploaded_file).convert("RGB")
            cv_image_bgr = cv2.cvtColor(np.array(st.session_state.original_image_pil), cv2.COLOR_RGB2BGR)
            
            warped_bgr, is_success = auto_warp_ktp(cv_image_bgr)
            
            warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
            st.session_state.warped_image_pil = Image.fromarray(warped_rgb)
            st.session_state.is_warped_success = is_success

    # ── PANEL 2: SHOWCASE AUTO-WARPING ──
    st.markdown('<div class="db-card">', unsafe_allow_html=True)
    st.markdown('<span class="section-badge" style="background:#FFF9C4; color:#F57F17;">☀️ FASE 2</span><h3 style="color:#1E293B; margin-top:0;">📐 Transformasi Perspektif Otomatis</h3>', unsafe_allow_html=True)
    
    if st.session_state.is_warped_success:
        st.success("🎉 Yeay! Algoritma berhasil menemukan 4 sudut KTP dan meratakannya secara ajaib!")
    else:
        st.info("💡 Hmm, batas luar KTP kurang jelas. Tidak apa-apa, kita gunakan gambar aslinya saja ya!")

    col_before, col_after = st.columns(2, gap="medium")
    with col_before:
        st.image(st.session_state.original_image_pil, caption="📸 Citra Asli (Input)", use_container_width=True)
    with col_after:
        st.image(st.session_state.warped_image_pil, caption="🖥️ Citra Hasil Diratakan (Auto-Warped)", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


    # ── PANEL 3: INTERACTIVE CANVAS WORKSPACE ──
    st.markdown('<div class="db-card">', unsafe_allow_html=True)
    st.markdown('<span class="section-badge" style="background:#E3F2FD; color:#1E88E5;">🌊 FASE 3</span><h3 style="color:#1E293B; margin-top:0;">✂️ Area Seleksi Interaktif</h3>', unsafe_allow_html=True)
    st.info("💡 **Tips:** Geser kotak hijau ceria di bawah ini untuk mengurung teks yang ingin dianalisis.")
    
    cropped_img = st_cropper(st.session_state.warped_image_pil, realtime_update=True, box_color='#00FF00', aspect_ratio=None)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if not st.session_state.show_results:
        if st.button("🚀 EKSTRAKSI SEKARANG", type="primary", use_container_width=True):
            st.session_state.show_results = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


    # ── PANEL 4: PERFORMANCE & ANALYTICS DASHBOARD ──
    if st.session_state.show_results:
        st.markdown('<div class="db-card" style="border-top: 6px solid #4FACFE;">', unsafe_allow_html=True)
        st.markdown('<span class="section-badge" style="background:#E8F5E9; color:#43A047;">🍀 FASE 4</span><h3 style="color:#1E293B; margin-top:0;">📊 Dashboard Kualitas Optik</h3>', unsafe_allow_html=True)
        
        img_cv = np.array(cropped_img)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        hasil = hitung_skor_blur(img_gray)
        
        col_m1, col_m2, col_m3 = st.columns(3, gap="large")
        with col_m1:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-grid-label">Skor Fokus Optik</div>
                <div class="metric-grid-value" style="color: {hasil['warna']}; text-shadow: 0 4px 15px {hasil['warna']}40;">{hasil['score']}</div>
                <div style="color: #64748b; font-size: 0.85rem; font-weight:600;">Poin Gabungan PCD</div>
            </div>""", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-grid-label">Kategori Kualitas</div>
                <div class="metric-grid-value" style="color: {hasil['warna']}; font-size: 1.8rem; padding: 0.6rem 0;">{hasil['kualitas']}</div>
                <div style="color: #64748b; font-size: 0.85rem; font-weight:600;">Status Kelayakan</div>
            </div>""", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-grid-label">Distribusi Fitur</div>
                <div style="color: #334155; font-size: 1.1rem; font-weight: 800; padding: 1.3rem 0;">{hasil['rincian']}</div>
                <div style="color: #64748b; font-size: 0.85rem; font-weight:600;">40% Teks | 40% FFT | 20% Laplacian</div>
            </div>""", unsafe_allow_html=True)

        # Kotak Rekomendasi 
        st.markdown(f"""
        <div style="background-color: {hasil['warna']}10; border: 2px dashed {hasil['warna']}40; padding: 1.5rem; border-radius: 16px; margin-top: 2rem; margin-bottom: 2.5rem; text-align: center;">
            <h5 style="margin: 0 0 0.5rem 0; color: {hasil['warna']}; font-weight: 800; font-size: 1.2rem;">💡 Kesimpulan & Saran:</h5>
            <p style="margin: 0; color: #334155; font-size: 1.1rem; font-weight: 600;">{hasil['rekomendasi']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h4 style='color:#1E293B;'>🖼️ Intip Proses di Balik Layar</h4>", unsafe_allow_html=True)
        tab_crop, tab_gray, tab_pcd, tab_biner = st.tabs([
            "🎯 Target Area (ROI)", 
            "🌓 Grayscale", 
            "🔮 Filter Anti-Pudar (CLAHE)", 
            "🔠 Bentuk Teks Akhir"
        ])
        
        with tab_crop:
            st.image(cropped_img, caption="Area yang Anda potong", use_container_width=True)
        with tab_gray:
            st.image(img_gray, caption="Gambar diubah menjadi hitam-putih murni", use_container_width=True)
        with tab_pcd:
            st.image(hasil["img_inner"], caption="Gambar dipertajam secara lokal untuk mengatasi teks pudar", use_container_width=True)
        with tab_biner:
            st.image(hasil["thresh_cleaned"], caption="Hasil binerisasi tempat komputer menghitung jumlah dan bentuk huruf", use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.session_state.show_results = False


# ── Footer ──
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color: #94A3B8; font-size: 0.9rem; font-weight: 700; padding: 2rem 0; letter-spacing: 1px;">
    ✨ DIBANGUN DENGAN PENUH SEMANGAT UNTUK TUGAS AKHIR PCD ✨<br>
    <span style="font-weight:500; font-size:0.8rem;">ANGIN TAK PUNYA KTP PRO</span>
</div>""", unsafe_allow_html=True)