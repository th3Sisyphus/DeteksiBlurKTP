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
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

# ── Custom Futuristic Dark Dashboard CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;500;600;700&display=swap');

/* Force Dark Mode Background for the whole app */
[data-testid="stAppViewContainer"], .stApp {
    background-color: #080c14 !important; /* Deep dark space color */
}

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif !important; 
    color: #e2e8f0; /* Light slate for readable text */
}

h1, h2, h3, h4, h5, h6, .metric-value {
    font-family: 'Outfit', sans-serif !important;
}

/* Header Dashboard Gelap & Elegan */
.db-header {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.5) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 2.5rem 2rem; 
    border-radius: 24px;
    margin-bottom: 2rem;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    overflow: hidden;
}

/* Efek Cahaya Halus di Background Header */
.db-header::before {
    content: '';
    position: absolute;
    top: -50px; left: -50px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

.db-title h1 {
    background: linear-gradient(90deg, #818cf8, #c4b5fd); /* Indigo to Purple */
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    position: relative;
    z-index: 2;
    letter-spacing: -0.5px;
}

.db-title p {
    color: #94a3b8;
    margin: 0.3rem 0 0 0;
    font-size: 1.1rem;
    font-weight: 400;
    position: relative;
    z-index: 2;
}

/* Card Container Glassmorphism */
.db-card {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.db-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.3);
}

/* Metric Display Futuristik */
.metric-grid-card {
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    transition: all 0.3s ease;
}
.metric-grid-card:hover {
    background: rgba(30, 41, 59, 0.6);
    border-color: rgba(129, 140, 248, 0.3);
    box-shadow: 0 10px 25px rgba(99, 102, 241, 0.1);
}

.metric-grid-label {
    color: #64748b;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.metric-grid-value {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0.5rem 0;
}

/* Section Badges Cyberpunk-ish */
.section-badge {
    background: rgba(99, 102, 241, 0.1);
    color: #818cf8;
    padding: 0.4rem 1.2rem;
    border-radius: 100px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    display: inline-block;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(99, 102, 241, 0.2);
}

/* Custom File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.2);
    border: 2px dashed rgba(99, 102, 241, 0.4);
    border-radius: 20px;
    padding: 2rem;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(30, 41, 59, 0.4);
    border-color: #818cf8;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
}

/* Tombol Biru Indigo */
.stButton>button {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
    color: white;
    border-radius: 14px;
    padding: 0.8rem 2.5rem;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
    border: none;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(79, 70, 229, 0.5);
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
}

/* Customizing Streamlit Tabs for Dark Mode */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# ── Dashboard Top Header ──
st.markdown("""
<div class="db-header">
    <div class="db-title">
        <h1>🪪 Angin Tak Punya KTP Pro</h1>
        <p>Sistem Pemrosesan Citra Digital Penilai Kualitas Optik Dokumen</p>
    </div>
    <div style="text-align: right; color: #818cf8; font-size: 0.85rem; font-weight: 700; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.2); padding: 0.5rem 1rem; border-radius: 12px; z-index:2; letter-spacing: 0.5px;">
        ENGINE: v2.0-HYBRID<br>STATUS: READY
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
st.markdown('<span class="section-badge">FASE 1</span><h3 style="color:#e2e8f0; margin-top:0;">📤 Unggah Dokumen KTP</h3>', unsafe_allow_html=True)

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
        
        with st.spinner("🤖 Mengaktifkan Komputasi Spasial: Mencari koordinat KTP..."):
            st.session_state.original_image_pil = Image.open(uploaded_file).convert("RGB")
            cv_image_bgr = cv2.cvtColor(np.array(st.session_state.original_image_pil), cv2.COLOR_RGB2BGR)
            
            warped_bgr, is_success = auto_warp_ktp(cv_image_bgr)
            
            warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
            st.session_state.warped_image_pil = Image.fromarray(warped_rgb)
            st.session_state.is_warped_success = is_success

    # ── PANEL 2: SHOWCASE AUTO-WARPING ──
    st.markdown('<div class="db-card">', unsafe_allow_html=True)
    st.markdown('<span class="section-badge" style="background:rgba(245, 158, 11, 0.1); color:#fbbf24; border-color:rgba(245, 158, 11, 0.2);">FASE 2</span><h3 style="color:#e2e8f0; margin-top:0;">📐 Transformasi Perspektif Otomatis</h3>', unsafe_allow_html=True)
    
    if st.session_state.is_warped_success:
        st.success("✨ Algoritma Berhasil! 4 Sudut eksternal terdeteksi. Perspektif citra telah diratakan secara geometri.")
    else:
        st.info("⚠️ Batas luar KTP tidak terdeteksi secara utuh. Sistem beralih menggunakan citra asli.")

    col_before, col_after = st.columns(2, gap="medium")
    with col_before:
        st.image(st.session_state.original_image_pil, caption="📸 Citra Asli (Input)", use_container_width=True)
    with col_after:
        st.image(st.session_state.warped_image_pil, caption="🖥️ Citra Hasil Perspektif (Auto-Warped)", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


    # ── PANEL 3: INTERACTIVE CANVAS WORKSPACE ──
    st.markdown('<div class="db-card">', unsafe_allow_html=True)
    st.markdown('<span class="section-badge" style="background:rgba(14, 165, 233, 0.1); color:#38bdf8; border-color:rgba(14, 165, 233, 0.2);">FASE 3</span><h3 style="color:#e2e8f0; margin-top:0;">✂️ Area Seleksi Teks Interaktif</h3>', unsafe_allow_html=True)
    st.info("💡 **Perintah Operasional:** Geser kotak *cropper* hijau di bawah ini khusus untuk menyeleksi blok teks karakter.")
    
    cropped_img = st_cropper(st.session_state.warped_image_pil, realtime_update=True, box_color='#00FF00', aspect_ratio=None)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if not st.session_state.show_results:
        if st.button("🚀 EKSEKUSI ANALISIS", type="primary", use_container_width=True):
            st.session_state.show_results = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


    # ── PANEL 4: PERFORMANCE & ANALYTICS DASHBOARD ──
    if st.session_state.show_results:
        st.markdown('<div class="db-card" style="border-top: 4px solid #6366f1;">', unsafe_allow_html=True)
        st.markdown('<span class="section-badge" style="background:rgba(16, 185, 129, 0.1); color:#34d399; border-color:rgba(16, 185, 129, 0.2);">FASE 4</span><h3 style="color:#e2e8f0; margin-top:0;">📊 Multi-Domain Quality Assessment Dashboard</h3>', unsafe_allow_html=True)
        
        img_cv = np.array(cropped_img)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        hasil = hitung_skor_blur(img_gray)
        
        col_m1, col_m2, col_m3 = st.columns(3, gap="large")
        with col_m1:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-grid-label">Skor Fokus Optik</div>
                <div class="metric-grid-value" style="color: {hasil['warna']}; text-shadow: 0 0 25px {hasil['warna']}50;">{hasil['score']}</div>
                <div style="color: #64748b; font-size: 0.85rem; font-weight:500;">Poin Gabungan Konvolusi & Spektrum</div>
            </div>""", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-grid-label">Kategori Kualitas</div>
                <div class="metric-grid-value" style="color: {hasil['warna']}; font-size: 1.8rem; padding: 0.6rem 0;">{hasil['kualitas']}</div>
                <div style="color: #64748b; font-size: 0.85rem; font-weight:500;">Klasifikasi Ambang Batas Citra</div>
            </div>""", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-grid-label">Distribusi Vektor Fitur</div>
                <div style="color: #e2e8f0; font-size: 1.1rem; font-weight: 700; padding: 1.3rem 0; letter-spacing: 0.5px;">{hasil['rincian']}</div>
                <div style="color: #64748b; font-size: 0.85rem; font-weight:500;">40% Teks | 40% FFT | 20% Laplacian</div>
            </div>""", unsafe_allow_html=True)

        # Kotak Rekomendasi 
        st.markdown(f"""
        <div style="background-color: {hasil['warna']}10; border: 1px solid {hasil['warna']}30; border-left: 5px solid {hasil['warna']}; padding: 1.5rem; border-radius: 12px; margin-top: 2rem; margin-bottom: 2.5rem;">
            <h5 style="margin: 0 0 0.4rem 0; color: {hasil['warna']}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; font-size: 1rem;">Rekomendasi Integritas Sistem:</h5>
            <p style="margin: 0; color: #cbd5e1; font-size: 1.05rem; font-weight: 400;">{hasil['rekomendasi']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h4 style='color:#e2e8f0; font-weight: 600;'>🖼️ Matriks Inspeksi Sinyal Citra</h4>", unsafe_allow_html=True)
        tab_crop, tab_gray, tab_pcd, tab_biner = st.tabs([
            "🎯 ROI Crop Target", 
            "🌓 Grayscale Space", 
            "🔮 CLAHE Enhancement & Filter", 
            "🔠 Segmentasi Geometri"
        ])
        
        with tab_crop:
            st.image(cropped_img, caption="Region of Interest (ROI) Terpilih", use_container_width=True)
        with tab_gray:
            st.image(img_gray, caption="Peta Distribusi Kecerahan Piksel Tunggal", use_container_width=True)
        with tab_pcd:
            st.image(hasil["img_inner"], caption="Hasil Ekualisasi Histogram Adaptif Lokal (CLAHE) + Reduksi Noise", use_container_width=True)
        with tab_biner:
            st.image(hasil["thresh_cleaned"], caption="Hasil Ekstraksi Bentuk Karakter Siap Hitung", use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.session_state.show_results = False


# ── Footer / About Creator (Dark Mode) ──
st.markdown("<br><br><hr style='border: none; border-top: 1px solid rgba(255,255,255,0.05); margin-bottom: 2rem;'>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: #e2e8f0; margin-bottom: 1.5rem; font-weight: 600;'>👨‍💻 About Creator</h3>", unsafe_allow_html=True)

# Grid 3 Kolom untuk Profil Kreator
col_c1, col_c2, col_c3 = st.columns(3)

creators = [
    ("Nicolas Krisna P.", "71231019"),
    ("Yehezkiel Darren P. W.", "71231023"),
    ("Hansel Ivano S.", "71231039")
]

for col, (name, nim) in zip([col_c1, col_c2, col_c3], creators):
    with col:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 16px; border-top: 4px solid #6366f1; border-bottom: 1px solid rgba(255,255,255,0.05); border-left: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05); text-align: center; transition: transform 0.2s ease;">
            <p style="margin: 0; font-weight: 700; color: #f8fafc; font-size: 1.05rem;">{name}</p>
            <p style="margin: 5px 0 0 0; color: #818cf8; font-size: 0.9rem; font-weight: 600; letter-spacing: 1px;">NIM: {nim}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color: #475569; font-size: 0.8rem; font-weight: 500; padding: 2.5rem 0 1rem 0; letter-spacing: 1px;">
    SYSTEM DEVELOPED FOR DIGITAL IMAGE PROCESSING PROJECT<br>
    <span style="color:#64748b;">© 2024 FTI UKDW • INFORMATIKA</span>
</div>""", unsafe_allow_html=True)