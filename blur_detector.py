import cv2
import numpy as np

def order_points(pts):
    """Mengurutkan 4 titik koordinat: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def auto_warp_ktp(image_bgr: np.ndarray) -> tuple:
    """
    Mendeteksi KTP dan meluruskannya. Tahan terhadap background ramai
    dan mencegah warping pada kontur noise yang terlalu kecil.
    """
    orig = image_bgr.copy()
    
    # Hitung luas gambar asli untuk filter nanti
    h_orig, w_orig = orig.shape[:2]
    
    ratio = image_bgr.shape[0] / 500.0
    res = cv2.resize(image_bgr, (int(image_bgr.shape[1] / ratio), 500))
    h_res, w_res = res.shape[:2]
    res_area = h_res * w_res # Luas area gambar setelah di-resize

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur seringkali lebih stabil untuk tepi struktural makro dibanding Median Blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0) 
    edged = cv2.Canny(gray, 50, 150)
    
    # Morphological Closing untuk menyambung garis tepi yang terputus
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # Gunakan RETR_LIST agar kontur yang menyentuh ujung tepi frame tetap terbaca
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Syarat 1: Harus punya 4 sudut
        if len(approx) == 4:
            contour_area = cv2.contourArea(approx)
            # Syarat 2 (KUNCI PERBAIKAN): Luas kontur harus > 30% dari luas frame gambar
            if contour_area > (res_area * 0.30):
                screenCnt = approx
                break # Dapatkan kontur terbesar yang memenuhi syarat, lalu hentikan pencarian

    # Jika KTP yang valid ditemukan
    if screenCnt is not None:
        pts = screenCnt.reshape(4, 2) * ratio
        rect = order_points(pts)
        
        # Resolusi standar KTP
        width_ktp, height_ktp = 856, 539
        dst = np.array([
            [0, 0], [width_ktp - 1, 0],
            [width_ktp - 1, height_ktp - 1], [0, height_ktp - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(orig, M, (width_ktp, height_ktp))
        return warped, True
        
    # Jika gagal menemukan KTP (karena gambar KTP terpotong parah / nyaris full frame), 
    # kembalikan gambar asli secara aman
    return orig, False

def hitung_skor_blur(img_gray: np.ndarray) -> dict:
    h, w = img_gray.shape[:2]
    
    # 1. Scale Mitigation
    if h < 80:
        scale = 80 / h
        img_gray = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        h, w = img_gray.shape[:2]
    
    # 2. Border Slicing
    img_inner = img_gray[15:h-15, 15:w-15] if h > 40 and w > 40 else img_gray
    h_inner, w_inner = img_inner.shape[:2]

    # 3. CLAHE (Senjata rahasia untuk KTP pudar)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_inner)

    # 4. Bilateral Filter (Lebih halus karena sudah di-CLAHE)
    img_filtered = cv2.bilateralFilter(img_clahe, 5, 50, 50)

    # 5. Semi-Dynamic Adaptive Thresholding
    mean_val, std_dev = cv2.meanStdDev(img_filtered)
    dynamic_c = 8 if std_dev[0][0] > 55 else (6 if std_dev[0][0] > 35 else 4)
    
    dynamic_block_size = int(h_inner * 0.08)
    if dynamic_block_size % 2 == 0: dynamic_block_size += 1
    dynamic_block_size = max(11, dynamic_block_size)
    
    thresh = cv2.adaptiveThreshold(
        img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, dynamic_block_size, dynamic_c
    )
    
    # 6. Morphological Cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 7. FITUR 1: Spatial Contour & Solidity Analysis
    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_h = max(8, int(h_inner * 0.02))  
    max_h = max(10, int(h_inner * 0.25)) 
    
    valid_text_contours = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        if (min_h * 0.8) <= cw <= (max_h * 1.5) and min_h <= ch <= max_h:
            aspect_ratio = float(cw) / ch
            if 0.15 <= aspect_ratio <= 1.2:
                extent = area / float(cw * ch) if (cw * ch) > 0 else 0
                # Tambahan Filter Solidity (Luas vs Convex Hull)
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = area / float(hull_area) if hull_area > 0 else 0
                
                # Huruf tidak memblok padat (extent <= 0.7) dan memiliki lekukan (solidity <= 0.9)
                if 0.2 <= extent <= 0.7 and 0.3 <= solidity <= 0.9:
                    valid_text_contours += 1

    score_text = min(100.0, (valid_text_contours / 80.0) * 100)
    score_text = 100 / (1 + np.exp(-0.12 * (score_text - 55)))

    # 8. FITUR 2: Frequency Analysis (FFT)
    f_transform = np.fft.fft2(img_inner.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    rows, cols = img_inner.shape
    crow, ccol = rows // 2, cols // 2
    radius_low = min(rows, cols) // 5
    
    Y, X = np.ogrid[:rows, :cols]
    mask_high = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2) > radius_low
    
    total_energy = np.sum(magnitude_spectrum ** 2)
    high_freq_ratio = (np.sum((magnitude_spectrum * mask_high) ** 2) / total_energy * 100) if total_energy > 0 else 0
    score_fft = 100 / (1 + np.exp(-0.15 * (min(100.0, (high_freq_ratio / 1.8) * 100) - 55)))

    # 9. FITUR 3: Laplacian Variance (Fokus Edge)
    laplacian_var = cv2.Laplacian(img_inner, cv2.CV_64F).var()
    score_lap = min(100.0, laplacian_var / 10.0) # Normalisasi kasar
    score_lap = 100 / (1 + np.exp(-0.10 * (score_lap - 40)))

    # 10. BOBOT AKHIR (40% Teks, 40% FFT, 20% Laplacian)
    final_score = (0.40 * score_text) + (0.40 * score_fft) + (0.20 * score_lap)
    final_score = max(0.0, min(100.0, final_score))

    # Klasifikasi
    if final_score >= 70:
        kualitas, warna, rekomendasi = "SANGAT BAIK (TAJAM)", "#00aa4f", "Foto KTP sangat jelas, tulisan terbaca sempurna!"
    elif final_score >= 45:
        kualitas, warna, rekomendasi = "BAIK (CUKUP JELAS)", "#0066cc", "Foto cukup baik dan layak digunakan."
    elif final_score >= 25:
        kualitas, warna, rekomendasi = "CUKUP (AGAK BLUR)", "#ff9900", "KTP agak buram. Pastikan pencahayaan cukup."
    else:
        kualitas, warna, rekomendasi = "KURANG (BLUR PARAH)", "#cc0000", "Foto tidak layak! Silakan foto ulang dengan fokus yang benar."

    return {
        "score": round(final_score, 1),
        "kualitas": kualitas,
        "warna": warna,
        "rekomendasi": rekomendasi,
        "rincian": f"TXT: {round(score_text,1)} | FFT: {round(score_fft,1)} | LAP: {round(score_lap,1)}",
        "thresh_cleaned": thresh_cleaned, 
        "img_inner": img_clahe, 
    }

def validate_file_extension(filename: str) -> bool:
    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    return filename.lower().endswith(VALID_EXTENSIONS)