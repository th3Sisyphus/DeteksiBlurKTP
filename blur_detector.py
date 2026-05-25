import cv2
import numpy as np

def hitung_skor_blur(img_gray: np.ndarray) -> dict:
    # Ambil tinggi dan lebar gambar
    h, w = img_gray.shape[:2]
    
    # Resize jika tinggi gambar kurang dari 80 piksel
    if h < 80:
        scale = 80 / h
        img_gray = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        h, w = img_gray.shape[:2]
    
    # Potong 15 piksel luar agar garis potong crop tidak ikut dihitung
    if h > 40 and w > 40:
        img_inner = img_gray[15:h-15, 15:w-15]
    else:
        img_inner = img_gray

    # Update ukuran setelah di-slice
    h_inner, w_inner = img_inner.shape[:2]

    # 2. PRE-PROCESSING: Bilateral Filter 
    # Meredam noise kompresi/bintik secara agresif tapi tetap mempertahankan ketajaman tepi huruf
    img_filtered = cv2.bilateralFilter(img_inner, 9, 75, 75)

    # 3. ADAPTIVE THRESHOLDING: Segmentasi teks lokal
    # # Adaptive/Dynamic Block Size
    dynamic_block_size = int(h_inner*0.08)
    
    if dynamic_block_size % 2 == 0:
        dynamic_block_size += 1
    
    # minimal 11 agar tidak terlalu hancur pada gambar resolusi sangat rendah
    dynamic_block_size = max(11, dynamic_block_size)
    
    # Dynamic Thresholding
    thresh = cv2.adaptiveThreshold(
        img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, dynamic_block_size, 2
    )
    
    # 4. MORPHOLOGICAL OPERATIONS: Menghapus Bintik Noise
    # Menggunakan kernel 3x3 agar noise bintik yang agak besar bisa terhapus
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 5. CONTOUR ANALYSIS: Menghitung objek karakter tulisan asli yang valid
    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # DYNAMIC SIZING: Hitung batas ukuran huruf berdasarkan tinggi potongan gambar
    # Huruf KTP biasanya memakan 2% hingga 25% dari tinggi area yang di-crop
    min_h = max(5, int(h_inner * 0.02))  
    max_h = max(10, int(h_inner * 0.25)) 
    
    valid_text_contours = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Filter Ukuran Dinamis
        if (min_h * 0.8) <= cw <= (max_h * 1.5) and min_h <= ch <= max_h:
            # Filter Aspek Rasio (Karakter huruf KTP umumnya proporsional tegak/persegi)
            aspect_ratio = float(cw) / ch
            if 0.2 <= aspect_ratio <= 1.2:
                # Filter Extent (Kepadatan Area) - Mencegah bintik noise lolos
                # Teks asli umumnya mengisi 20% - 80% dari kotak bounding box-nya
                extent = area / float(cw * ch) if (cw * ch) > 0 else 0
                if 0.2 <= extent <= 0.8:
                    valid_text_contours += 1

    # Normalisasi skor Kepadatan Teks (Text Density)
    score_text = min(100.0, (valid_text_contours / 80.0) * 100)
    score_text = 100 / (1 + np.exp(-0.12 * (score_text - 45)))

    # 6. FREQUENCY ANALYSIS (FFT): Analisis komponen frekuensi tinggi pendamping
    f_transform = np.fft.fft2(img_inner.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    rows, cols = img_inner.shape
    crow, ccol = rows // 2, cols // 2
    radius_low = min(rows, cols) // 5
    
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)
    mask_high = dist_from_center > radius_low
    
    total_energy = np.sum(magnitude_spectrum ** 2)
    high_freq_energy = np.sum((magnitude_spectrum * mask_high) ** 2)
    high_freq_ratio = (high_freq_energy / total_energy * 100) if total_energy > 0 else 0
    
    score_fft = min(100.0, (high_freq_ratio / 1.8) * 100)
    score_fft = 100 / (1 + np.exp(-0.15 * (score_fft - 45)))

    # 7. BOBOT AKHIR: 50% Text Density + 50% FFT Frekuensi
    final_score = (0.50 * score_text) + (0.50 * score_fft)
    final_score = max(0.0, min(100.0, final_score))

    # 8. KLASIFIKASI KUALITAS BERDASARKAN SKOR
    if final_score >= 70:
        kualitas, warna, rekomendasi = "SANGAT BAIK (TAJAM)", "#00aa4f", "Foto KTP sangat jelas, tulisan terbaca sempurna!"
    elif final_score >= 45:
        kualitas, warna, rekomendasi = "BAIK (CUKUP JELAS)", "#0066cc", "Foto cukup baik dan layak digunakan."
    elif final_score >= 25:
        kualitas, warna, rekomendasi = "CUKUP (AGAK BLUR)", "#ff9900", "KTP agak buram atau berisik. Disarankan foto ulang."
    else:
        kualitas, warna, rekomendasi = "KURANG (BLUR PARAH)", "#cc0000", "Foto tidak layak! Karakter tulisan rusak atau tidak terdeteksi."

    return {
        "score": round(final_score, 1),
        "kualitas": kualitas,
        "warna": warna,
        "rekomendasi": rekomendasi,
        "rincian": f"Clean_Text: {round(score_text,1)} | FFT_Domain: {round(score_fft,1)}",
        "thresh_cleaned": thresh_cleaned, 
        "img_inner": img_filtered,
    }

def validate_file_extension(filename: str) -> bool:
    """Memvalidasi ekstensi file gambar."""
    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    return filename.lower().endswith(VALID_EXTENSIONS)