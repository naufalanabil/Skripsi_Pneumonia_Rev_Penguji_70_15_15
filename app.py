import os
import warnings
import time
import urllib.request
import gdown
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Custom Layer untuk mengabaikan parameter 'quantization_config' pada file h5 buatan Keras 3
class FixedDense(layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# ==============================================================================
# 2. KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="CDSS Pneumonia ResNet-50 - Naufal Ardra Anabil",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS (Tema Medis Modern & Akademis)
st.markdown("""
    <style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        text-align: center;
        margin-bottom: 5px;
        text-transform: uppercase;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        text-align: center;
        margin-bottom: 20px;
    }
    .identity-box {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        border-left: 6px solid #0284C7;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card h2 {
        margin: 0;
        color: #0369A1;
        font-size: 1.9rem;
    }
    .metric-card p {
        margin: 0;
        color: #64748B;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0284C7;
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0369A1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. INISIALISASI SESSION STATE & PEMUATAN MODEL AMAN (DENGAN GDOWN)
# ==============================================================================
if 'history' not in st.session_state:
    st.session_state['history'] = []

@st.cache_resource
def safe_load_model():
    """Fungsi deteksi model lokal & unduh otomatis via gdown jika file belum ada."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = "resnet50_pneumonia_final_70_15_15.h5"
    model_path = os.path.join(base_dir, model_filename)

    # GDRIVE FILE ID untuk bobot model Softmax
    GDRIVE_FILE_ID = '1klZjZsqH18YqoeKBqCld1nmYIBBY_27a' 

    candidates = [
        model_filename,
        'resnet50_pneumonia_final.h5',
        'model_resnet50.h5',
        'resnet50_model.h5'
    ]

    found_path = None
    for candidate in candidates:
        full_candidate_path = os.path.join(base_dir, candidate)
        if os.path.exists(full_candidate_path):
            found_path = full_candidate_path
            break

    if not found_path:
        if GDRIVE_FILE_ID and GDRIVE_FILE_ID != 'MASUKKAN_GOOGLE_DRIVE_FILE_ID_DISINI':
            with st.spinner("Mengunduh berkas bobot model ResNet-50 dari Google Drive..."):
                url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
                gdown.download(url, model_path, quiet=False)
            found_path = model_path
        else:
            raise FileNotFoundError(
                f"Berkas model dengan format .h5 tidak ditemukan di direktori `{base_dir}`. "
                f"Pastikan Anda menempatkan file '{model_filename}' sejajar dengan file `app.py` atau isi `GDRIVE_FILE_ID`."
            )

    custom_objects = {'Dense': FixedDense, 'FixedDense': FixedDense}
    loaded_model = models.load_model(found_path, custom_objects=custom_objects, compile=False)
    return loaded_model, os.path.basename(found_path)

try:
    with st.spinner("Memuat arsitektur Deep Learning ResNet-50 & bobot pretrained..."):
        model, loaded_filename = safe_load_model()
except Exception as e:
    st.error(f"⚠️ **Gagal memuat berkas bobot model ResNet-50.**")
    st.caption(f"Detail kendala teknis: {e}")
    st.info("""
    **Panduan Perbaikan Cepat:**
    1. Letakkan file bobot berformat `.h5` Anda di dalam folder yang sama persis dengan file `app.py` ini.
    2. Pastikan penamaan file sesuai (contoh: `resnet50_pneumonia_final_70_15_15.h5`).
    """)
    st.stop()

# ==============================================================================
# 4. SIDEBAR NAVIGASI & IDENTITAS PENELITI
# ==============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=85)
st.sidebar.title("CDSS Pneumonia")
st.sidebar.markdown("**Clinical Decision Support System**")

menu = st.sidebar.radio(
    "Navigasi Sistem:",
    [
        "🏠 Beranda & Identitas",
        "🔍 Diagnosis Citra Interaktif",
        "📊 Performa & Evaluasi Model",
        "📐 Metodologi CRISP-DM",
        "📚 Glosarium & FAQ"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parameter Diagnosa")
st.sidebar.info("**Aktivasi Layer Output:** Softmax (2 Units)\n\n**Metode Klasifikasi:** Argmax Probability Vector $[P_0, P_1]$")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍🎓 Identitas Peneliti")
st.sidebar.caption("**Nama:** Naufal Ardra Anabil")
st.sidebar.caption("**NPM:** 51422215")
st.sidebar.caption("**Kelas:** 4IA14")
st.sidebar.caption("**Institusi:** Universitas Gunadarma")
st.sidebar.caption("**Arsitektur:** ResNet-50 (CNN)")

# ==============================================================================
# 5. HALAMAN 1: BERANDA & IDENTITAS
# ==============================================================================
if menu == "🏠 Beranda & Identitas":
    st.markdown("<p class='main-header'>PENERAPAN CNN UNTUK KLASIFIKASI PENYAKIT PNEUMONIA PADA CITRA X-RAY PARU-PARU MENGGUNAKAN ARSITEKTUR RESNET-50</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Sistem Pendukung Keputusan Medis Berbasis Web Menggunakan Metodologi CRISP-DM</p>", unsafe_allow_html=True)

    st.markdown("""
        <div class='identity-box'>
            <table style='width:100%; border-collapse: collapse; font-size: 0.95rem;'>
                <tr>
                    <td style='width: 15%; font-weight: bold;'>Nama Peneliti</td>
                    <td style='width: 2%;'>:</td>
                    <td style='width: 33%;'>Naufal Ardra Anabil</td>
                    <td style='width: 15%; font-weight: bold;'>NPM</td>
                    <td style='width: 2%;'>:</td>
                    <td style='width: 33%;'>51422215</td>
                </tr>
                <tr>
                    <td style='font-weight: bold;'>Kelas</td>
                    <td>:</td>
                    <td>4IA14</td>
                    <td style='font-weight: bold;'>Program Studi</td>
                    <td>:</td>
                    <td>Informatics Engineering / Teknik Informatika</td>
                </tr>
                <tr>
                    <td style='font-weight: bold;'>Perguruan Tinggi</td>
                    <td>:</td>
                    <td>Universitas Gunadarma</td>
                    <td style='font-weight: bold;'>Metodologi</td>
                    <td>:</td>
                    <td>CRISP-DM</td>
                </tr>
            </table>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("📌 Gambaran Umum Penelitian")
        st.write("""
        Pneumonia merupakan infeksi jaringan paru-paru yang menjadi salah satu penyebab utama kematian pada anak-anak dan lansia di seluruh dunia. Diagnosa konvensional dilakukan melalui analisis citra *Chest X-Ray* (CXR) oleh dokter spesialis radiologi, yang membutuhkan waktu serta rentan terhadap variabilitas subjektif.

        Aplikasi **Clinical Decision Support System (CDSS)** ini dikembangkan untuk mengimplementasikan teknologi *Deep Learning* berbasis arsitektur **ResNet-50** dengan strategi *Transfer Learning* dan fungsi aktivasi **Softmax**. Sistem dirancang untuk membantu memberikan opini sekunder (*second opinion*) secara instan, presisi, dan terukur.
        """)
        st.info("💡 **Petunjuk Akses:** Gunakan menu **🔍 Diagnosis Citra Interaktif** pada bilah navigasi kiri untuk melakukan pengujian citra X-Ray paru-paru.")

    with col2:
        st.subheader("📈 Ringkasan Metrik Performa (Test Set)")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("<div class='metric-card'><h2>82.71%</h2><p>Akurasi Keseluruhan</p></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='metric-card'><h2>89.86%</h2><p>Recall (Pneumonia)</p></div>", unsafe_allow_html=True)
        with m_col2:
            st.markdown("<div class='metric-card'><h2>86.88%</h2><p>Presisi (Pneumonia)</p></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='metric-card'><h2>88.34%</h2><p>F1-Score (Pneumonia)</p></div>", unsafe_allow_html=True)

# ==============================================================================
# 6. HALAMAN 2: DIAGNOSIS CITRA INTERAKTIF (LOGIKA SOFTMAX)
# ==============================================================================
elif menu == "🔍 Diagnosis Citra Interaktif":
    st.title("🔍 Diagnosis Citra X-Ray Paru-Paru")
    st.write("Unggah berkas citra X-Ray (CXR) untuk dianalisis oleh jaringan saraf tiruan ResNet-50 berbasis **Softmax**.")
    st.markdown("---")

    col_upload, col_result = st.columns([1, 1.1])

    with col_upload:
        st.subheader("📁 Unggah & Filter Citra")
        uploaded_file = st.file_uploader(
            "Pilih berkas gambar X-Ray...",
            type=["jpg", "jpeg", "png"],
            help="Pastikan gambar menunjukkan area dada/toraks dengan kejelasan struktur paru-paru."
        )

        if uploaded_file is not None:
            image_raw = Image.open(uploaded_file).convert('RGB')

            st.markdown("**Opsi Manipulasi Visual (Bantu Visualisasi):**")
            filter_type = st.radio("Pilih Filter Visual:", ["Asli (Original)", "Grayscale", "High Contrast"], horizontal=True)

            if filter_type == "Grayscale":
                display_img = ImageOps.grayscale(image_raw)
            elif filter_type == "High Contrast":
                enhancer = ImageEnhance.Contrast(image_raw)
                display_img = enhancer.enhance(1.8)
            else:
                display_img = image_raw

            st.image(display_img, caption=f"Preview Citra: {uploaded_file.name}", use_container_width=True)

            with st.expander("ℹ️ Metadata Berkas Citra"):
                st.write(f"**Nama File:** {uploaded_file.name}")
                st.write(f"**Dimensi Piksel:** {image_raw.width} x {image_raw.height} px")
                st.write(f"**Ukuran Berkas:** {uploaded_file.size / 1024:.2f} KB")
                st.write(f"**Format Warna:** {image_raw.mode}")

            btn_predict = st.button("🚀 Jalankan Diagnosa Model")

    with col_result:
        st.subheader("📊 Hasil Diagnosa & Inferensi")

        if uploaded_file is not None and ('btn_predict' in locals() and btn_predict):
            with st.spinner("Menjalankan preprocessing (224x224, Normalisasi) dan eksekusi ResNet-50 (Softmax)..."):
                time.sleep(0.3)

                # 1. Preprocessing
                img_resized = image_raw.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)

                # 2. Prediction (Array Softmax 2 Kolom)
                predictions = model.predict(img_batch)[0] # Shape: [2]
                
                prob_normal = float(predictions[0]) * 100
                prob_pneumonia = float(predictions[1]) * 100

                predicted_class_idx = np.argmax(predictions)

                if predicted_class_idx == 1:
                    label = "PNEUMONIA"
                    confidence = prob_pneumonia
                else:
                    label = "NORMAL"
                    confidence = prob_normal

                # Formatting string kepastian untuk keterbacaan yang rapi
                if confidence > 99.99:
                    conf_str = ">99.99%"
                elif confidence < 0.01:
                    conf_str = "<0.01%"
                else:
                    conf_str = f"{confidence:.2f}%"

                # 3. Log History Session
                st.session_state['history'].append({
                    "Nama Berkas": uploaded_file.name,
                    "Hasil Diagnosa": label,
                    "Kepastian": conf_str,
                    "Prob. Normal": f"{prob_normal:.4f}%",
                    "Prob. Pneumonia": f"{prob_pneumonia:.4f}%",
                    "Waktu": time.strftime("%H:%M:%S")
                })

            # Tampilan Banner Hasil
            if label == "PNEUMONIA":
                st.error(f"### ⚠️ DIAGNOSA: {label}")
                st.markdown(f"Sistem mengidentifikasi indikasi **Pneumonia** pada citra X-Ray dengan tingkat kepastian Softmax **{conf_str}**.")
            else:
                st.success(f"### ✅ DIAGNOSA: {label}")
                st.markdown(f"Sistem menandai citra X-Ray ini dalam kondisi **Normal** with tingkat kepastian Softmax **{conf_str}**.")

            st.markdown("---")
            st.markdown("**Distribusi Probabilitas Kelas (Softmax Output):**")
            
            # Progress bar Probabilitas Normal
            if prob_normal > 99.99:
                st.write("**Probabilitas NORMAL:** >99.99% *(Sangat Tinggi)*")
            elif prob_normal < 0.01:
                st.write("**Probabilitas NORMAL:** <0.01%")
            else:
                st.write(f"**Probabilitas NORMAL:** {prob_normal:.2f}%")
            st.progress(min(max(int(prob_normal), 0), 100))

            # Progress bar Probabilitas Pneumonia
            if prob_pneumonia > 99.99:
                st.write("**Probabilitas PNEUMONIA:** >99.99% *(Sangat Tinggi)*")
            elif prob_pneumonia < 0.01:
                st.write("**Probabilitas PNEUMONIA:** <0.01%")
            else:
                st.write(f"**Probabilitas PNEUMONIA:** {prob_pneumonia:.2f}%")
            st.progress(min(max(int(prob_pneumonia), 0), 100))

            st.caption(f"ℹ️ *Nilai Eksak:* Normal = `{prob_normal:.6f}%` | Pneumonia = `{prob_pneumonia:.6f}%` | *Activation:* `Softmax`")

            st.markdown("---")
            report_text = f"""=====================================================================
LAPORAN HASIL DIAGNOSA PENDUKUNG KEPUTUSAN (CDSS)
=====================================================================
JUDUL PENELITIAN:
PENERAPAN CNN UNTUK KLASIFIKASI PENYAKIT PNEUMONIA PADA CITRA X-RAY
PARU-PARU MENGGUNAKAN ARSITEKTUR RESNET-50 BERBASIS DEEP LEARNING

IDENTITAS PENELITI:
* Nama Peneliti : Naufal Ardra Anabil
* NPM           : 51422215
* Kelas         : 4IA14
* Institusi     : Universitas Gunadarma

DETAIL PENGUJIAN:
* Tanggal / Waktu   : {time.strftime("%Y-%m-%d %H:%M:%S")}
* Nama Berkas Citra : {uploaded_file.name}
* Model Arsitektur  : ResNet-50 (Transfer Learning)
* Activation Output : Softmax (2 Units)

---------------------------------------------------------------------
HASIL DIAGNOSA UTAMA : {label}
TINGKAT KEPASTIAN    : {conf_str}
---------------------------------------------------------------------
Probabilitas Normal   : {prob_normal:.6f}%
Probabilitas Pneumonia: {prob_pneumonia:.6f}%

CATATAN MEDIS:
Hasil ini dihasilkan secara otomatis oleh sistem kecerdasan buatan
dan bersifat sebagai alat bantu (second opinion). Diperlukan konfirmasi
lanjutan oleh Dokter Spesialis Radiologi.
====================================================================="""

            st.download_button(
                label="📥 Unduh Laporan Teks Diagnosa (.txt)",
                data=report_text,
                file_name=f"Laporan_Diagnosa_{uploaded_file.name}.txt",
                mime="text/plain"
            )
            st.warning("⚠️ **Catatan Medis:** Luaran aplikasi ini tidak menggantikan kepastian klinis dari dokter spesialis radiologi.")
        else:
            st.info("👈 Silakan unggah citra X-Ray di panel sebelah kiri lalu tekan tombol **Jalankan Diagnosa Model**.")

    if len(st.session_state['history']) > 0:
        st.markdown("---")
        st.subheader("📜 Riwayat Diagnosa Sesi Ini")
        df_history = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_history, use_container_width=True)

# ==============================================================================
# 7. HALAMAN 3: PERFORMA & EVALUASI MODEL
# ==============================================================================
elif menu == "📊 Performa & Evaluasi Model":
    st.title("📊 Laporan Performa Kuantitatif Model")
    st.write("Hasil pengujian komprehensif model ResNet-50 (Softmax) pada *Test Set* independen sebanyak 879 citra X-Ray.")
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📑 Laporan Klasifikasi (*Classification Report*)")
        report_data = {
            "Kelas Diagnosa": ["NORMAL", "PNEUMONIA", "Accuracy", "Macro Average", "Weighted Average"],
            "Precision": ["0.73", "0.85", "-", "0.79", "0.82"],
            "Recall": ["0.58", "0.92", "-", "0.75", "0.83"],
            "F1-Score": ["0.64", "0.89", "0.83", "0.77", "0.82"],
            "Support": [238, 641, 879, 879, 879]
        }
        df_report = pd.DataFrame(report_data)
        st.table(df_report)
        st.markdown("""
        **Analisis Metrik Medis:**
        * **Presisi Tinggi pada Pneumonia (85.40%):** Menunjukkan bahwa saat model memprediksi pasien menderita Pneumonia, kemungkinan besar prediksi tersebut tepat.
        * **Recall Pneumonia (92.20%):** Kemampuan model menjaring kembali kasus Pneumonia nyata dari total sampel yang diuji.
        """)

    with col2:
        st.subheader("🧩 Confusion Matrix Hasil Pengujian")
        cm_data = {
            "Prediksi NORMAL": [151, 87],
            "Prediksi PNEUMONIA": [65, 576]
        }
        df_cm = pd.DataFrame(cm_data, index=["Aktual NORMAL", "Aktual PNEUMONIA"])
        st.table(df_cm)
        st.caption("""
        * **True Negative (TN):** 151 citra Normal terprediksi Normal.
        * **False Positive (FP):** 87 citra Normal salah terprediksi Pneumonia.
        * **False Negative (FN):** 65 citra Pneumonia salah terprediksi Normal.
        * **True Positive (TP):** 576 citra Pneumonia terprediksi Pneumonia.
        """)

# ==============================================================================
# 8. HALAMAN 4: METODOLOGI CRISP-DM
# ==============================================================================
elif menu == "📐 Metodologi CRISP-DM":
    st.title("📐 Metodologi Penelitian (CRISP-DM)")
    st.write("Siklus pengembangan sistem standar industri yang diterapkan dalam penelitian skripsi oleh **Naufal Ardra Anabil (NPM: 51422215)**.")
    st.markdown("---")

    st.markdown("""
    #### 1. Business & Medical Understanding
    * **Masalah:** Tingginya beban kerja radiolog dan kebutuhan akan konfirmasi instan diagnosa Pneumonia dari citra X-Ray paru-paru.
    * **Solusi:** Membangun *Clinical Decision Support System* (CDSS) berbasis jaringan saraf konvolusional (CNN).

    #### 2. Data Understanding
    * Menggunakan dataset citra X-Ray dada (*Chest X-Ray*) terbagi menjadi 2 kategori utama: **NORMAL** dan **PNEUMONIA**.

    #### 3. Data Preparation
    * **Resizing:** Mengubah dimensi citra menjadi $224 \\times 224 \\times 3$ piksel sesuai prasyarat input ResNet-50.
    * **Normalisasi:** Membagi intensitas piksel $[0-255]$ dengan $255.0$ sehingga bernilai $[0.0, 1.0]$.

    #### 4. Modeling
    * **Base Model:** ResNet-50 dengan pretrained weights `imagenet` yang dibekukan (*frozen*).
    * **Top Layers Customization:** Penambahan *GlobalAveragePooling2D*, *Dense Layer (256, activation='relu')*, *Dropout (0.5)*, dan *Output Dense Layer (2, activation='softmax')*.

    #### 5. Evaluation
    * Evaluasi performa menggunakan *Confusion Matrix*, *Accuracy*, *Precision*, *Recall*, dan *F1-Score*.

    #### 6. Deployment
    * Pengemasan model ke dalam format `.h5` dan diintegrasikan ke antarmuka aplikasi interaktif berbasis **Streamlit**.
    """)

# ==============================================================================
# 9. HALAMAN 5: GLOSARIUM & FAQ
# ==============================================================================
elif menu == "📚 Glosarium & FAQ":
    st.title("📚 Glosarium Medis & FAQ")
    st.write("Penjelasan istilah teknis serta pertanyaan umum mengenai penelitian skripsi ini.")
    st.markdown("---")

    st.subheader("📖 Glosarium Istilah")
    st.markdown("""
    * **Pneumonia:** Infeksi akut pada salah satu atau kedua paru-paru yang disebabkan oleh bakteri, virus, atau jamur.
    * **Chest X-Ray (CXR):** Pemeriksaan radiologi menggunakan sinar-X untuk mengevaluasi organ paru-paru.
    * **ResNet-50:** Arsitektur Convolutional Neural Network (CNN) setebal 50 layer yang memanfaatkan *skip connections*.
    * **Transfer Learning:** Pemanfaatan bobot dari model pretrained ImageNet untuk diterapkan pada domain klasifikasi medis.
    * **Softmax Activation:** Fungsi aktivasi layer output yang mengubah output logit menjadi vektor distribusi probabilitas multinomial yang bernilai total 1.0 (100%).
    """)

    st.markdown("---")
    st.subheader("❓ FAQ (Frequently Asked Questions)")
    with st.expander("Siapa pengembang aplikasi ini?"):
        st.write("Aplikasi ini dikembangkan oleh **Naufal Ardra Anabil** (NPM: 51422215, Kelas: 4IA14), Mahasiswa Teknik Informatika Universitas Gunadarma, sebagai bagian dari penulisan Skripsi/Tugas Akhir.")
    with st.expander("Apakah aplikasi ini dapat menggantikan dokter spesialis radiologi?"):
        st.write("Tidak. Aplikasi ini dirancang secara ketat sebagai *Clinical Decision Support System* (CDSS) atau opini sekunder untuk membantu mempercepat screening awal.")
    with st.expander("Mengapa ukuran gambar otomatis diubah menjadi 224x224 piksel?"):
        st.write("Arsitektur dasar ResNet-50 yang dilatih menggunakan ImageNet dilatih secara standar untuk menerima input matriks berukuran 224 x 224 piksel dengan 3 kanal warna (RGB).")