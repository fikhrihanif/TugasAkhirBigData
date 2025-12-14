import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SalarySense • Prediksi Gaji",
    page_icon="💼",
    layout="wide"
)

# =====================================================
# LOAD MODEL & DATA
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("salary_prediction_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Salary_Data.csv")

model = load_model()
df = load_data()

# Pastikan tipe data kategori aman
for col in ["Job Title", "Education Level", "Gender"]:
    df[col] = df[col].astype(str)

job_titles = sorted(df["Job Title"].unique())
education_levels = sorted(df["Education Level"].unique())
genders = sorted(df["Gender"].unique())

# =====================================================
# SESSION STATE
# =====================================================
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# =====================================================
# GLOBAL CSS
# =====================================================
st.markdown("""
<style>
.main { background-color: #020617; }
#MainMenu, footer, header { visibility: hidden; }

.card {
    background-color: #020617;
    padding: 2rem;
    border-radius: 18px;
    border: 1px solid #1e293b;
    box-shadow: 0 12px 32px rgba(0,0,0,0.55);
    margin-bottom: 1.6rem;
}

h1, h2, h3 { color: #e5e7eb; }
p, label, li { color: #cbd5f5; }

input, select {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}

.stButton>button {
    background: linear-gradient(135deg, #6366f1, #3b82f6);
    color: white;
    border-radius: 16px;
    padding: 0.9em;
    font-weight: 700;
    width: 100%;
}

.metric-box {
    background-color: #020617;
    padding: 1.6rem;
    border-radius: 16px;
    border: 1px solid #1e293b;
    text-align: center;
}

.badge {
    background-color: #1e293b;
    color: #c7d2fe;
    padding: 0.35em 0.8em;
    border-radius: 999px;
    font-size: 0.8rem;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 3rem;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO
# =====================================================
st.markdown("""
<div style="text-align:center; padding:3rem 0;">
    <h1 style="font-size:3rem;">💼 SalarySense</h1>
    <p style="font-size:1.15rem; max-width:720px; margin:auto;">
        Sistem prediksi dan proyeksi gaji berbasis <b>Machine Learning</b><br>
        untuk membantu memahami nilai dan potensi karier di pasar kerja.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT
# =====================================================
left, right = st.columns([1.2, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>Profil</span>", unsafe_allow_html=True)

    usia = st.slider("Usia", 18, 65, 25)
    pengalaman = st.slider("Pengalaman Kerja (tahun)", 0, 40, 1)
    pendidikan = st.selectbox("Pendidikan", education_levels)
    pekerjaan = st.selectbox("Pekerjaan", job_titles)
    gender = st.radio("Jenis Kelamin", genders, horizontal=True)

    if st.button("🚀 Prediksi Gaji"):
        # ===============================
        # FIX UTAMA: PAKSA STRUKTUR INPUT
        # ===============================
        raw_input = pd.DataFrame([{
            "Age": float(usia),
            "Years of Experience": float(pengalaman),
            "Education Level": str(pendidikan),
            "Job Title": str(pekerjaan),
            "Gender": str(gender)
        }])

        # PAKSA kolom SAMA PERSIS seperti training
        input_df = raw_input.reindex(
            columns=model.feature_names_in_,
            fill_value=np.nan
        )

        st.session_state.input_df = input_df

        try:
            st.session_state.gaji = model.predict(input_df)[0]
            st.session_state.predicted = True
        except Exception as e:
            st.error("❌ Terjadi kesalahan prediksi. Struktur input tidak sesuai model.")
            st.stop()

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# OUTPUT UTAMA
# =====================================================
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>Hasil</span>", unsafe_allow_html=True)

    if st.session_state.predicted:
        gaji = st.session_state.gaji
        min_gaji = gaji * 0.9
        max_gaji = gaji * 1.1

        st.markdown(f"""
        <div class="metric-box">
            <h2>💰 ${gaji:,.0f}</h2>
            <p>Estimasi gaji tahunan</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("Batas Bawah Wajar", f"${min_gaji:,.0f}")
        c2.metric("Batas Atas Wajar", f"${max_gaji:,.0f}")

        avg_market = df[df["Job Title"] == pekerjaan]["Salary"].mean()

        if gaji < avg_market * 0.9:
            posisi = "🔻 Di bawah rata-rata pasar"
        elif gaji > avg_market * 1.1:
            posisi = "🔺 Di atas rata-rata pasar"
        else:
            posisi = "✅ Kompetitif di pasar kerja"

        st.info(f"📌 Posisi gaji Anda: **{posisi}**")

    else:
        st.info("Silakan isi profil dan klik **Prediksi Gaji**")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PROYEKSI KARIER
# =====================================================
if st.session_state.predicted:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>Proyeksi</span>", unsafe_allow_html=True)
    st.subheader("📈 Proyeksi Pertumbuhan Gaji")

    tahun = range(11)
    gaji_proyeksi = []

    for t in tahun:
        temp_df = st.session_state.input_df.copy()
        temp_df["Years of Experience"] += t
        gaji_proyeksi.append(model.predict(temp_df)[0])

    proyeksi_df = pd.DataFrame({
        "Tahun Pengalaman Tambahan": list(tahun),
        "Estimasi Gaji": gaji_proyeksi
    }).set_index("Tahun Pengalaman Tambahan")

    st.line_chart(proyeksi_df)

    growth_rate = ((gaji_proyeksi[-1] / gaji_proyeksi[0]) ** (1 / 10) - 1) * 100

    st.metric(
        "Rata-rata pertumbuhan gaji per tahun",
        f"{growth_rate:.2f}%"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# INSIGHT & REKOMENDASI
# =====================================================
if st.session_state.predicted:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>Insight</span>", unsafe_allow_html=True)
    st.subheader("🧠 Analisis & Rekomendasi")

    st.write("""
    **Faktor utama yang memengaruhi prediksi:**
    - Pengalaman kerja
    - Tingkat pendidikan
    - Jenis dan kompleksitas pekerjaan
    """)

    rekomendasi = []

    if pengalaman < 3:
        rekomendasi.append("🔹 Tingkatkan pengalaman dan skill praktis.")
    if pendidikan in ["High School", "Bachelor"]:
        rekomendasi.append("🔹 Pertimbangkan sertifikasi atau pendidikan lanjutan.")
    if posisi.startswith("🔻"):
        rekomendasi.append("🔹 Pertimbangkan negosiasi gaji atau peluang baru.")

    if not rekomendasi:
        rekomendasi.append("🔹 Jalur karier Anda sudah kompetitif, pertahankan konsistensi.")

    for r in rekomendasi:
        st.write(r)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    © 2025 • SalarySense<br>
    Final Project • Machine Learning
</div>
""", unsafe_allow_html=True)
