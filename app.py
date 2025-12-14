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
# LOAD MODEL & DATASET
# =====================================================
model = joblib.load("salary_prediction_model.pkl")
df = pd.read_csv("Salary_Data.csv")

job_titles = sorted(df["Job Title"].dropna().unique())

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
    cursor: pointer !important;
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
        Sistem prediksi dan proyeksi gaji berbasis <b>Machine Learning</b>  
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
    pendidikan = st.selectbox("Pendidikan", ["High School", "Bachelor", "Master", "PhD"])
    pekerjaan = st.selectbox("Pekerjaan", job_titles)
    gender = st.radio("Jenis Kelamin", ["Male", "Female"], horizontal=True)

    if st.button("🚀 Prediksi Gaji"):
        st.session_state.input_df = pd.DataFrame({
            "Age": [usia],
            "Years of Experience": [pengalaman],
            "Education Level": [pendidikan],
            "Job Title": [pekerjaan],
            "Gender": [gender]
        })

        st.session_state.gaji = model.predict(st.session_state.input_df)[0]
        st.session_state.predicted = True

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

        # === POSISI PASAR (2)
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

    tahun = list(range(0, 11))
    gaji_proyeksi = []

    for t in tahun:
        temp_df = st.session_state.input_df.copy()
        temp_df["Years of Experience"] += t
        gaji_proyeksi.append(model.predict(temp_df)[0])

    proyeksi_df = pd.DataFrame({
        "Tahun": tahun,
        "Estimasi Gaji": gaji_proyeksi
    }).set_index("Tahun")

    st.line_chart(proyeksi_df)

    growth_rate = ((gaji_proyeksi[-1] / gaji_proyeksi[0]) ** (1 / 10) - 1) * 100

    st.metric(
        "Rata-rata pertumbuhan gaji per tahun",
        f"{growth_rate:.2f} %"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PENJELASAN & REKOMENDASI
# =====================================================
if st.session_state.predicted:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>Insight</span>", unsafe_allow_html=True)
    st.subheader("🧠 Analisis & Rekomendasi")

    st.write("""
    **Faktor utama yang memengaruhi hasil prediksi:**
    - **Pengalaman kerja**: semakin lama pengalaman, semakin tinggi potensi gaji.
    - **Tingkat pendidikan**: pendidikan yang lebih tinggi meningkatkan baseline pendapatan.
    - **Jenis pekerjaan**: mencerminkan permintaan dan kompleksitas peran di pasar kerja.
    """)

    st.write("**Rekomendasi pengembangan karier:**")

    rekomendasi = []

    if pengalaman < 3:
        rekomendasi.append("🔹 Fokus menambah pengalaman kerja dan skill praktis.")
    if pendidikan in ["High School", "Bachelor"]:
        rekomendasi.append("🔹 Pertimbangkan peningkatan pendidikan atau sertifikasi profesional.")
    if posisi.startswith("🔻"):
        rekomendasi.append("🔹 Pertimbangkan negosiasi gaji atau eksplorasi peluang di perusahaan lain.")

    if not rekomendasi:
        rekomendasi.append("🔹 Pertahankan jalur karier Anda, posisi Anda sudah kompetitif.")

    for r in rekomendasi:
        st.write(r)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    © 2025 • SalarySense  
    <br>Final Project • Machine Learning
</div>
""", unsafe_allow_html=True)
