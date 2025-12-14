import streamlit as st
import pandas as pd
import joblib

# =====================================================
# CONFIG
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

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <h1 style='margin-bottom:0;'>💼 SalarySense</h1>
    <p style='color:gray; margin-top:4px;'>
    Sistem prediksi dan proyeksi gaji berbasis Machine Learning untuk membantu
    memahami nilai dan potensi karier Anda.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# INPUT SECTION
# =====================================================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("📋 Profil Pekerja")

    usia = st.slider("Usia", 18, 65, 25)
    pengalaman = st.slider("Pengalaman Kerja (tahun)", 0, 40, 1)

    pendidikan = st.selectbox(
        "Pendidikan",
        sorted(df["Education Level"].dropna().unique())
    )

    pekerjaan = st.selectbox(
        "Pekerjaan",
        sorted(df["Job Title"].dropna().unique())
    )

    gender = st.radio(
        "Jenis Kelamin",
        sorted(df["Gender"].dropna().unique()),
        horizontal=True
    )

    if st.button("🚀 Prediksi Gaji", use_container_width=True):
        # Ambil 1 baris asli agar struktur pasti cocok
        input_df = df.drop(columns=["Salary"]).iloc[[0]].copy()

        input_df["Age"] = usia
        input_df["Years of Experience"] = pengalaman
        input_df["Education Level"] = pendidikan
        input_df["Job Title"] = pekerjaan
        input_df["Gender"] = gender

        try:
            st.session_state.input_df = input_df
            st.session_state.gaji = model.predict(input_df)[0]
            st.session_state.predicted = True
        except Exception as e:
            st.error("❌ Prediksi gagal. Struktur input tidak sesuai model.")
            st.stop()

# =====================================================
# OUTPUT UTAMA
# =====================================================
with right:
    st.subheader("📊 Hasil Prediksi")

    if st.session_state.get("predicted", False):
        gaji = st.session_state.gaji
        avg_market = df[df["Job Title"] == pekerjaan]["Salary"].mean()

        st.metric(
            "💰 Estimasi Gaji Tahunan",
            f"${gaji:,.0f}"
        )

        if gaji < avg_market * 0.9:
            posisi = "🔻 Di bawah rata-rata pasar"
        elif gaji > avg_market * 1.1:
            posisi = "🔺 Di atas rata-rata pasar"
        else:
            posisi = "✅ Kompetitif di pasar kerja"

        st.info(f"**Posisi Pasar:** {posisi}")
    else:
        st.info("Masukkan data dan klik **Prediksi Gaji**")

# =====================================================
# PROYEKSI KARIER
# =====================================================
if st.session_state.get("predicted", False):
    st.divider()
    st.subheader("📈 Proyeksi Pertumbuhan Gaji (10 Tahun)")

    tahun = list(range(11))
    proyeksi = []

    for t in tahun:
        temp = st.session_state.input_df.copy()
        temp["Years of Experience"] += t
        proyeksi.append(model.predict(temp)[0])

    chart_df = pd.DataFrame({
        "Tahun": tahun,
        "Estimasi Gaji": proyeksi
    }).set_index("Tahun")

    st.line_chart(chart_df)

# =====================================================
# INSIGHT & REKOMENDASI (INI YANG TADI HILANG)
# =====================================================
if st.session_state.get("predicted", False):
    st.divider()
    st.subheader("🧠 Insight & Rekomendasi Karier")

    rekomendasi = []

    if pengalaman < 3:
        rekomendasi.append(
            "🔹 Pengalaman kerja masih relatif rendah. Fokus menambah jam terbang dan skill praktis."
        )

    if pendidikan in ["High School", "Bachelor"]:
        rekomendasi.append(
            "🎓 Peningkatan pendidikan atau sertifikasi profesional berpotensi menaikkan gaji."
        )

    if posisi.startswith("🔻"):
        rekomendasi.append(
            "💬 Gaji Anda berada di bawah rata-rata pasar. Pertimbangkan negosiasi atau eksplorasi peluang lain."
        )

    if not rekomendasi:
        rekomendasi.append(
            "✅ Profil Anda sudah cukup kompetitif. Pertahankan dan tingkatkan spesialisasi."
        )

    for r in rekomendasi:
        st.write(r)

    st.caption(
        "Catatan: Prediksi bersifat estimasi berdasarkan data historis dan tidak menjamin nilai absolut."
    )
