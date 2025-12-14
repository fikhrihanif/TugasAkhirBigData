import streamlit as st
import pandas as pd
import joblib

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="SalarySense",
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
# UI
# =====================================================
st.title("💼 SalarySense")
st.caption("Prediksi gaji berbasis Machine Learning")

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

    if st.button("🚀 Prediksi Gaji"):
        # =====================================================
        # FIX UTAMA: AMBIL ROW ASLI DATASET
        # =====================================================
        input_df = df.drop(columns=["Salary"]).iloc[[0]].copy()

        # TIMPA DENGAN INPUT USER
        input_df["Age"] = usia
        input_df["Years of Experience"] = pengalaman
        input_df["Education Level"] = pendidikan
        input_df["Job Title"] = pekerjaan
        input_df["Gender"] = gender

        try:
            gaji = model.predict(input_df)[0]
            st.session_state.gaji = gaji
            st.session_state.input_df = input_df
            st.session_state.predicted = True
        except Exception as e:
            st.error("❌ Prediksi gagal. Model tidak menerima input.")
            st.exception(e)
            st.stop()

# =====================================================
# OUTPUT
# =====================================================
with right:
    st.subheader("📊 Hasil")

    if st.session_state.get("predicted", False):
        gaji = st.session_state.gaji
        st.metric("💰 Estimasi Gaji Tahunan", f"${gaji:,.0f}")

        avg = df[df["Job Title"] == pekerjaan]["Salary"].mean()

        if gaji < avg * 0.9:
            posisi = "🔻 Di bawah rata-rata pasar"
        elif gaji > avg * 1.1:
            posisi = "🔺 Di atas rata-rata pasar"
        else:
            posisi = "✅ Kompetitif"

        st.info(f"Posisi pasar: **{posisi}**")
    else:
        st.info("Masukkan data dan klik Prediksi Gaji")

# =====================================================
# PROYEKSI
# =====================================================
if st.session_state.get("predicted", False):
    st.subheader("📈 Proyeksi 10 Tahun")

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
