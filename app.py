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
@st.cache_resource
def load_model():
    return joblib.load("salary_prediction_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Salary_Data.csv")

model = load_model()
df = load_data()

# ===============================
# PENTING: DEFINISI FITUR = SAMA DENGAN TRAINING
# ===============================
FEATURE_COLUMNS = df.drop(columns=["Salary"]).columns.tolist()

# Pisahkan tipe fitur (SESUAI TRAINING)
numerical_features = df[FEATURE_COLUMNS].select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = df[FEATURE_COLUMNS].select_dtypes(
    include=["object"]
).columns.tolist()

# Pastikan kategori string
for col in categorical_features:
    df[col] = df[col].astype(str)

# =====================================================
# SESSION STATE
# =====================================================
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# =====================================================
# UI
# =====================================================
st.title("💼 SalarySense")
st.caption("Prediksi dan proyeksi gaji berbasis Machine Learning")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("📋 Profil Pekerja")

    usia = st.slider("Usia", 18, 65, 25)
    pengalaman = st.slider("Pengalaman Kerja (tahun)", 0, 40, 1)

    pendidikan = st.selectbox(
        "Tingkat Pendidikan",
        sorted(df["Education Level"].unique())
    )

    pekerjaan = st.selectbox(
        "Pekerjaan",
        sorted(df["Job Title"].unique())
    )

    gender = st.radio(
        "Jenis Kelamin",
        sorted(df["Gender"].unique()),
        horizontal=True
    )

    if st.button("🚀 Prediksi Gaji"):
        # ===============================
        # FIX FINAL: BANGUN INPUT SESUAI TRAINING
        # ===============================
        input_data = {col: np.nan for col in FEATURE_COLUMNS}

        input_data["Age"] = float(usia)
        input_data["Years of Experience"] = float(pengalaman)
        input_data["Education Level"] = str(pendidikan)
        input_data["Job Title"] = str(pekerjaan)
        input_data["Gender"] = str(gender)

        input_df = pd.DataFrame([input_data])

        try:
            st.session_state.gaji = model.predict(input_df)[0]
            st.session_state.input_df = input_df
            st.session_state.predicted = True
        except Exception as e:
            st.error("❌ Prediksi gagal. Struktur input tidak sesuai model.")
            st.stop()

# =====================================================
# OUTPUT
# =====================================================
with right:
    st.subheader("📊 Hasil Prediksi")

    if st.session_state.predicted:
        gaji = st.session_state.gaji

        st.metric(
            label="💰 Estimasi Gaji Tahunan",
            value=f"${gaji:,.0f}"
        )

        avg_market = df[df["Job Title"] == pekerjaan]["Salary"].mean()

        if gaji < avg_market * 0.9:
            posisi = "🔻 Di bawah rata-rata pasar"
        elif gaji > avg_market * 1.1:
            posisi = "🔺 Di atas rata-rata pasar"
        else:
            posisi = "✅ Kompetitif di pasar kerja"

        st.info(f"Posisi gaji Anda: **{posisi}**")

# =====================================================
# PROYEKSI KARIER
# =====================================================
if st.session_state.predicted:
    st.subheader("📈 Proyeksi Gaji 10 Tahun")

    tahun = range(11)
    proyeksi = []

    for t in tahun:
        temp_df = st.session_state.input_df.copy()
        temp_df["Years of Experience"] += t
        proyeksi.append(model.predict(temp_df)[0])

    chart_df = pd.DataFrame({
        "Tahun": list(tahun),
        "Estimasi Gaji": proyeksi
    }).set_index("Tahun")

    st.line_chart(chart_df)

    growth_rate = ((proyeksi[-1] / proyeksi[0]) ** (1 / 10) - 1) * 100
    st.caption(f"📈 Rata-rata pertumbuhan gaji: **{growth_rate:.2f}% per tahun**")
