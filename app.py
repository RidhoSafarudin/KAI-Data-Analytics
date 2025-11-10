

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="KAI Data Analytics",
    page_icon="🚊",
    layout="wide"
)

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("KAI_dataset_clean.csv")

    # Pastikan month valid
    df["month_num"] = df["Bulan"].astype(int)
    df.loc[~df["month_num"].between(1, 12), "month_num"] = 1

    # Buat kolom Date
    df["Date"] = pd.to_datetime(
        df["Tahun"].astype(str) + "-" + df["month_num"].astype(str).str.zfill(2),
        format="%Y-%m"
    )

    # Drop kolom yang sudah tidak dipakai
    df = df.drop(columns=["Bulan", "Tahun", "month_num"])

    # Kolom moda (fixed manual)
    moda_cols = [
        "Jabodetabek",
        "NonJabodetabek",
        "Jawa",
        "NonJawa",
        "Kereta Bandara",
        "MRT",
        "LRT",
        "KeretaCepatWhoosh"
    ]

    # Hitung total penumpang
    df["Total"] = df[moda_cols].sum(axis=1)

    return df, moda_cols


df, moda_cols = load_data()

bulan_nama = ["Januari","Februari","Maret","April","Mei","Juni",
              "Juli","Agustus","September","Oktober","November","Desember"]
df["Bulan_Nama"] = df["Date"].dt.month.apply(lambda x: bulan_nama[x-1])


# SIDEBAR
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Logo_PT_Kereta_Api_Indonesia_%28Persero%29_2020.svg", width=150)
st.sidebar.title("KAI Data Analytics")
menu = st.sidebar.radio("Navigasi", ["📊 Dataset", "📈 Visualisasi", "🔮 Forecast"])
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by : Streamlit & Plotly")

# MENU: Dataset
if menu == "📊 Dataset":
    st.title("Data Jumlah Pengguna Layanan KAI 2022-2024")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "💾 Download Data",
        data=df.to_csv(index=False),
        file_name="KAI_clean_export.csv",
        mime="text/csv"
    )

# MENU: Visualisasi
elif menu == "📈 Visualisasi":
    st.title("Visualisasi Tren Penumpang")

    # Filter
    tahun_terpilih = st.multiselect(
        "Pilih Tahun",
        sorted(df["Date"].dt.year.unique()),
        default=sorted(df["Date"].dt.year.unique())
    )

    pilihan_moda = st.multiselect(
        "Pilih Moda Transportasi",
        ["Total"] + moda_cols,
        default=["Total"]
    )

    df_filtered = df[df["Date"].dt.year.isin(tahun_terpilih)]

    # Statistik ringkas
    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Periode Data", f"{df_filtered['Date'].min().strftime('%b %Y')} - {df_filtered['Date'].max().strftime('%b %Y')}")
    col2.metric("📈 Total Penumpang", f"{df_filtered['Total'].sum():,}")
    col3.metric("👥 Rata-rata Penumpang", f"{df_filtered['Total'].mean():,.0f}")

    # Grafik Tren
    fig = px.line(
        df_filtered,
        x="Date", y=pilihan_moda,
        markers=True,
        title=f"Tren Penumpang: {', '.join(pilihan_moda)}",
        template="plotly_white"
    )
    fig.update_layout(
        xaxis_title="Waktu",
        yaxis_title="Jumlah Penumpang",
        hovermode="x unified"
    )
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)


    # INSIGHT OTOMATIS

    st.subheader("💡 Insight")

    df_filtered["Tahun"] = df_filtered["Date"].dt.year
    df_filtered["month_num"] = df_filtered["Date"].dt.month

    bulan_nama = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
                "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    df_filtered["Bulan_Nama"] = df_filtered["month_num"].apply(lambda x: bulan_nama[x-1])


    for moda in pilihan_moda:
        df_filtered_sorted = df_filtered.sort_values("Date").copy()
        df_filtered_sorted["Perubahan"] = df_filtered_sorted[moda].diff()

        if df_filtered_sorted["Perubahan"].notna().any():
            max_up = df_filtered_sorted.loc[df_filtered_sorted["Perubahan"].idxmax()]
            max_down = df_filtered_sorted.loc[df_filtered_sorted["Perubahan"].idxmin()]

            st.markdown(
                f"""
                **Moda {moda}:**
                - 📈 Kenaikan terbesar terjadi pada **{max_up['Bulan_Nama']} {int(max_up['Tahun'])}** sebesar **{max_up['Perubahan']:,.0f} penumpang**.
                - 📉 Penurunan terbesar terjadi pada **{max_down['Bulan_Nama']} {int(max_down['Tahun'])}** sebesar **{abs(max_down['Perubahan']):,.0f} penumpang**.
                """
            )
        else:
            st.write(f"Data {moda} tidak memiliki perubahan bulanan yang dapat dianalisis.")

# MENU: Forecast
elif menu == "🔮 Forecast":
    st.title("🔮 Prediksi Penumpang (Linear Regression)")

    # Pilihan moda yang valid
    pilihan_moda = st.selectbox("Pilih Moda untuk Prediksi", ["Total"] + moda_cols)
    y = df[pilihan_moda]

    # Model regresi linear berbasis index waktu
    df["time_index"] = np.arange(len(df))
    X = df[["time_index"]]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Nilai R²
    r2 = r2_score(y, y_pred)

    # Prediksi 12 bulan ke depan
    future_idx = np.arange(len(df), len(df) + 12)
    future_pred = model.predict(future_idx.reshape(-1, 1))
    
    hasil_pred = pd.DataFrame({
        "Date": pd.date_range(start=df["Date"].max() + pd.offsets.MonthBegin(1), 
                              periods=12, freq="MS"),
        "Prediksi": future_pred.astype(int)
    })

    fig2 = px.line(
        df, x="Date", y=pilihan_moda,
        markers=True,
        title=f"Prediksi {pilihan_moda} 12 Bulan ke Depan",
        template="plotly_white"
    )
    fig2.add_scatter(
        x=hasil_pred["Date"], y=hasil_pred["Prediksi"],
        mode="lines+markers", name="Prediksi",
        line=dict(dash="dash", width=3)
    )
    fig2.update_layout(
        xaxis_title="Waktu",
        yaxis_title="Jumlah Penumpang",
        hovermode="x unified"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.metric("🎯 R² Score", f"{r2:.3f}")
        st.write("**Tabel Prediksi 12 Bulan ke Depan**")
        st.dataframe(hasil_pred, use_container_width=True)
        st.download_button(
            "💾 Download Hasil Prediksi",
            data=hasil_pred.to_csv(index=False),
            file_name=f"Prediksi_{pilihan_moda}.csv",
            mime="text/csv"
        )

    # Bersihkan kolom tambahan supaya tidak terbawa ke menu lain
    df.drop(columns=["time_index"], inplace=True)

# FOOTER
st.markdown("---")
st.markdown(
    "<center>🚆 KAI Analytics Dashboard © 2025 | Dibuat oleh Mahasiswa Visualisasi Data</center>",
    unsafe_allow_html=True
)
