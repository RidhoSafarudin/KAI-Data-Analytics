

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import time
import plotly.graph_objects as go


# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="KAI Data Analytics",
    page_icon="🚊",
    layout="wide"
)

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_KAI_fix.csv")

    # Pastikan kolom Date bertipe datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Identifikasi kolom moda transportasi (semua kecuali Total & Date)
    moda_cols = df.columns.drop(["Total", "Date"]).tolist()

    # Konversi satuan ribu orang → orang asli
    df[moda_cols] = df[moda_cols] * 1000
    df["Total"] = df["Total"] * 1000

    # Urutkan berdasarkan waktu
    df = df.sort_values("Date")

    return df, moda_cols



df, moda_cols = load_data()


# SIDEBAR
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Logo_PT_Kereta_Api_Indonesia_%28Persero%29_2020.svg", width=150)
st.sidebar.title("KAI Data Analytics")
menu = st.sidebar.radio("Navigasi", ["Dataset", "Visualisasi", "PolaData", "Forecast"])
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by : Streamlit & Plotly")

# Helper format ribuan ala Indonesia
def id_format(x):
    return f"{int(x):,}".replace(",", ".")

# MENU: Dataset

def animate_metric(container, label, value, duration=1.5, steps=50):
    for i in range(steps + 1):
        current = int(value * i / steps)
        container.metric(label, id_format(current))
        time.sleep(duration / steps)


if menu == "Dataset":
    st.title("Dataset Penumpang KAI 2022–2024")

    st.markdown("Lihat informasi dasar sebelum kita masuk ke analisis dan prediksi.")

    col1, col2, col3 = st.columns(3)

    with col1:
        cont1 = st.empty()
        animate_metric(cont1, "📅 Jumlah Bulan Data", len(df))

    with col2:
        cont2 = st.empty()
        animate_metric(cont2, "👥 Total Penumpang", df["Total"].sum())

    with col3:
        cont3 = st.empty()
        animate_metric(cont3, "📈 Rata-rata per Bulan", df["Total"].mean())


    st.markdown("---")

    st.subheader("Tren Singkat Jumlah Penumpang (Total)")
    fig_overview = px.line(
        df, x="Date", y="Total",
        markers=True,
        template="plotly_white",
        height=300
    )
    fig_overview.update_layout(
        xaxis_title="Waktu",
        yaxis_title="Total Penumpang",
        hovermode="x unified"
    )
    st.plotly_chart(fig_overview, use_container_width=True)

    st.markdown("---")

    st.subheader("Tabel Dataset")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "💾 Download Dataset",
        data=df.to_csv(index=False),
        file_name="dataset_KAI_fix.csv",
        mime="text/csv"
    )


## MENU: Visualisasi
elif menu == "Visualisasi":
    st.title("Visualisasi Tren Penumpang")

    st.markdown("### Filter Data")

    # 🔹 Filter dalam satu baris (3 kolom)
    colA, colB, colC = st.columns(3)

    tahun_terpilih = colA.multiselect(
        "Tahun",
        sorted(df["Date"].dt.year.unique()),
        default=sorted(df["Date"].dt.year.unique())
    )

    pilihan_moda = colB.multiselect(
        "Moda",
        ["Total"] + moda_cols,
        default=["Total"]
    )

    jenis_chart = colC.selectbox(
        "Grafik",
        ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot"]
    )

    df_filtered = df[df["Date"].dt.year.isin(tahun_terpilih)].copy()

    df_filtered["Total_Selected"] = df_filtered[pilihan_moda].sum(axis=1)

    periode_data = f"{df_filtered['Date'].min().strftime('%b %Y')} - {df_filtered['Date'].max().strftime('%b %Y')}"
    total_penumpang = df_filtered["Total_Selected"].sum()
    avg_penumpang = df_filtered["Total_Selected"].mean()

    # Statistik ringkas
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("🗓️ Periode Data", periode_data)
    col2.metric("👥 Total Penumpang", id_format(total_penumpang))
    col3.metric("📉 Rata-rata/Bulan", id_format(avg_penumpang))

    st.markdown("---")
    st.subheader("📊 Grafik Tren Penumpang")

    # ✅ Render grafik dinamis
    if jenis_chart == "Line Chart":
        fig = px.line(df_filtered, x="Date", y=pilihan_moda, markers=True)
        fig.update_traces(line=dict(width=3))

    elif jenis_chart == "Bar Chart":
        fig = px.bar(df_filtered, x="Date", y=pilihan_moda, barmode="group")

    elif jenis_chart == "Area Chart":
        fig = px.area(df_filtered, x="Date", y=pilihan_moda)

    elif jenis_chart == "Scatter Plot":
        fig = px.scatter(df_filtered, x="Date", y=pilihan_moda)
        fig.update_traces(marker=dict(size=10))

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Waktu",
        yaxis_title="Jumlah Penumpang"
    )

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
elif menu == "Forecast":
    st.title("Prediksi Penumpang KAI")

    pilihan_moda = st.selectbox(
        "Pilih Moda untuk Prediksi",
        ["Total"] + moda_cols
    )
    
    df_model = df.copy()
    df_model["time_index"] = np.arange(len(df_model))  # index waktu
    X = df_model[["time_index"]]
    y = df_model[pilihan_moda]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)

    future_months = st.slider("Berapa Bulan ke Depan?", 3, 24, 6)
    future_idx = np.arange(len(df_model), len(df_model) + future_months)
    future_pred = model.predict(future_idx.reshape(-1, 1))

    hasil_pred = pd.DataFrame({
        "Date": pd.date_range(
            start=df_model["Date"].max() + pd.offsets.MonthBegin(1),
            periods=future_months,
            freq="MS"
        ),
        "Prediksi": future_pred.astype(int)
    })

    fig2 = go.Figure()

    # Data historis
    fig2.add_trace(go.Scatter(
        x=df_model["Date"], y=df_model[pilihan_moda],
        mode="lines+markers",
        name="Data Aktual",
        line=dict(width=3)
    ))

    # Garis prediksi
    fig2.add_trace(go.Scatter(
        x=hasil_pred["Date"], y=hasil_pred["Prediksi"],
        mode="lines+markers",
        name="Prediksi",
        line=dict(width=3, dash="dot")
    ))

    fig2.update_layout(
        template="plotly_white",
        xaxis_title="Waktu",
        yaxis_title=f"Penumpang Moda: {pilihan_moda}",
        hovermode="x unified",
        title=f"Prediksi {pilihan_moda} {future_months} Bulan ke Depan"
    )

    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.metric("🎯 Akurasi Model (R²)", f"{r2:.3f}")
        st.markdown("📆 **Tabel Hasil Prediksi**")
        hasil_pred_show = hasil_pred.copy()
        hasil_pred_show["Prediksi"] = hasil_pred_show["Prediksi"].apply(id_format)
        st.dataframe(hasil_pred_show, use_container_width=True)

        st.download_button(
            "💾 Download Hasil Prediksi",
            data=hasil_pred.to_csv(index=False),
            file_name=f"Prediksi_{pilihan_moda}_{future_months}bulan.csv",
            mime="text/csv"
        )

elif menu == "PolaData":
    st.title("Heatmap Pola Lonjakan Penumpang")

    pilihan_moda_heat = st.selectbox(
        "Pilih Moda",
        ["Total"] + moda_cols
    )

    df_heat = df.copy()
    df_heat["Bulan"] = df_heat["Date"].dt.month_name()
    df_heat["Tahun"] = df_heat["Date"].dt.year

    # Urutan bulan agar rapih
    urutan_bulan = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    # Pivot data → Bulan vs Tahun
    pivot_data = df_heat.pivot_table(
        index="Bulan",
        columns="Tahun",
        values=pilihan_moda_heat,
        aggfunc="sum"
    ).reindex(urutan_bulan)

    fig_heat = px.imshow(
        pivot_data,
        labels=dict(x="Tahun", y="Bulan", color=f"Jumlah Penumpang ({pilihan_moda_heat})"),
        aspect="auto",
        text_auto=True
    )

    fig_heat.update_layout(
        title=f"Heatmap Lonjakan Penumpang - Moda {pilihan_moda_heat}",
        template="plotly_white"
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    max_month = pivot_data.sum(axis=1).idxmax()
    st.markdown(f"""
    ### Insight
    🚆 Bulan dengan permintaan tertinggi untuk moda **{pilihan_moda_heat}** adalah: **{max_month}**
    
    🚆 Ini artinya bulan tersebut layak diprioritaskan untuk **penambahan armada & kapasitas operasional**.
    """)


# FOOTER
st.markdown("---")
st.markdown(
    "<center>🚆 KAI Analytics Dashboard © 2025 | Dibuat oleh Mahasiswa Visualisasi Data</center>",
    unsafe_allow_html=True
)
