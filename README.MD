# 🚄 KAI Analytics Dashboard

Dashboard interaktif berbasis **Streamlit** dan **Plotly** untuk menganalisis tren penumpang **Kereta Api Indonesia (KAI)** dari data historis 2022–2024.  
Aplikasi ini juga menampilkan **pola musiman (heatmap)** dan **prediksi penumpang ke depan** menggunakan _Linear Regression_ sederhana.

---

## 🎯 Tujuan Proyek

Memberikan alat analisis yang interaktif dan informatif untuk membantu pengguna memahami:

- Perkembangan jumlah penumpang dari waktu ke waktu.
- Moda transportasi dengan lonjakan atau penurunan signifikan.
- Pola bulanan dan musiman melalui heatmap.
- Estimasi jumlah penumpang beberapa bulan ke depan.

---

## 💡 Alasan Desain

### 1. **Pemilihan Visualisasi**

- **Line Chart / Area Chart** → menunjukkan tren temporal.
- **Bar Chart** → membandingkan jumlah antar periode.
- **Heatmap** → menyoroti pola musiman atau bulan dengan permintaan tinggi.
- **Scatter Plot** → memperlihatkan variasi data per bulan.
- Warna dan marker digunakan untuk membedakan tiap moda transportasi.

### 2. **Desain Interaksi**

- Navigasi melalui **sidebar** agar tampilan utama tetap bersih.
- Pengguna dapat memfilter **tahun** dan **moda** yang ingin dianalisis.
- Tombol **Download CSV** untuk mengunduh dataset dan hasil prediksi.
- Insight otomatis dan animasi metrik menambah konteks cepat pada data.

---

## 🔧 Fitur Utama

| Fitur                      | Deskripsi                                                                |
| -------------------------- | ------------------------------------------------------------------------ |
| 📊 **Dataset**             | Menampilkan dataset lengkap dan metrik animasi jumlah total, rata-rata.  |
| 📈 **Visualisasi Tren**    | Grafik interaktif (Line, Bar, Area, Scatter) dengan filter moda & tahun. |
| 💡 **Insight Otomatis**    | Menampilkan bulan dengan kenaikan/penurunan terbesar tiap moda.          |
| 🔥 **Pola Data (Heatmap)** | Menunjukkan pola musiman berdasarkan bulan dan tahun.                    |
| 🔮 **Forecasting**         | Prediksi penumpang 3–24 bulan ke depan dengan Linear Regression.         |
| 💾 **Ekspor Data**         | Dataset dan hasil prediksi bisa diunduh langsung.                        |

---

## 🔄 Gambaran Umum Proses Pengembangan

### 1. **Pembagian Tugas Tim**

| Anggota   | Tanggung Jawab                                            |
| --------- | --------------------------------------------------------- |
| **B**     | Data cleaning dan integrasi CSV ke Streamlit              |
| **A & B** | Desain layout dashboard (sidebar, navigasi, interaksi)    |
| **A**     | Implementasi model Linear Regression & visualisasi Plotly |
| **C**     | Insight otomatis, Heatmap, serta dokumentasi proyek       |

**Ket:**  
A = Ridho Safarudin  
B = Faaris Muwaffaq  
C = Ivandri Yudha Santoso

---

### 2. **Estimasi Waktu Pengerjaan**

| Tahapan                                    | Estimasi Waktu |
| ------------------------------------------ | -------------- |
| Persiapan & pembersihan data               | 2 jam          |
| Desain layout & interaksi Streamlit        | 4 jam          |
| Implementasi visualisasi & uji coba Plotly | 5 jam          |
| Pengembangan model prediksi                | 1 jam          |
| Penambahan insight otomatis & debugging    | 3 jam          |
| Pembuatan Heatmap pola data                | 2 jam          |
| Dokumentasi & penyusunan README            | 2 jam          |
| **Total Estimasi Waktu**                   | **≈19 jam**    |

---

### 3. **Bagian yang Paling Memakan Waktu**

- Desain interaksi dan visualisasi (Plotly + Streamlit) paling kompleks.  
  Mengatur format tanggal, label, serta warna agar mudah dibaca memerlukan banyak uji coba.
- Pembuatan insight otomatis dan heatmap juga menuntut pengolahan data tambahan.
- Bagian forecasting lebih ringan karena menggunakan model regresi linear sederhana.

---

## ▶️ Cara Menjalankan Aplikasi

1. **Instal dependensi:**

   ```bash
   pip install streamlit plotly scikit-learn pandas numpy
   ```

2. **Jalankan aplikasi:**

   ```bash
   streamlit run app.py
   ```

3. **Akses di browser:**
   ```
   http://localhost:8501
   ```
