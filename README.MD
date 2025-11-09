# 🚄 KAI Analytics Dashboard

Dashboard interaktif berbasis Streamlit dan Plotly yang menampilkan analisis tren penumpang Kereta Api Indonesia (KAI) dari data historis, serta memprediksi jumlah penumpang ke depan menggunakan Linear Regression.

---

## Tujuan Proyek

Menyediakan alat analisis data penumpang KAI yang interaktif, informatif, dan mudah digunakan untuk membantu pengguna memahami tren perjalanan penumpang dari waktu ke waktu serta melakukan prediksi sederhana terhadap data tersebut.

---

## Alasan di Balik Desain

### 1. **Pemilihan Pengkodean Visual**

- Kami menggunakan grafik garis (line chart) karena sangat efektif dalam memperlihatkan tren temporal (perubahan terhadap waktu).
- Setiap moda transportasi direpresentasikan dengan garis berwarna berbeda untuk membandingkan pola antar moda.
- Penambahan marker (titik) di setiap data membantu pengguna melihat nilai spesifik setiap bulan dengan jelas.
- Skala waktu (Date) digunakan di sumbu X, dan jumlah penumpang di sumbu Y untuk menjaga konsistensi persepsi waktu dan volume data.

### 2. **Desain Interaksi**

- Menggunakan sidebar untuk navigasi agar antarmuka utama tetap bersih.
- Pengguna dapat memilih moda transportasi dan tahun yang ingin dianalisis secara interaktif.
- Disediakan tombol Download CSV agar pengguna dapat mengekspor data dan hasil prediksi dengan mudah.
- Menambahkan insight otomatis di bawah grafik untuk membantu pengguna mendapatkan kesimpulan cepat tanpa analisis manual.

## 🔧 Fitur Utama

| Fitur                   | Deskripsi                                                                       |
| ----------------------- | ------------------------------------------------------------------------------- |
| 📊 **Dataset**          | Menampilkan dataset hasil pembersihan dan menyediakan tombol unduh CSV.         |
| 📈 **Visualisasi**      | Menampilkan grafik tren penumpang per bulan/tahun dengan filter moda dan tahun. |
| 💡 **Insight Otomatis** | Mengidentifikasi bulan dengan kenaikan & penurunan penumpang terbesar.          |
| 🔮 **Forecasting**      | Melakukan prediksi penumpang 12 bulan ke depan menggunakan _Linear Regression_. |
| 💾 **Ekspor Data**      | Hasil prediksi dan dataset dapat diunduh langsung.                              |

---

## 🔄 Gambaran Umum Proses Pengembangan

### 1. **Pembagian Tugas Tim**

| Anggota | Tanggung Jawab                                            |
| ------- | --------------------------------------------------------- |
| B       | Data cleaning dan integrasi CSV ke Streamlit              |
| A & B   | Desain layout dashboard (sidebar, navigasi, interaksi)    |
| A       | Implementasi model Linear Regression & visualisasi Plotly |
| C       | Pembuatan insight otomatis dan dokumentasi proyek         |

Ket:
A = Ridho Safarudin
B = Faaris Muwaffaq
C = Ivandri Yudha Santoso

---

### 2. **Estimasi Waktu Pengerjaan**

| Tahapan                                    | Estimasi Waktu |
| ------------------------------------------ | -------------- |
| Persiapan data & pembersihan               | 2 jam          |
| Pengembangan layout & interaksi Streamlit  | 4 jam          |
| Implementasi visualisasi & uji coba Plotly | 5 jam          |
| Pengembangan model prediksi                | 1 jam          |
| Penambahan insight otomatis & debugging    | 3 jam          |
| Dokumentasi & penyusunan README            | 2 jam          |
| **Total Estimasi Waktu**                   | **≈17 jam**    |

---

### 3. **Bagian yang Paling Memakan Waktu**

- **Desain interaksi dan visualisasi Plotly** adalah bagian yang paling memakan waktu.
  - Menyesuaikan format tanggal, label, serta warna agar grafik mudah dibaca membutuhkan banyak iterasi.
  - Penentuan insight otomatis (kenaikan & penurunan terbesar) juga memerlukan pengujian ulang agar hasil analisis sesuai dengan konteks data.
- Bagian forecasting relatif cepat karena hanya menggunakan model regresi linear sederhana.

---

## Cara Menjalankan Aplikasi

1. Pastikan semua dependensi telah diinstal:
   pip install streamlit plotly scikit-learn pandas numpy

2. Jalankan aplikasi:
   streamlit run app.py

3. Akses melalui browser di:
   http://localhost:8501
