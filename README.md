# Analisis Sentimen Bahasa Indonesia 🇮🇩

## Deskripsi

Aplikasi ini merupakan **Analisis Sentimen Bahasa Indonesia** berbasis model **deep learning dari Hugging Face (BERT dan RoBERTa)**.
Aplikasi ini mampu mengenali sentimen dari teks Bahasa Indonesia dan mengklasifikasikannya menjadi **positif**, **negatif**, atau **netral** secara otomatis.

Model ini menggabungkan dua model besar (ensemble) untuk meningkatkan akurasi, yaitu model IndoRoBERTa dan XLM-RoBERTa.
Aplikasi ini dibuat menggunakan **Streamlit** dan dapat berjalan di komputer lokal maupun di platform online seperti **Streamlit Cloud** atau **Hugging Face Spaces**.

---

## Fitur Utama

* Analisis otomatis sentimen Bahasa Indonesia.
* Menggunakan dua model terbaik dari Hugging Face.
* Dapat berjalan di CPU tanpa perlu GPU.
* Menampilkan grafik tingkat kepercayaan (confidence) dari hasil analisis.
* Antarmuka interaktif berbasis Streamlit.

---

## Persyaratan Sistem

Sebelum menjalankan aplikasi, pastikan telah menginstal:

* Python versi 3.9 atau lebih baru.
* Pip versi terbaru.
  Jalankan perintah berikut di terminal:
  `pip install --upgrade pip`

---

## Langkah Instalasi (Lokal)

1. **Unduh atau clone repositori ini.**
   Jalankan perintah:
   `git clone https://github.com/username/sentiment-app.git`
   `cd sentiment-app`

2. **(Opsional) Buat virtual environment.**

   * Untuk Windows:
     `python -m venv venv`
     `venv\Scripts\activate`
   * Untuk Linux/Mac:
     `python -m venv venv`
     `source venv/bin/activate`

3. **Instal semua dependensi.**
   Jalankan perintah:
   `pip install -r requirements.txt`

4. **Jalankan aplikasi Streamlit.**
   `streamlit run app.py`

5. **Buka browser.**
   Aplikasi akan berjalan otomatis di alamat:
   [http://localhost:8501](http://localhost:8501)

---

## Struktur Proyek

sentiment-app/
├── app.py → File utama aplikasi Streamlit
├── requirements.txt → Daftar dependensi Python
└── README.md → Dokumentasi lengkap

---

## Isi File requirements.txt

streamlit==1.38.0
transformers==4.42.4
torch==2.2.2+cpu
numpy==1.26.4
matplotlib==3.9.2
sentencepiece==0.2.0
protobuf==4.25.3

Catatan:
Jika aplikasi dijalankan di Streamlit Cloud, ubah baris `torch==2.2.2+cpu` menjadi `torch==2.2.2` agar sistem otomatis menyesuaikan dengan CPU environment.

---

## Cara Menggunakan Aplikasi

1. Jalankan perintah `streamlit run app.py`.
2. Masukkan teks Bahasa Indonesia di kolom input.
   Contoh:

   * “Saya sangat puas dengan layanan ini.”
   * “Produknya mengecewakan dan tidak sesuai harapan.”
   * “Biasa saja, tidak terlalu bagus tapi juga tidak jelek.”
3. Klik tombol “Analisis Sentimen”.
4. Aplikasi akan menampilkan hasil analisis berupa label sentimen (positif, negatif, atau netral) dan tingkat kepercayaan dari masing-masing model.

---

## Contoh Hasil Analisis

Kalimat: Saya sangat senang dan puas
Hasil: Positif
Confidence: 97.2%

Kalimat: Saya kecewa dengan produk ini
Hasil: Negatif
Confidence: 94.8%

Kalimat: Biasa aja sih
Hasil: Netral
Confidence: 88.1%

Kalimat: Tidak buruk kok
Hasil: Positif
Confidence: 90.5%

---

## Model yang Digunakan

1. w11wo/indonesian-roberta-base-sentiment-classifier
   → Model RoBERTa yang dilatih khusus untuk Bahasa Indonesia.

2. cardiffnlp/twitter-xlm-roberta-base-sentiment
   → Model multibahasa yang mendukung Bahasa Indonesia dan berbagai bahasa lain.

Kedua model ini digunakan bersama (ensemble) agar hasil lebih akurat dan stabil.

---

## Troubleshooting

Masalah: Error “torch.classes”
Penyebab: Versi PyTorch tidak sesuai.
Solusi: Gunakan versi `torch==2.2.2+cpu`.

Masalah: Model gagal dimuat
Penyebab: Nama model salah atau koneksi internet terputus.
Solusi: Pastikan nama model benar dan koneksi internet aktif.

Masalah: Semua hasil netral
Penyebab: Kalimat tidak mengandung kata emosional yang kuat.
Solusi: Booster logika otomatis sudah aktif, tetapi bisa ditambah daftar kata kunci emosional di dalam kode.

---

## Cara Deploy ke Streamlit Cloud

1. Upload file `app.py`, `requirements.txt`, dan `README.md` ke GitHub.
2. Buka situs [https://share.streamlit.io](https://share.streamlit.io).
3. Hubungkan akun GitHub ke Streamlit Cloud.
4. Pilih repositori dan branch utama.
5. Klik tombol “Deploy”.
6. Tunggu hingga proses build selesai (sekitar 3–5 menit).

Setelah selesai, aplikasi akan langsung bisa digunakan secara online.

---

## Lisensi

Aplikasi ini menggunakan lisensi **MIT License**, sehingga bebas digunakan, dimodifikasi, dan dikembangkan untuk tujuan pembelajaran atau penelitian.

---

## Kontributor

Eko Setiawan — Developer
ChatGPT (OpenAI) — Model dan optimisasi
