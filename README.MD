## Aplikasi Klasifikasi Gambar "Rock, Paper, dan Scissor"

### Overview Aplikasi
Aplikasi ini merupakan sistem klasifikasi gambar sederhana yang dapat mengenali apakah gambar tersebut merupakan "Rock" (batu), "Paper" (kertas), atau "Scissor" (gunting). Aplikasi ini memanfaatkan model deep learning VGG-16 untuk melakukan klasifikasi dengan tingkat akurasi yang tinggi.

![Example](/static/contoh1.png)
![Example2](/static/contoh2.png)

### Penggunaan Aplikasi

**Langkah-langkah:**
- Upload gambar pada input yang disediakan.
- Tekan tombol "Predict" untuk memulai proses klasifikasi.
- Aplikasi akan memberikan hasil prediksi, termasuk label klasifikasi (Rock, Paper, atau Scissor), akurasi, dan waktu runtime.

### Model yang Digunakan
Aplikasi ini menggunakan model VGG-16, sebuah arsitektur deep learning yang terbukti efektif dalam tugas klasifikasi gambar. VGG-16 memiliki kemampuan untuk mengenali pola dan fitur kompleks dalam gambar, menjadikannya cocok untuk tugas klasifikasi yang melibatkan gambar kompleks seperti "Rock, Paper, dan Scissor."

### Cara Menjalankan Aplikasi (Development)

1. **Persiapkan Environment:**
   - Instal dependencies dengan menjalankan `pip install -r requirements.txt`.
   
2. **Konfigurasi:**
   - Pastikan model VGG-16 telah disiapkan dan dapat diakses oleh aplikasi. You can download the model [here](https://drive.google.com/file/d/1pP3faZUi7jBMJ-5fQlYgGJXYKFfuxwLR/view?usp=sharing)

3. **Menjalankan Aplikasi:**
   - Jalankan aplikasi dengan perintah `python app.py`.
   - Buka browser dan akses `http://127.0.0.1:2000/`.

4. **Input Gambar:**
   - Upload gambar pada input yang disediakan.

5. **Klasifikasi:**
   - Tekan tombol "Predict" untuk memulai proses klasifikasi.
