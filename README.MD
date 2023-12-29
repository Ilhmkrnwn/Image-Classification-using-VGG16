## Aplikasi Klasifikasi Gambar "Rock, Paper, dan Scissor"

### Overview Aplikasi
Aplikasi ini merupakan sistem klasifikasi gambar sederhana yang dapat mengenali apakah gambar tersebut merupakan "Rock" (batu), "Paper" (kertas), atau "Scissor" (gunting). Aplikasi ini memanfaatkan model deep learning ResNet50 untuk melakukan klasifikasi dengan tingkat akurasi yang tinggi.

![Example](/static/img1.png)
![Example2](/static/img2.png)

### Penggunaan Aplikasi

**Langkah-langkah:**
- Upload gambar pada input yang disediakan.
- Tekan tombol "Predict" untuk memulai proses klasifikasi.
- Aplikasi akan memberikan hasil prediksi, termasuk label klasifikasi (Rock, Paper, atau Scissor), akurasi, dan waktu runtime.

### Model yang Digunakan
Aplikasi ini menggunakan model ResNet50, yang merupakan salah satu arsitektur deep learning yang telah terbukti efektif dalam tugas klasifikasi gambar. ResNet50 memiliki kemampuan untuk mengenali pola dan fitur kompleks dalam gambar, membuatnya cocok untuk tugas klasifikasi yang melibatkan gambar kompleks seperti "Rock, Paper, dan Scissor."

### Cara Menjalankan Aplikasi (Development)

1. **Persiapkan Environment:**
   - Instal dependencies dengan menjalankan `pip install -r requirements.txt`.
   
2. **Konfigurasi:**
   - Pastikan model ResNet50 telah disiapkan dan dapat diakses oleh aplikasi.

3. **Menjalankan Aplikasi:**
   - Jalankan aplikasi dengan perintah `python app.py`.
   - Buka browser dan akses `http://localhost:5000`.

4. **Input Gambar:**
   - Upload gambar pada input yang disediakan.

5. **Klasifikasi:**
   - Tekan tombol "Predict" untuk memulai proses klasifikasi.