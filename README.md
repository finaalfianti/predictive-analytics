# Laporan Proyek Machine Learning – Fina Alfianti
**1. Domain Proyek**

Stunting pada anak merupakan masalah kesehatan yang serius di Indonesia dan berbagai negara berkembang. Stunting menyebabkan gangguan pertumbuhan fisik dan kognitif yang dapat berdampak jangka panjang pada kualitas hidup anak. Berdasarkan data Kementerian Kesehatan RI (2023), prevalensi stunting pada anak balita masih mencapai sekitar 24,4% [1]. Oleh karena itu, prediksi dini status gizi anak menggunakan data umur, jenis kelamin, dan tinggi badan dapat membantu intervensi lebih cepat dan tepat.

Menurut World Health Organization (WHO), tinggi badan yang tidak sesuai dengan usia merupakan indikator utama stunting yang memerlukan penanganan segera [2]. Dengan menggunakan metode machine learning, prediksi status gizi dapat dilakukan secara otomatis dan akurat sehingga dapat mempercepat proses deteksi dan penanganan.

**2. Business Understanding**

**Problem Statements**

- Bagaimana cara memprediksi status gizi anak (severely stunted, stunted, normal, tinggi) berdasarkan data umur, jenis kelamin, dan tinggi badan?
Permasalahan utama adalah mengembangkan metode yang mampu memanfaatkan data antropometri dasar seperti umur, jenis kelamin, dan tinggi badan untuk mengklasifikasikan status gizi anak ke dalam kategori yang sudah ditentukan. Data ini harus diproses sedemikian rupa agar model dapat mengenali pola-pola yang membedakan setiap kategori status gizi.
- Model machine learning apa yang paling efektif untuk masalah klasifikasi status gizi tersebut?
Karena status gizi merupakan variabel kategori dengan beberapa kelas, diperlukan algoritma klasifikasi yang dapat menangani data dengan baik dan memberikan akurasi tinggi. Pertanyaan ini menuntut evaluasi dan perbandingan model-model machine learning yang berbeda untuk menentukan algoritma yang paling cocok dan optimal untuk dataset ini.
- Bagaimana meningkatkan akurasi prediksi agar hasilnya dapat diandalkan untuk intervensi medis?
Dalam konteks medis, prediksi yang salah dapat berdampak serius. Oleh karena itu, model harus tidak hanya akurat tetapi juga konsisten dan dapat diandalkan. Hal ini mencakup upaya meningkatkan performa model melalui teknik pemodelan yang tepat, pemilihan fitur, dan evaluasi yang komprehensif agar hasilnya layak dijadikan dasar intervensi kesehatan anak.

**Goals**

- Mengembangkan model klasifikasi status gizi anak dengan menggunakan data antropometri.
Tujuan utama adalah membangun sebuah model klasifikasi berbasis machine learning yang dapat memprediksi status gizi anak (severely stunted, stunted, normal, tinggi) secara akurat dengan input data yang mudah diperoleh, seperti umur, jenis kelamin, dan tinggi badan.
- Membandingkan dua algoritma machine learning yaitu Random Forest dan XGBoost untuk menemukan model terbaik.
Dengan membandingkan performa dua algoritma populer ini, diharapkan dapat diketahui algoritma mana yang lebih efektif dan efisien dalam memproses data serta memberikan hasil klasifikasi terbaik pada data status gizi anak.
- Mengevaluasi keandalan model berdasarkan metrik akurasi, precision, recall, dan F1-score untuk memastikan hasil prediksi dapat digunakan sebagai dasar intervensi medis.
Selain hanya mengukur akurasi, evaluasi dilakukan secara menyeluruh dengan metrik precision, recall, dan F1-score agar dapat menilai kualitas model secara detail. Ini penting agar model tidak hanya tepat secara keseluruhan, tapi juga dapat meminimalkan kesalahan yang mungkin berisiko jika digunakan dalam pengambilan keputusan medis.
“Hasil model ini diharapkan dapat membantu tenaga medis dalam melakukan deteksi dini status gizi anak sehingga intervensi yang tepat bisa dilakukan lebih cepat dan efektif.”

**Solution Statements**

-	Menggunakan model machine learning yang bisa memproses data umur, jenis kelamin, dan tinggi badan untuk klasifikasi status gizi.
Data yang bersifat numerik (umur, tinggi badan) dan kategorikal (jenis kelamin) diproses dan dimasukkan ke dalam algoritma klasifikasi untuk melatih model dalam mengenali pola dan relasi antara fitur-fitur tersebut dengan status gizi anak.
-	Menerapkan model Random Forest dan XGBoost untuk melakukan klasifikasi status gizi anak berdasarkan data umur, jenis kelamin, dan tinggi badan.
Random Forest digunakan sebagai baseline karena kemampuannya menangani data dengan fitur campuran dan mengurangi risiko overfitting, sementara XGBoost dipilih sebagai model pembanding yang dikenal memiliki performa tinggi melalui boosting dan optimasi pohon keputusan.
-	Menggunakan metrik evaluasi untuk menilai model itu tepat karena metrik seperti akurasi, precision, recall, dan F1-score untuk mengukur performa klasifikasi dan keandalan prediksi.
Model dievaluasi menggunakan metrik-metrik yang relevan untuk klasifikasi multi-kelas agar dapat memastikan prediksi yang dihasilkan bukan hanya akurat secara umum, tetapi juga sensitif dan spesifik dalam mengenali setiap kelas status gizi, sehingga dapat diandalkan untuk kebutuhan intervensi medis.

**3. Data Understanding**
   
Dataset yang digunakan dalam proyek ini berasal dari Kaggle dengan judul "Stunting Balita Detection - 121k rows" yang dapat diakses melalui tautan berikut: https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows

Dataset ini berisi lebih dari 121.000 data balita dengan berbagai fitur yang relevan untuk mendeteksi status stunting, antara lain:

- Umur (bulan): Umur balita dalam bulan (numerik).
-	Jenis kelamin: Jenis kelamin balita, biasanya dikodekan sebagai Laki-laki atau Perempuan (kategorikal).
-	Tinggi badan (cm): Tinggi badan balita dalam centimeter (numerik).
-	Status gizi/stunting: Label target berupa kategori status stunting seperti severely stunted, stunted, normal, tall.
  
Melalui exploratory data analysis (EDA), dilakukan pengecekan distribusi umur, jenis kelamin, dan distribusi tinggi badan. Visualisasi histogram dan boxplot digunakan untuk mengetahui distribusi data serta mendeteksi nilai outlier. Data juga dicek untuk missing values dan inkonsistensi agar dapat dilakukan pembersihan dan persiapan sebelum pemodelan.

**4. Data Preparation**
   
Proses data preparation meliputi:
-	Mengisi nilai kosong (missing values) pada fitur umur dan tinggi badan menggunakan nilai rata-rata (mean).
-	Mengubah variabel kategorikal jenis kelamin menjadi bentuk numerik dengan one-hot encoding menjadi dua fitur baru: JK_Laki dan JK_Perempuan.
-	Melakukan normalisasi pada fitur numerik agar rentang data seimbang dan tidak mempengaruhi performa model.
-	Membagi dataset menjadi data training dan testing dengan proporsi 80:20.
Tahapan ini penting untuk memastikan data dalam format yang tepat agar algoritma dapat memprosesnya secara optimal serta mengurangi bias akibat data tidak lengkap atau tidak konsisten.

**5. Modeling**

Pada tahapan pemodelan ini, digunakan dua algoritma machine learning untuk menyelesaikan masalah klasifikasi status gizi anak berdasarkan data umur, jenis kelamin, dan tinggi badan, yaitu Random Forest dan XGBoost.

1. Random Forest Classifier
   
- Parameter utama:
  - n_estimators=100 (jumlah pohon keputusan yang dibangun)
  - max_depth=10 (kedalaman maksimum tiap pohon)
- Proses:
Random Forest membangun banyak pohon keputusan secara acak (bagging) dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting.
- Kelebihan:
  - Mampu menangani data dengan fitur yang saling berkorelasi, seperti data antropometri.
  -	Lebih stabil dan kurang rentan terhadap overfitting dibandingkan pohon tunggal.
  -	Relatif mudah untuk disetel dan tidak terlalu sensitif terhadap parameter.
-	Kekurangan:
    - Model bisa menjadi kompleks dan memakan waktu komputasi lebih lama jika jumlah pohon terlalu besar.
    -	Kurang optimal jika terdapat fitur dengan hubungan yang sangat non-linear atau kompleks.
2. XGBoost Classifier
-	Parameter utama:
    - n_estimators=100
    -	max_depth=5
    -	learning_rate=0.1
-	Proses:
XGBoost menggunakan teknik boosting yang secara iteratif mengoptimalkan kesalahan dari model sebelumnya, sehingga biasanya meningkatkan akurasi pada berbagai dataset.
-	Kelebihan:
    -	Sangat efektif dalam menangani data kompleks dengan hubungan non-linear antar fitur.
    -	Dapat melakukan regularisasi untuk menghindari overfitting.
    -	Memiliki performa tinggi pada banyak kasus klasifikasi dan regresi.
-	Kekurangan:
    -	Membutuhkan tuning hyperparameter yang cukup rumit untuk mendapatkan performa terbaik.
    -	Sensitif terhadap parameter yang tidak optimal sehingga bisa menurunkan performa.
    -	Proses pelatihan lebih lambat dibanding Random Forest pada dataset besar.
      
Pada proyek ini, kedua model dilatih dan dievaluasi dengan parameter yang telah ditentukan. Hasil evaluasi menunjukkan bahwa Random Forest memberikan performa yang lebih baik dan stabil dibandingkan XGBoost, yang meskipun sudah menggunakan parameter yang disetel manual, performanya sedikit lebih rendah terutama pada kelas dengan distribusi data yang menantang. Hal ini mengindikasikan bahwa Random Forest lebih cocok untuk dataset status gizi anak ini.

**6. Evaluation**

Pada bagian evaluasi ini, digunakan beberapa metrik untuk mengukur performa model klasifikasi status gizi anak, yaitu akurasi (accuracy), precision, recall, dan F1-score. Metrik-metrik ini dipilih karena sesuai dengan konteks klasifikasi multi-kelas dan kebutuhan untuk memastikan hasil prediksi dapat diandalkan sebagai dasar intervensi medis.

**Penjelasan Metrik Evaluasi**

-	Akurasi (Accuracy) adalah rasio prediksi yang benar terhadap seluruh data yang diuji.

 	Akurasi mengukur seberapa banyak model benar secara keseluruhan, namun tidak cukup jika kelas data tidak seimbang.
-	Precision menunjukkan seberapa tepat prediksi model untuk setiap kelas, yaitu proporsi data yang benar-benar positif dari semua data yang diprediksi positif oleh model.
-	Recall (Sensitivity) mengukur kemampuan model dalam menemukan seluruh data positif pada kelas tertentu, yaitu proporsi data yang berhasil ditemukan dari seluruh data positif sebenarnya.
-	F1-score adalah harmonisasi rata-rata precision dan recall yang memberikan gambaran keseimbangan antara keduanya.
 
Metrik precision, recall, dan F1-score dihitung untuk tiap kelas status gizi (severely stunted, stunted, normal, tinggi) agar evaluasi model lebih detail dan tidak bias karena imbalance data.

**Hasil Evaluasi**

Dua model yang diuji adalah Random Forest dan XGBoost. Berikut ringkasan hasil evaluasi pada data train dan test:

| Model          | Data  | Accuracy | Precision (avg) | Recall (avg) | F1-Score (avg) |
|----------------|--------|----------|------------------|--------------|----------------|
| Random Forest  | Train  | 0.9954   | ~1.00           | ~0.99        | ~0.99          |
| Random Forest  | Test   | 0.9936   | ~0.99           | ~0.99        | ~0.99          |
| XGBoost        | Train  | 0.9649   | ~0.96           | ~0.93        | ~0.95          |
| XGBoost        | Test   | 0.9623   | ~0.96           | ~0.93        | ~0.94          |

-	Random Forest memberikan performa yang sangat baik dengan akurasi mendekati 99.5% di data train dan 99.3% di data test, serta precision, recall, dan F1-score yang konsisten tinggi di semua kelas. Ini menunjukkan model sangat andal dalam mengklasifikasikan status gizi anak dan dapat diandalkan untuk intervensi medis.
-	XGBoost memiliki akurasi dan metrik lain yang sedikit lebih rendah dibandingkan Random Forest. Terutama pada kelas 'normal', recall menurun (sekitar 77% pada train), mengindikasikan beberapa data kelas ini kurang terdeteksi dengan baik. Meskipun demikian, model ini tetap memberikan hasil yang cukup baik secara keseluruhan.
-	Perbedaan performa ini terlihat juga dari confusion matrix masing-masing model, dimana Random Forest menunjukkan kesalahan klasifikasi yang lebih sedikit.

**Referensi**

[1] Kementerian Kesehatan Republik Indonesia, “Profil Kesehatan Indonesia 2023,” Jakarta, 2024. [Online]. Available: https://kemkes.go.id/app_asset/file_content_download/172231123666a86244b83fd8.51637104.pdf

[2] World Health Organization, “Shaping Health Insights: WHO’s Support for the Indonesia Health Survey 2023,” 2024. [Online]. Available: https://www.who.int/indonesia/news/detail/11-01-2024-shaping-health-insights--who-s-support-for-the-indonesia-health-survey-2023


