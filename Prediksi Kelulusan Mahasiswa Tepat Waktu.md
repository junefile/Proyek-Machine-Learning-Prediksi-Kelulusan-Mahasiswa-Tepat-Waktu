# Laporan Proyek Machine Learning: Prediksi Kelulusan Mahasiswa Tepat Waktu


## 1. Domain Proyek

### Latar Belakang
Sektor pendidikan dihadapkan pada tantangan signifikan dalam memastikan mahasiswa dapat menyelesaikan studinya tepat waktu dan berhasil. Tingginya angka mahasiswa yang **tidak lulus tepat waktu (drop out)** atau **terlambat lulus** bukan hanya merugikan mahasiswa secara individual (penundaan karir, biaya tambahan), tetapi juga berdampak pada reputasi dan efisiensi institusi pendidikan. Fenomena ini dapat membebani sumber daya universitas dan mengurangi outcome pendidikan secara keseluruhan. Mengidentifikasi mahasiswa yang berisiko sejak dini memungkinkan institusi untuk memberikan intervensi yang ditargetkan, seperti bimbingan akademik, dukungan finansial, atau konseling, sehingga meningkatkan tingkat kelulusan dan keberhasilan mahasiswa.

### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan
Masalah kelulusan tepat waktu perlu diselesaikan karena ini adalah indikator kunci keberhasilan mahasiswa dan efektivitas sistem pendidikan. Secara tradisional, identifikasi mahasiswa berisiko seringkali bersifat reaktif, dilakukan setelah masalah muncul, atau didasarkan pada intuisi. Dengan memanfaatkan machine learning, kita dapat beralih ke pendekatan yang lebih proaktif dan berbasis data. Model machine learning dapat menganalisis berbagai pola dalam data historis mahasiswa (akademik, demografi, perilaku) untuk memprediksi risiko kelulusan tidak tepat waktu jauh sebelum masalah tersebut menjadi kritis. Solusi ini akan memungkinkan intervensi dini yang lebih efektif, membantu mahasiswa tetap di jalur, dan pada akhirnya meningkatkan angka kelulusan tepat waktu secara keseluruhan.

### Riset Terkait / Referensi
Anwar, M. T., Heriyanto, L., & Fanini, F. (2021). Model Prediksi Dropout Mahasiswa Menggunakan Teknik Data Mining. Jurnal Informatika Upgris, 7(1). [https://doi.org/10.26877/jiu.v7i1.8023]

Nurhayati, S., Kusrini, K., & Luthfi, E. T. (2015). PREDIKSI MAHASISWA DROP OUT MENGGUNAKAN METODE SUPPORT VECTOR MACHINE. SISFOTENIKA, 5(1), Article 1. [https://sisfotenika.stmikpontianak.ac.id/index.php/ST/article/viewFile/25/29]

## 2. Business Understanding

### Pernyataan Masalah (Problem Statements)
- Bagaimana kita dapat secara akurat mengidentifikasi mahasiswa yang berisiko tidak lulus tepat waktu?
- Faktor-faktor apa saja (akademik, demografi, perilaku) yang paling berpengaruh terhadap kelulusan tepat waktu seorang mahasiswa?

### Tujuan (Goals)
- Mengembangkan model machine learning yang mampu memprediksi apakah seorang mahasiswa akan lulus tepat waktu atau tidak.
- Model harus mencapai akurasi klasifikasi minimal 80% dalam memprediksi status kelulusan.
- Mengidentifikasi fitur-fitur kunci yang paling berkorelasi dengan kelulusan tepat waktu.

### Pernyataan Solusi (Solution Statements)
Untuk mencapai tujuan di atas, saya akan mengajukan dua pendekatan solusi yang akan dievaluasi dan dibandingkan:
1. Penggunaan Algoritma Baseline dengan Peningkatan Hyperparameter Tuning:
Kami akan memulai dengan membangun model Logistic Regression sebagai baseline kami karena kesederhanaan dan interpretasinya yang baik. Untuk meningkatkan performanya, kami akan melakukan penyetelan hyperparameter menggunakan GridSearchCV untuk menemukan kombinasi parameter optimal yang memaksimalkan metrik evaluasi F1-Score atau ROC AUC. Solusi ini akan diukur berdasarkan peningkatan F1-Score, Presisi, Recall, dan Akurasi yang dicapai dibandingkan dengan kinerja model baseline tanpa tuning.

2. Perbandingan Dua Algoritma Klasifikasi Berbeda:
Selain Logistic Regression, saya akan membangun model Random Forest Classifier. Random Forest dikenal karena kemampuannya menangani kompleksitas data, memberikan akurasi tinggi, dan kurang rentan terhadap overfitting dibandingkan pohon keputusan tunggal. Kedua model ini (Logistic Regression dan Random Forest yang di-tune ) akan dilatih dan dievaluasi menggunakan metrik klasifikasi yang sama. Model yang menunjukkan kinerja terbaik (berdasarkan F1-Score dan ROC AUC) pada test set akan dipilih sebagai solusi akhir.


## 3. Data Understanding

### Informasi dan Sumber Data
Proyek ini menggunakan dataset mahasiswa yang berisi informasi akademik, demografi, dan perilaku belajar. Dataset ini memiliki 2000 sampel data dan 17 fitur (kolom) sebelum preprocessing. Data ini bersih dan tidak ditemukan adanya missing values yang signifikan.

### Tautan Sumber Data:
https://www.kaggle.com/datasets/markmedhat/student-scores?resource=download

### Variabel/Fitur Data
Berikut adalah deskripsi lengkap dari variabel-variabel yang terdapat dalam dataset ini:

| Fitur / Variabel | Tipe Data | Deskripsi | Relevansi untuk Prediksi |
| :--------------- | :-------- | :-------- | :----------------------- |
| `id`| Integer| Pengidentifikasi unik untuk setiap mahasiswa. | Tidak Relevan |
| `first_name` | Objek (String) | Nama depan mahasiswa. | Tidak Relevan |
| `last_name` | Objek (String) | Nama belakang mahasiswa. | Tidak Relevan |
| `email` | Objek (String) | Alamat email mahasiswa. | Tidak Relevan |
| `gender` | Objek (String) | Jenis kelamin mahasiswa (e.g., 'Male', 'Female'). | Relevan |
| `part_time_job` | Boolean | Indikator apakah mahasiswa memiliki pekerjaan paruh waktu (True/False). | Relevan |
| `absence_days` | Integer| Jumlah hari mahasiswa tidak hadir di kelas. | Relevan |
| `extracurricular_activities` | Boolean | Indikator partisipasi dalam kegiatan ekstrakurikuler (True/False). | Relevan |
| `weekly_self_study_hours` | Integer| Rata-rata jam belajar mandiri per minggu. | Relevan |
| `career_aspiration`  | Objek (String) | Aspirasi karir mahasiswa (e.g., 'Doctor', 'Engineer'). | Berpotensi Relevan |
| `math_score` | Integer| Nilai mahasiswa di mata pelajaran Matematika. | Relevan |
| `history_score` | Integer| Nilai mahasiswa di mata pelajaran Sejarah. | Relevan |
| `physics_score` | Integer| Nilai mahasiswa di mata pelajaran Fisika. | Relevan |
| `chemistry_score` | Integer| Nilai mahasiswa di mata pelajaran Kimia. | Relevan |
| `biology_score` | Integer| Nilai mahasiswa di mata pelajaran Biologi. | Relevan |
| `english_score` | Integer| Nilai mahasiswa di mata pelajaran Bahasa Inggris. | Relevan |
| `geography_score` | Integer| Nilai mahasiswa di mata pelajaran Geografi. | Relevan |

## 4. Data Preparation
Tahap persiapan data sangat penting untuk mengubah data mentah menjadi format yang cocok untuk machine learning, memastikan kualitas dan relevansi fitur.

### Teknik Data Preparation yang Dilakukan
1. Penghapusan Kolom Tidak Relevan:

Kolom `id`, `first_name` , `last_name` , dan `email` dihapus.
    - Alasan: Kolom-kolom ini adalah pengidentifikasi unik atau data tekstual yang tidak memberikan informasi prediktif tentang kelulusan mahasiswa. Menghapusnya akan mengurangi dimensi data dan noise tanpa kehilangan informasi penting.

2. Pembuatan Target Variabel (`On_Time_Graduation`):

Karena dataset asli tidak memiliki kolom ini, saya membuat target variabel `On_Time_Graduation` berdasarkan rata-rata skor dari semua mata pelajaran. Mahasiswa dianggap 'Lulus Tepat Waktu' (nilai 1) jika `Average_Score` mereka lebih besar atau sama dengan 70, dan 'Tidak Lulus Tepat Waktu' (nilai 0) jika di bawah 70.
    - Alasan: Ini adalah cara paling logis untuk mendefinisikan target kelulusan tepat waktu dari data yang tersedia. Rata-rata skor adalah indikator kinerja akademik yang kuat.

3. Konversi Tipe Data Boolean: 

Kolom `part_time_job` dan `extracurricular_activities` yang bertipe boolean (True/False) dikonversi menjadi integer (1/0).
    - Alasan: Algoritma machine learning umumnya membutuhkan input numerik. Konversi ini mengubah nilai boolean menjadi representasi numerik yang dapat diproses oleh model.

4. Encoding Variabel Kategorikal:
- `gender`: Dikonversi menggunakan Label Encoding. 'Male' menjadi 0 dan 'Female' menjadi 1 (atau sebaliknya).
    - Alasan: Fitur ini hanya memiliki dua kategori, sehingga Label Encoding sudah cukup dan efisien.
- `career_aspiration`: Dikonversi menggunakan One-Hot Encoding. Ini menciptakan kolom biner baru untuk setiap kategori unik dalam fitur ini (misalnya, Career_Aspiration_Doctor, Career_Aspiration_Engineer).
    - Alasan: Fitur `career_aspiration` bersifat nominal (tidak ada urutan inheren). One-Hot Encoding mencegah model secara keliru menganggap adanya hubungan urutan atau hierarki antar kategori, yang dapat terjadi jika menggunakan Label Encoding pada data nominal dengan lebih dari dua kategori. Ini memastikan representasi yang adil untuk setiap aspirasi karir.

5. Definisi Preprocessor dan Normalisasi Fitur Numerik:

Sebuah ColumnTransformer (preprocessor) didefinisikan untuk menerapkan transformasi pada kolom-kolom yang berbeda. Di dalamnya, semua fitur numerik (`absence_days`, `weekly_self_study_hours`, `math_score`, `history_score`, `physics_score`, `chemistry_score`, `biology_score `, `english_score`, `geography_score`, dan `Average_Score`) diskalakan menggunakan StandardScaler. Fitur boolean yang sudah dikonversi ke integer dan fitur Gender yang sudah di-encode akan dilewatkan tanpa transformasi lebih lanjut. Penerapan preprocessor ini akan dilakukan di tahap pemodelan sebagai bagian dari pipeline.
    - Alasan: Fitur-fitur ini memiliki skala dan rentang nilai yang sangat berbeda. Standardisasi memastikan bahwa semua fitur berkontribusi secara proporsional terhadap proses pelatihan model, mencegah fitur dengan nilai yang lebih besar mendominasi perhitungan bobot model, yang penting untuk algoritma seperti Logistic Regression dan SVM. Penggunaan ColumnTransformer memungkinkan pengelolaan berbagai jenis transformasi secara terstruktur dan efisien.

6. Pembagian Data Latih dan Uji: 

Data dibagi menjadi training set (80%) dan test set (20%) menggunakan `train_test_split` dengan `stratify=y`.
    - Alasan: Pembagian ini memastikan bahwa model dilatih pada data yang belum pernah dilihat sebelumnya (test set) untuk evaluasi yang objektif. Penggunaan stratify=y sangat penting untuk menjaga proporsi kelas target (Lulus Tepat Waktu vs. Tidak Lulus Tepat Waktu) tetap sama di training dan test set, mencegah bias dalam evaluasi jika ada ketidakseimbangan kelas.

## 5. Modeling

Pada tahap ini, saya membangun model machine learning untuk memprediksi kelulusan mahasiswa tepat waktu, sesuai dengan Pernyataan Solusi yang diajukan.

### Tahapan dan Parameter Pemodelan
Saya akan menggunakan Random Forest Classifier sebagai model utama yang dioptimalkan, dan Logistic Regression sebagai model baseline untuk perbandingan kinerja.

**1. Random Forest Classifier (Model Utama dengan Penyetelan Hyperparameter)**
- Cara Kerja Algoritma: Random Forest adalah algoritma ensemble learning berbasis pohon keputusan. Ide dasarnya adalah membangun sejumlah besar pohon keputusan (n_estimators) selama proses pelatihan. Setiap pohon dilatih secara independen pada subsampel data pelatihan yang berbeda (metode bagging atau bootstrap aggregating) dan subset fitur yang dipilih secara acak. Untuk membuat prediksi, Random Forest menggabungkan prediksi dari semua pohon individu: untuk masalah klasifikasi, ini berarti mengambil suara mayoritas dari prediksi kelas oleh setiap pohon (atau rata-rata probabilitas). Proses bagging membantu mengurangi variance dan mencegah overfitting.
- Tahapan: Model ini diimplementasikan dalam sebuah pipeline yang menggabungkan preprocessing (preprocessor yang dibuat di tahap Data Preparation) dengan algoritma RandomForestClassifier. Untuk optimasi dan pengembangan solusi utama, kami melakukan Grid Search Cross-Validation pada pipeline ini.
- Parameter yang Digunakan pada Tuning:
  - classifier__n_estimators: [100, 200, 300] (jumlah pohon keputusan dalam forest)
  - classifier__max_depth: [None, 10, 20] (kedalaman maksimum pohon; None berarti pohon akan berkembang penuh)
  - classifier__min_samples_split: [2, 5] (jumlah minimum sampel yang diperlukan untuk membagi internal node)
  - classifier__random_state: 42 (untuk reproduktibilitas hasil)
- Proses Improvement: GridSearchCV secara sistematis mencoba setiap kombinasi parameter yang ditentukan (param_grid_rf). Untuk setiap kombinasi, model dilatih dan dievaluasi menggunakan cross-validation pada training data. Model Random Forest dengan kombinasi parameter terbaik berdasarkan metrik f1-score (karena pentingnya keseimbangan antara presisi dan recall dalam klasifikasi ini) akan dipilih sebagai best_rf_model.

**2. Logistic Regression (Model Baseline untuk Perbandingan)**
- Cara Kerja Algoritma: Logistic Regression adalah algoritma klasifikasi linear yang digunakan untuk memprediksi probabilitas suatu kejadian. Meskipun namanya mengandung "Regression", ini adalah model klasifikasi. Ia bekerja dengan menggunakan fungsi logistik (atau sigmoid) untuk memetakan setiap kombinasi input fitur ke dalam probabilitas antara 0 dan 1. Probabilitas ini kemudian dikonversi menjadi prediksi kelas biner berdasarkan ambang batas tertentu (misalnya, 0.5). Model belajar bobot untuk setiap fitur yang memaksimalkan kemungkinan data yang diamati.
- Tahapan: Model ini diimplementasikan dalam sebuah pipeline yang menggabungkan preprocessing (preprocessor) dengan algoritma LogisticRegression. Model ini berfungsi sebagai model baseline dan dilatih dengan parameter default (tanpa hyperparameter tuning ekstensif).
- Parameter yang Digunakan:
  - random_state: 42 (untuk reproduktibilitas hasil)

### Pemilihan Model Terbaik
Setelah melatih dan mengevaluasi kedua model (Random Forest yang di-tune dan Logistic Regression sebagai baseline) pada test set, kami akan membandingkan metrik evaluasinya, terutama F1-Score dan ROC AUC.

Berdasarkan hasil evaluasi (detail akan disajikan di bagian Evaluasi), Random Forest Classifier menunjukkan kinerja yang lebih unggul dibandingkan Logistic Regression, terutama dalam hal F1-Score sebesar 0.9961 dan ROC AUC sebesar 1.0000. Kinerja Random Forest yang superior pada metrik-metrik kunci ini mengindikasikan bahwa kemampuan model ensemble untuk menangani kompleksitas dan interaksi non-linear pada dataset ini memberikan hasil yang lebih optimal dibandingkan model linear dasar. Oleh karena itu, Random Forest Classifier dipilih sebagai model terbaik dan solusi akhir untuk proyek ini.

## 6. Evaluation
Evaluasi model adalah tahap krusial untuk mengukur seberapa baik model dapat membuat prediksi pada data yang belum pernah dilihat sebelumnya, dan apakah ia memenuhi tujuan proyek.

**Metrik Evaluasi yang Digunakan**

Untuk masalah klasifikasi biner seperti prediksi kelulusan mahasiswa tepat waktu, metrik evaluasi yang digunakan adalah:
1. Akurasi (Accuracy)
- Proporsi total prediksi yang benar (baik lulus tepat waktu maupun tidak) dari semua prediksi yang dibuat.
- Memberikan gambaran umum kinerja model, tetapi bisa menyesatkan jika ada ketidakseimbangan kelas yang signifikan (misalnya, jika mayoritas mahasiswa lulus tepat waktu).

2. Presisi (Precision)
-  Dari semua mahasiswa yang diprediksi akan "Lulus Tepat Waktu" (kelas positif), berapa persen yang benar-benar lulus tepat waktu.
- Penting jika biaya false positive (memprediksi lulus tepat waktu padahal tidak) sangat tinggi. Dalam konteks ini, ini berarti institusi tidak salah mengalokasikan sumber daya ke mahasiswa yang sebenarnya tidak memerlukan intervensi.

3. Recall (Sensitivity / True Positive Rate)
- Dari semua mahasiswa yang sebenarnya "Lulus Tepat Waktu" (actual positive), berapa persen yang berhasil diidentifikasi oleh model.
- Penting jika biaya false negative (memprediksi tidak lulus tepat waktu padahal sebenarnya lulus) tinggi. Dalam kasus ini, kita juga sangat peduli dengan kelas "Tidak Lulus Tepat Waktu" (kelas negatif), sehingga recall untuk kelas negatif (disebut Specificity) juga relevan untuk memastikan kita tidak melewatkan mahasiswa yang berisiko. Namun, secara konvensional, recall biasanya mengacu pada kelas positif.

4. F1-Score
- Rata-rata harmonik dari Presisi dan Recall. Metrik ini menyeimbangkan kedua metrik tersebut.
- Sangat relevan ketika ada kebutuhan untuk menyeimbangkan Presisi dan Recall, terutama jika ada ketidakseimbangan kelas. Ini adalah metrik utama saya untuk memilih model terbaik.

5. Matriks Kebingungan (Confusion Matrix)
- Tabel yang menunjukkan jumlah True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN).
    - TP (True Positives): Model memprediksi lulus tepat waktu, dan mahasiswa benar-benar lulus tepat waktu.
    - TN (True Negatives): Model memprediksi tidak lulus tepat waktu, dan mahasiswa benar-benar tidak lulus tepat waktu.
    - FP (False Positives): Model memprediksi lulus tepat waktu, tapi mahasiswa tidak lulus tepat waktu (Type I error).
    - FN (False Negatives): Model memprediksi tidak lulus tepat waktu, tapi mahasiswa benar-benar lulus tepat waktu (Type II error).

6. ROC AUC (Receiver Operating Characteristic - Area Under Curve)
- Mengukur kemampuan model untuk membedakan antara kelas positif dan negatif di berbagai ambang batas klasifikasi. Nilai AUC berkisar dari 0 hingga 1; semakin dekat ke 1, semakin baik model dalam membedakan kelas.
- Berguna ketika ambang batas klasifikasi perlu disesuaikan. Model dengan AUC tinggi dianggap lebih baik secara keseluruhan.

### Hasil Proyek Berdasarkan Metrik Evaluasi
**Model Terbaik (Berdasarkan Kinerja): Logistic Regression**

**Model Solusi Utama Proyek (Disesuaikan Tuning): Random Forest**

| Metrik | Nilai Logistic Regression | Nilai Random Forest | 
| :----- | :------------------------ | :------------------ |
| Akurasi | 0.9875 | 0.9925 | 
| Presisi | 0.9871 | 0.9922 | 
| Recall | 1.0000 | 1.0000 | 
| F1-Score | 0.9935 | 0.9961 | 
| ROC AUC | 0.9980 | 1.0000 | 

**Confusion Matrix (Untuk Model Random Forest, sebagai Fokus Solusi Proyek):**
```
 [[ 14   3]
 [  0 383]]
```
- TN: 14 (Mahasiswa diprediksi Tidak Lulus Tepat Waktu dan memang tidak)
- FP: 3 (Mahasiswa diprediksi Lulus Tepat Waktu tapi mahasiswa tidak lulus tepat waktu)
- FN: 0 (Mahasiswa diprediksi Tidak Lulus Tepat Waktu tapi mahasiswa benar-benar lulus tepat waktu)
- TP: 383 (Mahasiswa diprediksi Lulus Tepat Waktu dan memang lulus)

### Analisis Hasil
Model Random Forest Classifier, sebagai fokus utama proyek ini setelah fine-tuning, mencapai Akurasi 0.9925, F1-Score 0.9961, dan ROC AUC 1.0000. Matriks kebingungannya menunjukkan 3 False Positives dan 0 False Negatives. Performa ini sangat baik dan memenuhi target akurasi proyek (>80%), serta menunjukkan kemampuan prediksi yang mendekati sempurna untuk ROC AUC.

Di sisi lain, model Logistic Regression yang digunakan sebagai baseline tanpa hyperparameter tuning khusus, secara numerik menunjukkan Akurasi 0.9875, F1-Score 0.9935, dan ROC AUC 0.9980. Matriks kebingungan Logistic Regression menunjukkan 5 False Positives dan 0 False Negatives.

Perbandingan ini menunjukkan bahwa Random Forest Classifier yang di-tune berhasil mengungguli Logistic Regression (baseline) dalam hampir semua metrik utama, terutama pada F1-Score dan ROC AUC yang lebih tinggi. Ini mengonfirmasi bahwa penggunaan model ensemble dengan optimasi hyperparameter mampu menangkap pola data yang lebih kompleks dan menghasilkan prediksi yang sedikit lebih akurat pada dataset ini.

Hasil ini menunjukkan bahwa model Random Forest yang dioptimalkan memiliki potensi besar untuk menjadi alat bantu yang sangat efektif dalam program dukungan mahasiswa.

## 7. Kesimpulan

Proyek ini berhasil mengembangkan model machine learning untuk memprediksi kelulusan mahasiswa tepat waktu berdasarkan data akademik, demografi, dan perilaku. Dengan menggunakan Random Forest Classifier yang melalui proses data preparation yang cermat dan hyperparameter tuning, model kami mampu mencapai akurasi (0.9925), F1-Score (0.9961), dan ROC AUC (1.0000) yang sangat baik. Fitur-fitur seperti nilai mata pelajaran, jam belajar mandiri, dan hari absensi terbukti menjadi prediktor yang sangat signifikan.

Model ini dapat menjadi alat proaktif bagi institusi pendidikan untuk:
- Mengidentifikasi mahasiswa berisiko: Memberikan daftar mahasiswa yang kemungkinan besar tidak akan lulus tepat waktu, sehingga intervensi (bimbingan, dukungan) dapat diberikan sejak dini.
- Memahami faktor pendorong: Memberikan wawasan tentang faktor-faktor apa saja yang paling memengaruhi kelulusan, membantu merumuskan kebijakan atau program yang lebih efektif.
