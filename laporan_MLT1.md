# Laporan Proyek Machine Learning - Klasifikasi Kinerja Siswa - Dimas Meidiansyah Putra

## Domain Proyek

Salah satu aspek dalam bidang pendidikan yang perlu diperhatikan adalah kemampuan atau kinerja yang dimiliki oleh siswa dengan melakukan pemahaman dan sekaligus memprediksi kinerja akademik dari siswa tersebut. dengan adanya kemajuan teknologi di zaman sekarang ini, tentu tidak luput dari adanya penerapan dari sebuah machine learning yang dibuat untuk menganalisis kinerja dari siswa sendiri yang dapat membantu permasalahan ini secara efektif. Tujuan diadakannya proyek ini untuk membangun model klasifikasi dalam memprediksi kinerja siswa berdasarkan waktu belajar dari siswa, kehadiran dari siswa dalam ke sekolah, ajaran dari orang tua, serta kegiatan yang diikuti oleh siswa seperti ekstrakurikuler dan lain sebagainya. dengan demikian harapan proyek machine learning klasifikasi ini dapat membantu dalam mengetahui kinerja dari siswa yang dilihat berdasarkan karakteristik dari siswa tersebut.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa masalah tersebut harus diselesaikan ?
Masalah kinerja pada siswa harus diselesaikan karena sangat fatal jika tidak dilakukan sebab pendidikan sangat rawan khususnya bagi siswa dengan faktor yang akan terjadi jika tidak diperhatikan adalah terjadinya putus sekolah dan retensi siswa. masalah tersebut tentu memicu dalam hal menciptakan sumber daya manusia. karena itu permasalahan ini harus diselesaikan.

- Jelaskan bagaimana masalah tersebut harus diselesaikan ?
masalah ini dapat diselesaikan dengan berbagai cara, salah satunya yaitu membuat atau membangun sistem klasifikasi kinerja siswa untuk memprediksi kinerja dari siswa dengan basis machine learning yang memanfaatkan model pada machine learning dan evaluasi hasil untuk memberikan hasil klasifikasi yang baik, dan pastinya dapat menyelesaikan case permasalahan ini.

- Format Referensi: [Preprocessing Data dan Klasifikasi untuk Prediksi Kinerja Akademik Siswa](https://jtiik.ub.ac.id/index.php/jtiik/article/view/8074)


## Business Understanding
### Problem Statements
Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara memprediksi sebuah tingkat performa akademik siswa berdasarkan fitur dari data yang tersedia?
- Apakah dari model machine learning yang dibangun, dapat membantu sekolah dalam mendeteksi kinerja siswa yang berisiko rendah secara akademik?

### Goals
Menjelaskan tujuan dari pernyataan masalah:
- Membangun model klasifikasi untuk memprediksi kategori nilai akhir siswa.
- Mengidentifikasi fitur-fitur yang berkontribusi terhadap kinerja akademik siswa   .

**Rubrik/Kriteria Tambahan (Opsional)**:
### Solution statements
- Membangun model dengan menggunakan algoritma Random Forest untuk membantu dalam memprediksi kinerja siswa.
- Melakukan peningkatan performa model dengan menerapkan hyperparameter tuning menggunakan GridSearchCV.
- Model akan dievaluasi menggunakan metrik F1-score untuk menangani klasifikasi multikelas yang tidak seimbang.


## Data Understanding
dataset yang dipilih adalah dataset kinerja dari siswa yang menampilkan berbagai atribut kondisi dari setiap siswa dan target dari variabel yaitu peringkat kelas 
**link dataset** : [Students Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset).
**jumlah data**: 2392
### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- Age: Usia siswa. `typedata:int64`
- Gender: Jenis kelamin siswa. `typedata:int64`
- Ethnicity: Kelompok etnis dari siswa. `typedata:int64`
- ParentalEducation: Tingkat pendidikan orang tua siswa. `typedata:int64`
- StudyTimeWeekly: Waktu belajar per minggu. `typedata:float64`
- Absences: Jumlah ketidakhadiran oleh siswa. `typedata:int64`
- Tutoring: kegiatan bimbel dari siswa ya/tidak mengikuti. `typedata:int64` 
- ParentalSupport: dukungan orang tua siswa dalam belajar. `typedata:int64`
- Extracurricular: kegiatan ekstrakurikuler oleh siswa ya/tidak mengikuti. `typedata:int64`
- Sports: kegiatan olah raga oleh siswa ya/tidak mengikuti. `typedata:int64`
- Music: kegiatan musik oleh siswa ya/tidak mengikuti. `typedata:int64`
- Volunteer: kegiatan sukarelawan oleh siswa ya/tidak mengikuti. `typedata:int64`
- GPA: Nilai rata-rata siswa.`typedata:float64`
- GradeClass: Target klasifikasi berupa label performa akhir.`typedata:float64`
 

**Rubrik/Kriteria Tambahan (Opsional)**:
**teknik visualisasi data atau exploratory data analysis**
- identifikasi atribut dari dataset
![imgidentifikasidata](https://drive.google.com/file/d/1lvxer7WwFTqU_lpUlonoI6VmxRx35Z09/view?usp=drive_link)
- melakukan korelasi pada variabel numerik semua atribut dataset
![imgkorelasidata](https://drive.google.com/file/d/1IlXDAYn89vzjlHytP79sZJo22TvPzQKv/view?usp=drive_link)
- barplot distribusi untuk melihat persebaran jumlah tiap gradeClass
![imgdistribusitarget](https://drive.google.com/file/d/1llwceayZoOmBU6S3niE641B5UCh8L09Z/view?usp=drive_link)
- menganalisis distribusi data melalui boxplot dan scatter plot dengan membandingkan atribut target class yaitu GradeClass dan GPA, GradeClass dan Absences, GPA dan Absences yang dilihat berdasarkan hasil korelasi yang dilihat pada matriks
- ![GPA&GradeClass](https://drive.google.com/file/d/1ifi-I1dYDNook6w4SgNYbjp3jKQ7_ND9/view?usp=drive_link)
- ![absences&Gradeclass](https://drive.google.com/file/d/1HcRzfSboc0fV43ljifMZSo7Z79Sf6S57/view?usp=drive_link)
- ![GPA&Absences](https://drive.google.com/file/d/1Bsmhel9Rl_UKi_FpWN3XOn_MMH1a_Wan/view?usp=drive_link)

## Data Preparation
**tahapan**:
- cek missing value
Melakukan cek nilai kosong terhadap data dengan menggunakan teknik `print(data.isnull().sum())` bertujuan menghindari data kosong yang menyebabkan mengurangi performa pada model
- cek duplikat data
Melakukan cek nilai duplikat terhadap data dengan menggunakan teknik `print(data.duplicated().sum())` dengan tujuan agar tidak ada data yang sama sehingga menghambat dalam proses belajar pada data
- normalisasi
melakukan normalisasi pada data dengan menggunakan MinMaxScaler dengan teknik `scaler = MinMaxScaler()` dengan tujuan menormlisasi data numerik agar model lebih mudah memahami
- split dataset dengan teknik `train_test_split()` yang bertujuan membagi data ke dalam data training ataupun data testing 

melakukan tahapan ini bertujuan agar modelling tetap belajar dengan baik pada saat training dan di normalisasi agar data lebih baik saat dipelajari, dan tidak bias terhadap skala nilai fitur tertentu.


## Modeling
model yang digunakan untuk machine learning klasifikasi ini yaitu `random forrest`. dengan parameter `random_state sebanyak 42`. melakukan hyperparameter tunning pada model menggunakan `GridSearchCV`, model randomforrest ini, untuk mengeksplorasi kombinasi terbaik parameter yang digunakan antaranya: 
- n_estimators[50,100,200]: Semakin banyak pohon, semakin stabil prediksi model. Namun, semakin banyak juga waktu komputasi, 
- max_depth[none,10,20,30]: Kedalaman pohon memengaruhi kompleksitas dan risiko overfitting. Batasan kedalaman membantu mengontrol hal tersebut, 
- min_samples_split[2,5,10]: Mencegah pohon membelah terlalu cepat dan menghasilkan cabang kecil yang tidak bermakna (overfitting),
- min_samples_leaf[1,2,4]:  Daun dengan sampel terlalu sedikit bisa menyebabkan model terlalu sensitif terhadap data minor.
- Cross-validation (cv=5) untuk menghindari overfitting
- Scoring: f1_weighted — sesuai dengan kebutuhan klasifikasi multikelas yang tidak seimbang.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- kelebihan: akurasi skor model menunjukkan hasil tinggi dan algoritma cukup sederhana
- Kekurangan: berpotensi overfitting karena model terlalu hapal terhadap data yang dipelajari
- pada model setelah melakukan hyperparamer tunning, model akurasi skor tetap, dan berpotensi mengurangi overfitting yaitu model mengurangi proses belajar terhadap data dan tidak terlalu hapal dalam belajar data. 


## Evaluation
Matriks yang digunakan dalam evaluasi model:
- Accuracy untuk mengetahui seberapa banyak prediksi yang tepat dari total data: `sebanyak 91%`
- Precision digunakan untuk mengevaluasi ketepatan model dalam memprediksi kelas: `sebanyak 91%`
- Recall digunakan untuk melihat seberapa baik model dapat menangkap kelas target: `sebanyak 91%`
- F1 Score digunakan untuk rata-rata harmonis dari precision dan recall, dihitung berdasarkan proporsi setiap kelas: `sebanyak 90%`
dikarena target berupa multikelas dan distribusi kelas tidak seimbang, perhitungan F1-score (weighted) digunakan sebagai metrik utama.
dengan Hasil evaluasi:
- Accuracy skor: `91%` - menunjukkan bahwa sebagian besar prediksi model sesuai dengan label asli.
- F1 Score (weighted): `90%` - metrik utama yang mencerminkan bahwa model cukup seimbang dalam mengenali setiap kelas.
Model juga divisualisasikan menggunakan confusion matrix dan learning curve, yang menunjukkan bahwa model tidak mengalami overfitting setelah tuning.

- Metrik F1-score sangat relevan untuk proyek ini karena tujuannya adalah mendeteksi siswa dengan risiko performa rendah, dan kesalahan klasifikasi bisa berarti intervensi yang salah atau terlambat.
- Dengan F1-score yang tinggi, berarti model dapat memberikan dukungan keputusan yang andal kepada sekolah.
- Model ini tidak hanya akurat, tapi juga sensitif terhadap ketidakseimbangan data kelas — penting dalam konteks pendidikan di mana kelas minoritas (misal siswa berisiko rendah) tidak boleh terabaikan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

model randomforrest menghasilkan skor tinggi di angka 91% meskipun terdapat adanya overfitting, namun dengan adanya hyperparameter tunning dapat menstabilkan performa model sehingga mengurangi overfitting dan klasifikasi dilakukan terhadap beberapa kelas (GradeClass) dengan distribusi yang tidak merata, maka digunakan F1-score (weighted) agar metrik akhir mempertimbangkan kontribusi dari setiap kelas sesuai proporsinya. yang diharapkan model ini dapat membantuk dalam memprediksi dan mendeteksi dari kinerja siswa di akademik sekolah.


