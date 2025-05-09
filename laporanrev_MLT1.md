# Laporan Proyek Machine Learning - Klasifikasi Kinerja Siswa - Dimas Meidiansyah Putra

## Domain Proyek

Salah satu aspek dalam bidang pendidikan yang perlu diperhatikan adalah kemampuan atau kinerja yang dimiliki oleh siswa dengan melakukan pemahaman dan sekaligus memprediksi kinerja akademik dari siswa tersebut. dengan adanya kemajuan teknologi di zaman sekarang ini, tentu tidak luput dari adanya penerapan dari sebuah machine learning yang dibuat untuk menganalisis kinerja dari siswa sendiri yang dapat membantu permasalahan ini secara efektif. Tujuan diadakannya proyek ini untuk membangun model klasifikasi dalam memprediksi kinerja siswa berdasarkan waktu belajar dari siswa, kehadiran dari siswa dalam ke sekolah, ajaran dari orang tua, serta kegiatan yang diikuti oleh siswa seperti ekstrakurikuler dan lain sebagainya. dengan demikian harapan proyek machine learning klasifikasi ini dapat membantu dalam mengetahui kinerja dari siswa yang dilihat berdasarkan karakteristik dari siswa tersebut.

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

### Solution statements
- Membangun model dengan menggunakan algoritma Random Forest untuk membantu dalam memprediksi kinerja siswa.
- Melakukan peningkatan performa model dengan menerapkan hyperparameter tuning menggunakan GridSearchCV.
- Model akan dievaluasi menggunakan metrik F1-score untuk menangani klasifikasi multikelas yang tidak seimbang.


## Data Understanding
dataset yang dipilih adalah dataset kinerja dari siswa yang menampilkan berbagai atribut kondisi dari setiap siswa dan target dari variabel yaitu peringkat kelas 
**Tautan Sumber Data** : [Students Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset).
**jumlah data**: Baris = 2392, Kolom = 15
**Kondisi Data**:
- Missing Value: Tidak ditemukan nilai kosong pada dataset (hasil dari kode `data.isnull().sum()` manghasilkan nol untuk semua fitur pada dataset).
- Data Duplikat: Tidak terdapat duplikat dalam dataset (hasil dari kode `data.duplicated().sum()` = 0).

### Variabel-variabel pada dataset **student performance classification** adalah sebagai berikut:
| Nama Fitur          | Deskripsi                                                                | Tipe Data |
| ------------------- | -------------------------------------------------------------------------| --------- |
| Student ID          | id dari siswa (1001 to 3392)                                             | Integer   |
| `Age`               | Usia siswa (15 -18 tahun)                                                | Integer   |
| `Gender`            | Jenis kelamin siswa (0 = Laki-laki, 1 = Perempuan)                       | Integer   |
| `Ethnicity`         | Kelompok etnis siswa (0: Caucasian,1: African American,2: Asian,3: Other)| Integer   |
| `ParentalEducation` | Tingkat pendidikan tertinggi orang tua siswa                             |           |
|                     | (0: None,1: High School,,2: Some College, 3: Bachelor's,4: Higher)       | Integer   |
| `StudyTimeWeekly`   | Rata-rata waktu belajar siswa per minggu (0-20)                          | float64   |
| `Absences`          | Jumlah ketidakhadiran siswa di sekolah(0-30)                             | Integer   |
| `Tutoring`          | Keikutsertaan dalam les/bimbingan belajar (0 = Tidak, 1 = Ya)            | Integer   |
| `ParentalSupport`   | Dukungan orang tua dalam kegiatan akademik siswa                         |           |
|                     | (0: None, 1: Low, 2: Moderate, 3: High, 4: Very High)                    | Integer   |
| `Extracurricular`   | Partisipasi dalam kegiatan ekstrakurikuler (0 = Tidak, 1 = Ya)           | Integer   |
| `Sports`            | Keikutsertaan dalam kegiatan olahraga (0 = Tidak, 1 = Ya)                | Integer   |
| `Music`             | Keikutsertaan dalam kegiatan musik (0 = Tidak, 1 = Ya)                   | Integer   |
| `Volunteer`         | Keikutsertaan dalam kegiatan sukarelawan (0 = Tidak, 1 = Ya)             | Integer   |
| `GPA`               | Nilai rata-rata akademik siswa (2.0-4.0)                                 | float64   |
| `GradeClass`        | Target label performa akhir siswa (0-4)                                  | float64   |

**teknik visualisasi data atau exploratory data analysis**
- identifikasi atribut dari dataset
![Identifikasi data](https://github.com/user-attachments/assets/cce7f07d-a05c-44dc-a345-fe6b22c945b8)
- melakukan korelasi pada variabel numerik semua atribut dataset
![korelasi_data](https://github.com/user-attachments/assets/b8c87202-bfe8-4d83-a35b-5bdc6c7b89a5)

- barplot distribusi untuk melihat persebaran jumlah tiap gradeClass
![dis](https://github.com/user-attachments/assets/b8a46cb1-1a9e-4c61-a06e-4f9f5ca8a043)

- menganalisis distribusi data melalui boxplot dan scatter plot dengan membandingkan atribut target class yaitu GradeClass dan GPA, GradeClass dan Absences, GPA dan Absences yang dilihat berdasarkan hasil korelasi yang dilihat pada matriks
- ![GPAvsGradeClass](https://github.com/user-attachments/assets/a14cc8f2-7ec9-4f84-9194-9fa1ecc666e9)

- ![GradeClassvsAbsences](https://github.com/user-attachments/assets/8b10aebb-9b93-473d-983d-bdee47d53431)

- ![GPAvsAbsences](https://github.com/user-attachments/assets/896ae56d-5424-4c74-a44d-8338676302d8)


## Data Preparation
**tahapan**:
- Handling Missing Value
Tidak ditemukan nilai kosong, sehingga tidak diperlukan imputasi atau penghapusan pada dataset.
- Handling Duplikat
Tidak ditemukan data duplikat, tidak ada tindakan tambahan yang dilakukan pada dataset.
- normalisasi/standardisasi
melakukan normalisasi pada semua fitur kecuali fitur target pada data dengan menggunakan MinMaxScaler dengan teknik `scaler = MinMaxScaler()` dengan tujuan menormlisasi data numerik agar seragam (0-1) dan model lebih mudah memahami
- split dataset dengan teknik `train_test_split()` yang bertujuan membagi data ke dalam data training ataupun data testing. pembagian data dengan rasio `training:testing = 80:20` dan `random state sebanyak 42`

melakukan tahapan ini bertujuan agar modelling tetap belajar dengan baik pada saat training dan di normalisasi agar data lebih baik saat dipelajari, dan tidak bias terhadap skala nilai fitur tertentu.

## Modeling
#### Random Forrest
Cara kerja untuk model random forrest ini dengan menerapkan ensemble berbasis pohon keputusan / decision tree yang bekerja dengan cara membuat banyak pohon keputusan secara acak dari subset data, dan mengambil hasil prediksi dari mayoritas hasil data sebagai output akhir.
Parameter yang digunakan pada pembangunan model random forrest ini, hanya menggunakan (randomstate = 42) yang digunakan sebagai seed untuk replikasi dari hasil.
pada model ini menggunakan hyperparameter tunning GridSearchCV dengan tujuan memperbaiki hasil model agar meminimalisir overfitting dan cara keja dengan melakukan kombinasi dari beberapa parameter yang digunakan, parameter yang digunakan pada tuning diantaranya:
- n_estimators[50,100,200]: Semakin banyak pohon, semakin stabil prediksi model. Namun, semakin banyak juga waktu komputasi, 
- max_depth[none,10,20,30]: Kedalaman pohon memengaruhi kompleksitas dan risiko overfitting. Batasan kedalaman membantu mengontrol hal tersebut, 
- min_samples_split[2,5,10]: Mencegah pohon membelah terlalu cepat dan menghasilkan cabang kecil yang tidak bermakna (overfitting),
- min_samples_leaf[1,2,4]:  Daun dengan sampel terlalu sedikit bisa menyebabkan model terlalu sensitif terhadap data minor.
- Cross-validation (cv=5) untuk menghindari overfitting
- Scoring: f1_weighted — sesuai dengan kebutuhan klasifikasi multikelas yang tidak seimbang.

kelebihan: 
- Sangat baik dalam menangani dataset dengan banyak fitur dan kompleksitas tinggi.
- meminimalisir overfitting setelah melakukan tuning menggunakan parameter yang telah digunakan.
- Mendukung interpretasi pentingnya fitur.
- hasil akurasi cukup tinggi

Kekurangan:
- berpotensi overfitting karena model terlalu hapal terhadap data yang dipelajari
- Lebih lambat dalam proses pelatihan dibandingkan model sederhana.
- Konsumsi memori tinggi untuk banyak pohon.
- Kurang interpretatif dibanding model linear.

## Evaluation
- Hasil evaluasi dari model yang telah dibangun memiliki dampak yang sangat berpengaruh terhadap prediksi kinerja siswa, karena hasil menunjukkan model cukup akurat dan seimbang. hasil evaluasi model klasifikasi yang dibangun telah menghasilkan akurasi skor di angka 91% dan evaluasi F1 score diangka 90% dengan memanfaatkan fitur yang tersedia pada data yaitu absensi, nilai GPA, dan lain sebagainya. menunjukkan bahwa model bisa memprediksi performa kinerja siswa berdasarkan fitur pada data dilihat melalui hasil akurasi dan F1 score pada evaluasi model.
- melalui hasil evaluasi model akurasi F1 score sebagai acuan hasil, menunjukkan model condong ke kelas minoritas dengan arti siswa dengan kinerja rendah dapat di identifikasi oleh model dengan cepat, sehingga sekolah menjadi terbantu dalam mendeteksi konerja siswa yang rendah dan dapat melakukan intervensi lebih dini dan efektif.
- model klasifikasi yang menggunakan random forrest ini berhasil dibangun sesuai harapan dengan menunjukkan evaluasi hasil performa yang cukup tinggi, pembangunan model ini berhasil dibangun dengan kontribusi dari semua fitur pada data dan beberapa fitur tersebut yang berkontirbusi lebih karena terdapat korelasi cukup tinggi yang diantaranya absensi, nilai GPA.
- dari solusi yang direncanakan sangat berdampak pada prediksi kinerja pada siswa. Strategi ini memberikan model yang dibuat menggunakan randomforrest menjadi yang kuat dan stabil dengan hasil tinggi, serta menangani menggunakan F1-score untuk ketidakseimbangan data dengan baik. Hasil evaluasi dengan menggunakan tunning GridSearchcv berdampak bahwa model mengurang dalam hal overfitting dan memberikan dampak pada prediksi yang dapat diandalkan.
