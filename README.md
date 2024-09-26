# Laporan Proyek Machine Learning - Fila Sofiyati

## Domain Bisnis dan Ekonomi

Perkembangan teknologi di era digital saat ini sangat pesat, dengan banyak aktivitas yang kini dapat dilakukan secara praktis dari rumah, salah satunya adalah berbelanja. Platform e-commerce telah menjadi sarana utama bagi konsumen untuk melakukan pembelian (revenue), yang pada akhirnya menciptakan persaingan ketat di antara berbagai platform dan perusahaan [Badan Pusat Statistik Indonesia](https://web-api.bps.go.id/download.php?f=5g3vWbrA+R3h1GRyjNFOdHcxNW1QNmZVS1BvK0hRdEQ3VjRLT1NZYVFOa25qb1hscXBlbWxzU3prSzJWNUpXeGJwMHY1L3lNY1V5QXZmOW11TlBSQUlhVEFjNFNScjh2aS9xL1NEZVpvRlhiV3RNT2JPU3VCQnpndkEvYmlZdHpHaGNKV0N4enNYaEpaTHB2cDdqVE13a0pYVmdETmVFeVZFN0pBU2ZNUTQwRU1UQS81MkhwMXBYd2l0dFhYS2JzY2Frd1RjNU1LYkJRZGVtQWkzcW1lR041SnVwVU02a2VCNlBwbmwyWGFmMXJhbHZpbzF5U08wdEJPWHdMMjNvY0lRdy9IOHhPeXhYalJlRmc=&_gl=1*1xp0dwg*_ga*MjAzMjcyMjU2Ni4xNjk1MzA3OTIx*_ga_XXTTVXWHDB*MTcyNzI0OTAxOS4xOC4xLjE3MjcyNTAyNzkuMC4wLjA). Untuk tetap kompetitif dan meningkatkan penjualan, perusahaan e-commerce perlu memahami perilaku dan niat pembelian pengunjung platform mereka dengan lebih baik. Dataset "Online Shoppers Purchasing Intention" yang tersedia di [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) memberikan kesempatan untuk mendapatkan wawasan berharga tentang bagaimana pengunjung berinteraksi dengan situs e-commerce dan faktor-faktor yang mempengaruhi keputusan pembelian mereka. 
Dengan menganalisis dan mengolah dataset "Online Shoppers Purchasing Intention", perusahaan e-commerce dapat memperoleh wawasan yang mendalam tentang perilaku pengunjung mereka, mengoptimalkan strategi pemasaran dan pengalaman pengguna, serta meningkatkan pembeli dan pendapatan perusahaan.


## Business Understanding
Berikut beberapa pernyataan masalah yang akan di selesaikan berdasarkan pemaparan latar belakang, sebagai berikut:
- Apa faktor yang paling mempengaruhi pengunjung e-commerce dalam menghasilkan revenue?
- Bagaimana cara membuat model machine learning yang bisa mengklasifikasikan revenue dari pengunjung situs e-commerce?

### Goals
- Mengidentifikasi Pola Pembelian 
- Mengetahui faktor yang mempengaruhi revenue dari pengunjung e-commerce.
- Mampu membuat model yang memiliki akurasi serta nilai ROC AUC yang tinggi untuk mengklasifikasikan revenue dari pengunjung e-commerce

    ### Solution statements
    - Melakukan proses *Exploratory Data Analysis* untuk mengetahui fitur yang paling mempengaruhi revenue dar pengunjung e-commerce.
    - Menggunakan model *Machine Learning* untuk memprediksi revenue dari pengunjung e-commerce, menggunakan Deep Learning yaitu Artificial Neural Network.
## Data Understanding 
Dataset yang digunakan yaitu Online Shoppers Purchasing Intention untuk memprediksi pengunjung situs e-commerce dalam melakukan revenue yang tersedia di UCI Machine Learning Repository dari Creative Commons Attribution 4.0 International.
Dataset ini bertujuan untuk mendapatkan wawasan berharga tentang bagaimana pengunjung e-commerce berinteraksi dengan situs e-commerce dan faktor-faktor yang mempengaruhi keputusan pembelian (revenue) dari pengunjung. Dataset ini terdiri dari 1 file csv, 17 fitur dan 12.330 sampel.

### Informasi data:
| Fitur                               | Deskripsi                                                                                                                                            |
|-------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Administrative                      | Mewakili jumlah halaman administratif yang dikunjungi selama sesi.                                                                  |
| Administrative Duration             | Total waktu yang dihabiskan pada halaman administratif selama sesi.                                                                 |
| Informational                       | Mewakili jumlah halaman informasional yang dikunjungi selama sesi.                                                                  |
| Informational Duration              | Total waktu yang dihabiskan pada halaman informasional selama sesi.                                                                 |
| Product Related                     | Mewakili jumlah halaman terkait produk yang dikunjungi selama sesi.                                                                 |
| Product Related Duration            | Total waktu yang dihabiskan pada halaman terkait produk selama sesi.                                                                |
| Bounce Rate                         | Persentase pengunjung yang memasuki situs dari halaman tertentu dan kemudian meninggalkan situs tanpa melakukan permintaan lain ke server analitik. |
| Exit Rate                           | Persentase tampilan halaman untuk halaman tertentu yang merupakan halaman terakhir yang dilihat dalam sesi.                         |
| Page Value                          | Nilai rata-rata untuk halaman yang dikunjungi pengguna sebelum menyelesaikan transaksi e-commerce.                                  |
| Special Day                         | Menunjukkan seberapa dekat tanggal kunjungan dengan hari istimewa tertentu (misalnya, Hari Ibu, Hari Valentine), di mana sesi lebih mungkin diakhiri dengan transaksi. |
| Operating System                    | Sistem operasi yang digunakan oleh pengunjung.                                                                                      |
| Browser                             | Browser yang digunakan oleh pengunjung.                                                                                             |
| Region                              | Wilayah geografis pengunjung.                                                                                                       |
| Traffic Type                        | Jenis lalu lintas pengunjung (misalnya, organik, langsung, rujukan, iklan).                                                         |
| Visitor Type                        | Jenis pengunjung (kembali atau baru).                                                                                               |
| Weekend                            | Nilai Boolean yang menunjukkan apakah kunjungan terjadi pada akhir pekan.                                                            |
| Month                               | Bulan dalam setahun saat kunjungan terjadi.                                                                                         |

Pada Dataset tersebut berisikan informasi ppengunjung sebanyak 12.330 sampel dengan 17 fitur serta terdapat 0 missing values dan 125 data duplikat, untuk data yang duplikat di hapus sehingga hanya tersisa 12.205 sampel, kemudian pada dataset ini ada sekitar 17,9% outlier, karena outlier ini cukup besar, jadi tidak dihilangkan.

### Berikut rangkuman `statistik deskriptif` dari fitur dalam dataset: <br>


| **Fitur**                    | **count**   | **mean**   | **std**    | **min**    | **25%**    | **50%**    | **75%**    | **max**    |
|------------------------------|-------------|------------|------------|------------|------------|------------|------------|------------|
| **Administrative**            | 12205.0000  | 2.3389     | 3.3304     | 0.0000     | 0.0000     | 1.0000     | 4.0000     | 27.0000    |
| **Administrative_Duration**   | 12205.0000  | 81.6463    | 177.4918   | 0.0000     | 0.0000     | 9.0000     | 94.7000    | 3398.7500  |
| **Informational**             | 12205.0000  | 0.5087     | 1.2756     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 24.0000    |
| **Informational_Duration**    | 12205.0000  | 34.8255    | 141.4248   | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 2549.3750  |
| **ProductRelated**            | 12205.0000  | 32.0456    | 44.5936    | 0.0000     | 8.0000     | 18.0000    | 38.0000    | 705.0000   |
| **ProductRelated_Duration**   | 12205.0000  | 1206.9825  | 1919.6014  | 0.0000     | 193.0000   | 608.9429   | 1477.1548  | 63973.5222 |
| **BounceRates**               | 12205.0000  | 0.0204     | 0.0453     | 0.0000     | 0.0000     | 0.0029     | 0.0167     | 0.2000     |
| **ExitRates**                 | 12205.0000  | 0.0415     | 0.0462     | 0.0000     | 0.0142     | 0.0250     | 0.0485     | 0.2000     |
| **PageValues**                | 12205.0000  | 5.9496     | 18.6537    | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 361.7637   |
| **SpecialDay**                | 12205.0000  | 0.0619     | 0.1997     | 0.0000     | 0.0000     | 0.0000     | 0.0000     | 1.0000     |
| **OperatingSystems**          | 12205.0000  | 2.1242     | 0.9068     | 1.0000     | 2.0000     | 2.0000     | 3.0000     | 8.0000     |
| **Browser**                   | 12205.0000  | 2.3578     | 1.7101     | 1.0000     | 2.0000     | 2.0000     | 2.0000     | 13.0000    |
| **Region**                    | 12205.0000  | 3.1533     | 2.4023     | 1.0000     | 1.0000     | 3.0000     | 4.0000     | 9.0000     |
| **TrafficType**               | 12205.0000  | 4.0739     | 4.0167     | 1.0000     | 2.0000     | 2.0000     | 4.0000     | 20.0000    |

#### Interpretasi Statistika Deskriptif Data data 
- Mayoritas dataset memiliki angka **Mean < Median** , artinya distribusi data cenderung positively skewed.
- Pada fitur yang menunjukkan **traffic website** seperti 'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', dan 'PageValues mayoritas memiliki nilai yang menumpuk di angka 0. Sehingga dilakukan Feature Transformation menggunakan PowerTransformer Yeo-Johnson untuk membuat distribusi lebih mendekati normal (Guassian) dan mendukung value data memiliki nilai positif atau negatif.

### Feature Encoding
Beberapa fitur yang di encoding dengan one hot encoding yaitu 
- VisitorType dengan other dianggap sebagai returning visitor (modus). Jadi Returning Visitor: 1 dan New Visitor: 0.
- Revenue, dengan True: 1 dan False: 0.
- Weekend, dengan True: 1 dan False: 0.

Kemudian Fitur Month, disini akan dilakukan label encoding berdasarkan indeks terbanyak.

## Analisis Univariat dan Visualisasi Data
### Analisis Univariat Visitor Type dengan Revenue
![Persentase Revenue dengan Visitor Type](https://github.com/user-attachments/assets/df1c7aa4-94cb-4b56-8b06-e470c7d02aa3)
Dapat dilihat bahwa pada setahun terakhir jumlah kunjungan dan revenue pengunjung dari new visitor memiliki perbandingan yang cukup rendah di banding dengan returning visitor.

### Analisis Univariat Month dengan Revenue
![plt.title(Total Visitor per Month vs Revenue](https://github.com/user-attachments/assets/2741bed9-ffe4-482a-937a-e2de615723f1)
Dapat dilihat bahwa trafik jumlah kunjungan pelanggan setiap bulan memiliki jumlah yang paling tinggi pada bulan Mei dan di susul pada bulan November. Namun pada bulan Mei tingginya trafik tidak diikuti dengan tingginya angka Revenue yang hanya menghasilkan 11%. Sedangkan pada bulan November merupakan bulan yang memiliki cukup banyak pengunjung dengan nilai Revenue bulanan yang paling tinggi, yaitu mencapai 25%.

### Analisis Univariat Region dengan Revenue
![plt.title('Visitor Region vs Revenue', fontweight='bold')
](https://github.com/user-attachments/assets/2f26f158-48bc-471f-9491-ddd17201b2b0)
Kunjungan pelanggan didominasi pada region "1" dengan Revenue Rate yang cukup tinggi.

### Analisis Univariat Weekend dengan Revenue
![Persentase Revenue Coversion Rate saat Weekend'](https://github.com/user-attachments/assets/ae29a14b-2cc5-4a7d-86ed-f13a3afe0fbe)
Pada dasarnya presentase kunjungan Weekend dan Weekday tidak terlalu berbeda signifikan. Namun Weekday memiliki nilai yang lebih tinggi. Namun hal ini bisa terjadi dikarenakan memang jumlah hari di Weekday lebih banyak daripada Weekend atau memang pengunjung memang lebih sering mengunjungi website di hari-hari weekday.


## Analisis Multivariat dan Visualisasi Data
### Hubungan Antar Fitur Numerikal
![Heatmap Hubungan antar Fitur Numerikal](https://github.com/user-attachments/assets/41e9098d-2ea8-42e2-8c0c-2285665f6bae)

- Fitur ExitRates dan BounceRates memiliki nilai multikolinieritas yang tinggi 0.9, kedua kolom ini pun berhubungan satu sama lain sehingga dapat di drop salah satunya.
- Terdapat beberapa fitur-fitur yang kemungkinan redundan karena memiliki korelasi yang cukup tinggi diantaranya ProductRelated dengan - ProductRelated_Duration, Adminisitrative dengan Adminisitrative_Duration, Informational dengan Informational_Duration, dan begitu pula BounceRates dengan ExitRates. Dalam tahap data prepocessing feature-feature tersebut dapat di drop ataupun dipilih salah satu.
- Kolom PageValues ternyata memiliki korelasi tinggi dengan Revenue (0.49).
- Kolom BounceRates dengan beberapa kolom lain, ExitRates dengan beberapa kolom lain, dan page values dengan beberapa kolom lain berkumpul di bawah dan samping kiri cenderung membentuk pola logaritmik. Itu artinya, Apabila kolom BounceRates dan kolom Informational_Duration berhubungan secara logaritmik, semakin besar nilai BounceRates, nilai Informational_Duration semakin kecil secara logaritmik.

### Hubungan Antara Fitur Kategorikal
![Heatmap Hubungan antar Fitur Kategorikal](https://github.com/user-attachments/assets/cd0ae6e0-b86d-4e4f-98ba-df9cb6e86e87)
- Browser dan OperatingSystems memiliki korelasi yang cukup tinggi.
- Browser dan TrafficType juga memiliki korelasi yang tinggi. Nilai korelasi antar kolom dengan kolom Revenue cenderung rendah.

### Hubungan Antara Fitur Numerikal dan Kategorikal (Semua Fitur)

![Heatmap Hubungan Semua Fitur](https://github.com/user-attachments/assets/55076bf9-d8c3-4546-ac94-7f4d2cfa7cbc)

- Fitur ExitRates dan BounceRates memiliki nilai multikolinieritas yang tinggi 0.9, kedua kolom ini pun berhubungan satu sama lain sehingga dapat di drop salah satunya.
- Terdapat beberapa fitur-fitur yang redundan karena memiliki korelasi yang cukup tinggi diantaranya ProductRelated dengan ProductRelated_Duration, Adminisitrativedengan Adminisitrative_Duration, Informational dengan Informational_Duration, dan begitu pula BounceRates dengan ExitRates. Dalam tahap data prepocessing feature-feature tersebut dapat di drop ataupun dipilih salah satu.
- Kolom PageValues memiliki korelasi tinggi dengan Revenue (0.49)
- Browser dan OperatingSystems memiliki korelasi yang cukup tinggi.
- Browser dan TrafficType juga memiliki korelasi yang tinggi. Nilai korelasi antar kolom dengan kolom Revenue cenderung rendah.

## Data Preparation
### Feature Selection
Memilin fitur dengan korelasi tinggi dengan Revenue
| **Fitur**                 | **Korelasi dengan Revenue** |
|---------------------------|-----------------------------|
| Administrative            | 0.164376                    |
| Administrative_Duration   | 0.164306                    |
| Informational             | 0.110966                    |
| Informational_Duration    | 0.107878                    |
| ProductRelated            | 0.196981                    |
| ProductRelated_Duration   | 0.211123                    |
| BounceRates               | -0.172585                   |
| ExitRates                 | -0.249863                   |
| PageValues                | 0.611599                    |
| SpecialDay                | -0.088071                   |
| VisitorType               | -0.102694                   |
| Revenue                   | 1.000000                    |

Berdasar, nilai korelasi variabel dengan label revenue, maka fitur yang dipilih adalah Administrative, Administrative_Duration, Informational, Informational_Duratiom, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay, VisitorType.

Tetapi beberapa variabel diatas ada yang redundan yaitu

Administrative, Administrative_Duration dipilih Administrative_Duration,
Informational, Informational_Duratiom, dipilih Informational_Duration,
ProductRelated, ProductRelated_Duration, dipilih ProductRelated_Duration
Kemudian ada variabel yang multikolinearitas nya tinggi yaitu BounceRates, ExitRates jadi dipilih salah satu yaitu ExitRates.

### Splitting Data
- Fitur Revenue merupakan label data.
- Digunakan random state 42 untuk memastikan reprodusibilitas hasil pembagian data.
- test_size menentukan proporsi data yang akan dialokasikan untuk set pengujian. Dalam kasus ini, nilai 0.2 berarti 20% dari data akan digunakan untuk pengujian, sementara 80% akan digunakan untuk pelatihan.
- Pembagian data dilakukan dengan stratifikasi, untuk memastikan distribusi kelas pada set pelatihan dan set pengujian tetap seimbang. Hal ini penting karena kelas Revenue tidak seimbang dalam dataset.
- fungsi StandarScaler untuk melakukan normalisasi (scaling) pada data. Normalisasi adalah proses mengubah fitur-fitur data sehingga memiliki rata-rata 0 dan standar deviasi 1.

## Model Development
Deep Learning adalah subbidang machine learning yang memanfaatkan jaringan saraf tiruan (*artificial neural networks*) dengan banyak lapisan untuk mempelajari pola dan informasi dari data.

Dalam penerapan ini, TensorFlow digunakan untuk membangun model jaringan saraf tiruan. Berikut penjelasan langkah-langkah pembuatannya:

### Membangun Model
- Kelas **Sequential** diimpor dari TensorFlow untuk membuat model secara sekuensial.
- Kelas **Dense** digunakan untuk menambahkan lapisan-lapisan ke dalam model.
- **Dropout** juga diimpor untuk mencegah *overfitting*, dengan secara acak menonaktifkan beberapa neuron selama pelatihan.

### Arsitektur Model
1. **Lapisan Input**: 
   - Lapisan input memiliki 12 neuron, fungsi aktivasi **ReLU**, dan bentuk input `(7,)`. Model ini dirancang untuk menerima data dengan dimensi 7.
   
2. **Lapisan Tersembunyi**:
   - Ditambahkan dua lapisan tersembunyi, masing-masing dengan 24 dan 48 neuron, menggunakan fungsi aktivasi **ReLU**.
   
3. **Dropout Layer**:
   - Lapisan **Dropout(0.17)** digunakan untuk mengurangi risiko *overfitting*, dengan "mengabaikan" 17% neuron secara acak selama pelatihan.

4. **Lapisan Output**:
   - Lapisan output memiliki 1 neuron yang menggunakan fungsi aktivasi **sigmoid**. Aktivasi sigmoid cocok untuk tugas klasifikasi biner karena menghasilkan output probabilitas antara 0 dan 1, yang mewakili kemungkinan kelas positif.

### Loss Function dan Optimizer
- **Fungsi Loss**: Fungsi loss bertugas mengukur performa model. Nilai loss adalah yang akan diminimalkan oleh model selama pelatihan.
  
- **Optimizer Adam**: Kelas **Adam** dari `tensorflow.keras.optimizers` digunakan untuk memperbarui bobot model. 
  Berikut adalah parameter dari optimizer **Adam**:
  
  - `learning_rate=0.00001`: Menentukan seberapa besar perubahan bobot model dalam setiap iterasi. Nilai ini kecil sehingga model belajar lebih lambat namun stabil.
  - `beta_1=0.9`: Nilai ini mengontrol pengaruh rata-rata gradien pertama dalam pembaruan bobot. Nilai besar seperti 0.9 membantu membuat rata-rata gradien lebih halus.
  - `beta_2=0.999`: Nilai ini mengontrol pengaruh rata-rata kuadrat gradien kedua, yang sangat besar, sehingga membuat model lebih stabil.
  - `amsgrad=False`: Modifikasi dari Adam yang tidak digunakan dalam model ini.

### Kompilasi Model
- Model dikompilasi dengan menentukan:
  - **Fungsi loss**: Untuk mengukur kesalahan prediksi model.
  - **Optimizer**: Algoritma untuk memperbarui bobot model.
  - **Metrik evaluasi**: Seperti akurasi untuk memantau kinerja model selama pelatihan.

### Melatih Model
- Fungsi **fit** digunakan untuk melatih model dengan data pelatihan.
- Parameter yang digunakan:
  - `epochs=100`: Model akan melalui seluruh dataset pelatihan sebanyak 100 kali.
  - `batch_size=64`: Dalam setiap iterasi, model akan memproses 64 sampel data sekaligus.

## Evaluasi
**Hasil Evaluasi Model**
- **Training progress**: 306/306 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step
- **Evaluation progress**: 77/77 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step
- **Accuracy**: 0.8921
- **Loss**: 0.2351
## Classification Report

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| 0          | 0.92      | 0.96   | 0.94     | 2059    |
| 1          | 0.71      | 0.55   | 0.62     | 382     |
| **Accuracy**  |           |        | **0.90** | **2441** |
| **Macro Avg** | 0.82      | 0.76   | 0.78     | 2441    |
| **Weighted Avg** | 0.89      | 0.90   | 0.89     | 2441    |

- **Precision**: Proporsi prediksi positif yang benar.
- **Recall**: Proporsi data positif yang berhasil ditemukan oleh model.
- **F1-Score**: Harmoni antara precision dan recall.
- **Support**: Jumlah sampel sebenarnya dari setiap kelas.
## Interpretasi Metrik

- **Accuracy**: Akurasi model adalah sekitar **90%**, menunjukkan seberapa baik model dapat memprediksi dengan benar dari seluruh sampel.

### Precision:
- **0 (Kelas Negatif)**: Dari semua prediksi yang dilakukan sebagai kelas negatif, sekitar **92%** benar-benar negatif.
- **1 (Kelas Positif)**: Sekitar **71%** dari semua prediksi sebagai kelas positif adalah benar.

### Recall:
- **0 (Kelas Negatif)**: Dari semua sampel yang benar-benar negatif, model mampu mengenali sekitar **96%**.
- **1 (Kelas Positif)**: Dari semua sampel yang benar-benar positif, model hanya dapat mengenali sekitar **55%**.

### F1-Score:
- F1-score adalah rata-rata harmonis dari precision dan recall.
  - **Kelas 0**: F1-score sebesar **94%**, yang mengindikasikan keseimbangan yang baik antara precision dan recall.
  - **Kelas 1**: F1-score sebesar **62%**, menunjukkan performa model yang cukup baik dalam mendeteksi kelas positif, meskipun precision dan recall belum optimal.

F1-score yang lebih tinggi mengindikasikan bahwa model mampu menyeimbangkan precision dan recall dengan baik, terutama pada kelas negatif.

![Confusion Matrix](https://github.com/user-attachments/assets/05f53b9b-2ef7-40cb-9dfe-6c0bb8b833bd)
- True Positif (TP) (data positif yang diprediksi benar) customer benar melakukan pembelian: 212
- False Positif (FP) (data negatif yang diprediksi sebagai data positif) customer tidak melakaukan pembelian, tapi diprediksi melakukan pembelian: 170
- False Negatif (FN) (data positif yang diprediksi sebagai data negatif) customer melakukan pembelian, tapi diprediksi tidak melakukan pembelian: 86
- True Negatif (TN) (data negatif yang diprediksi benar) customer benar tidak melakukan pembelian : 1973

### ROC AUC CURVE
![ROC CURVE](https://github.com/user-attachments/assets/cbfcde32-aef0-40d5-8803-7054a864286f)
- ROC AUC : Mengukur seberapa baik model dapat membedakan antara kelas positif dan negatif.
- Sumbu x menunjukan nilai False Positive Rate (FPR) dan Sumbu y menunjukan True Positif Rate (TPR). Kurva ROC yang memiliki kinerja lebih baik adalah kurva yang lebih dekat dengan sudut kiri atas. AUROC yaitu area di bawah kurva ROC, semakin besar luas dibawah ROC, semakin baik dan semakin besar skor AUC, semakin Baik suatu mode
- Dimana pada model ini Nilai AUC yaitu 0.91 dan berada di range 0.9 - 1.00 , ini berarti KLASIFIKASINYA SANGAT BAIK dalam membedakan kelas negatif dan positif (customer yang melakukan pembelian dan tidak).

## Interpretasi:
- **Fitur yang paling berpengaruh dalam menghasilkan revenue oleh pelanggan yaitu Administrative_Duration, Informational_Duration ProductRelated_Duration, ExitRates, PageValues, SpecialDay, VisitorType.**
- **Dengan menggunakan Deep Learning, akurasi secara keseluruhan adalah 90% dan Nilai ROC AUC 91%, maka model ini sudah sangat baik digunakan untuk memisahkan customer yang diprediksi akan melakukan pembelian (kelas positif) atau tidak (kelas negatif) di E-commerce.**











