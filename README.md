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


