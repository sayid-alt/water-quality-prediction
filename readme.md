# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Kualitas air merupakan sumber daya alam yang memiliki peran besar dalam faktor kesehatan manusia dan kehidupan lainnya. Suatu tempat dianggap sebagai lingkungan yang layak huni apabila indeks air bersih pada area tersebut sebagian besarnya bersih. Selain itu, tertulis pada website https://www.usgs.gov/special-topics/water-science-school/science/water-you-water-and-human-body, bahwa badan manusia dewasa mengandung air hingga 60%, dan pada sebagian orgasme ada yang mencapai 90%.  

Sudah banyak penelitian yang melakukan berbagai asesmen maupun analisa terhadap kualitas air pada suatu tempat atau waktu tertentu. 

Assesmen yang dilakukan oleh [Ji, Y., Wu, J., Wang, Y. _et al](https://link.springer.com/article/10.1007/s12403-020-00357-6)._ di Hancheng City, China. Berjumlah 48 sampel dari sistem air minum dalam musim kemarau dan hujan. Sampel tersebut dianalisa menggunakan entropy water quality index (EWQI) dengan hasil sebesar 80% dapat diminum atau digunakan dengan kebutuhan domestik. Resiko juga yang dinilai menggunakan rekomendasi model dari US Environmental Protection Agency (USEPA) menghasilkan 2 jenis, Pada musim kemarau resiko pada _non-carcinogenic_ lebih tinggi dari pada musim hujan terhadap orang dewasa maupun anak-anak. Resiko Carcinogenic lebih besar pada musim kemarau yang disebabkan dari *chlorinated water* dan pada musim hujan *carcinogenic* memiliki resiko lebih besar yang disebabkan oleh *terminal tap water*. 

Metode tradisional seperti _Entropy Water Quality Index (EWQI)_ memerlukan analisis manual, yang kurang efisien jika diterapkan pada skala besar, pembuatan model klasifikasi menggunakan machine learning akan meningkatan efesiensi dalam identifikasi pada kualitas air tertentu. Dengan memanfaatkan algoritma yang menghitung dan menyesuaikan pada variabel-variabel tertentu, model akan mempelajari dari data yang ada untuk dapat mengidentifikasi secara otomatis terhadap kualitas air.

## Business Understanding

### Problem Statements
---
Menjelaskan pernyataan masalah latar belakang:
- Metode tradisional seperti _Entropy Water Quality Index (EWQI)_ memerlukan analisis manual, yang kurang efisien jika diterapkan pada skala besar

### Goals
___
Menjelaskan tujuan dari pernyataan masalah:
- Pembuatan model untuk klasifikasi antara air layak atau tidak layak dikonsumsi  secara otomatis menggunakan algoritma machine learning.

### Solution statements
---
1. Logistic Regression:
    - Sebagai basic model pada *binary classification*.
    - Hasil yang dapat diterjemahkan dengan jelas, dan baik dalam mengklasifikasi data linier
    
1. Support Vector Machine (SVM):
    - Efektif untuk kumpulan data berdimensi tingg.
    - Menangani kekurangan pada klasifikasi data non-linier dengan penggunaan kernel.

2. **Random Forest**
	- Menggabungkan beberapa algoritma *Decision Tree* untuk meningkatkan akurasi dan mengurangi overfitting dengan merata-ratakan prediksi (untuk regresi) atau pemungutan suara mayoritas (untuk klasifikasi).

## Data Understanding
---
Dataset **water_potability.csv** berisi 3276 sampel kualitas air yang dievaluasi berdasarkan berbagai parameter, seperti **pH**, **Hardness**, **TDS**, **Chloramines**, **Sulfate**, **Conductivity**, **Organic Carbon**, **Trihalomethanes**, dan **Turbidity**. Parameter-parameter ini mengukur sifat kimia dan fisik air sesuai dengan standar WHO dan US EPA untuk menentukan kelayakan konsumsi. Label **Potability** menunjukkan apakah air aman diminum (1) atau tidak (0). Dataset ini berguna untuk analisis kualitas air dan pengambilan keputusan terkait pengelolaan sumber daya air. 
Sumber dataset diambil dari webiste kaggle: [kaggle dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability).

### Variabel-variabel pada Water Quality Dataset adalah sebagai berikut:
---
1. **pH**: Mengukur keseimbangan asam-basa air (standar WHO: 6,5-8,5).
2. **Hardness**: Kandungan kalsium dan magnesium dalam air.
3. **Solids (TDS)**: Total zat terlarut dalam air (standar WHO: <500 mg/L).
4. **Chloramines**: Disinfektan air minum (aman hingga 4 mg/L).
5. **Sulfate**: Kandungan sulfat dari sumber alami atau industri.
6. **Conductivity**: Kemampuan air menghantarkan listrik (standar WHO: <400 µS/cm).
7. **Organic Carbon**: Kandungan karbon organik dari bahan alami/sintetis (standar US EPA: <4 mg/L).
8. **Trihalomethanes (THMs)**: Senyawa kimia dari proses klorinasi (aman hingga 80 ppm).
9. **Turbidity**: Tingkat kekeruhan air (standar WHO: <5 NTU).
10. **Potability**: Status kelayakan air untuk diminum (1 = layak, 0 = tidak layak).

## Data Preparation
---
**1. Menyamakan Penamaan Kolom menjadi Lower Case**
Tujuannya adalah untuk konsistensi dalam penamaan kolom, sehingga lebih mudah diakses dan tidak rentan terhadap kesalahan akibat case-sensitive.

**2. Mengisi Missing Values dengan Metode Multiple Imputer**
Menggunakan teknik imputasi berbasis algoritma, seperti _Iterative Imputer_ atau _KNN Imputer_, untuk mengisi nilai yang hilang berdasarkan pola dalam data lain.

**3. Membuat Fitur Baru dengan Tipe Kategorikal dari Fitur Numerikal**
Fitur numerik diubah menjadi kategorikal berdasarkan rentang nilai tertentu, misalnya _binning_ untuk mengelompokkan data ke dalam kategori.

**4. Menskalakan Nilai Dataset antara -1 dan 1**
Menskalakan nilai fitur untuk memastikan bahwa data numerik berada dalam skala yang sama, yang penting untuk algoritma berbasis gradien atau jarak.

> ==Poin-poin diatas sebagai pipeline yang memastikan bahwa data **train** dan **test** memiliki struktur dan transformasi yang sama, sehingga kompatibel untuk digunakan dalam model pembelajaran mesin.==

Berikut contoh pada kode python dalam implementasi pipeline tersebut menggunakan library scikit-learn

```python
preproc_pipe = Pipeline([
    ('columns_lowercase', LowerColumnNames()),  # Lowercase column names
    ('imputer', IterativeImputer()),            # Fill missing values
    ('numerical_cutter', NumericalCutterAttribs(columns_to_cut=['ph', 'hardness'])),  # Extract categorical features
    ('scaler', StandardScaler())               # Scale dataset to range -1, 1
])

```
## Modeling
### Training and Predictions

Proses modeling ini dilakukan dengan tahap awal  dengan melakukan training dan prediksi pada setiap model dari algoritma **[Logistic Regression](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html)**, **[Support Vector Machine (SVM)](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html)**, dan [**Random Forest**](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html), dengan parameters default untuk melihat akurasi mana yang lebih superior dalam prediksi.

Model yang menunjukan akurasi lebih tinggi dalam prediksi akan dipilih untuk proses hyperparameter-tuning yang bertujuan mendapatkan akurasi yang lebih optimal. Proses ini akan dilakukan secara otomatis menggunakan [RandomSearchCV](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) dari scikit-learn libary

Selain itu, masing-masing model memiliki kelebihan dan kekurangan tersendiri, Berikut adalah masing-masing kelebihan dan kekurangan dari algoritma **Logistic Regression**, **Support Vector Machine (SVM)**, dan **Random Forest** secara global:

**1. Logistic Regression**
Kelebihan Logistic Regression adalah interpretasi hasil yang jelas, efisien untuk dataset kecil hingga menengah, dan tidak memerlukan banyak tuning. Namun, model ini kurang efektif untuk data non-linier, sensitif terhadap outlier, dan tidak fleksibel dalam menangani hubungan kompleks antar variabel.

**2. Support Vector Machine (SVM)**
SVM unggul dalam menangani data berdimensi tinggi, bekerja baik dengan data non-linier berkat kernel, dan optimal untuk dataset kecil. Kekurangannya adalah proses pelatihan yang lambat untuk dataset besar, hasil sulit diinterpretasi, dan kinerja sangat bergantung pada tuning parameter.

**3. Random Forest**
Random Forest mampu mengurangi overfitting, bekerja baik pada data kompleks dan besar, serta dapat mengidentifikasi fitur penting. Namun, model ini sulit diinterpretasikan, membutuhkan komputasi tinggi, dan performa optimal memerlukan tuning parameter yang cermat.

Hasil dari training didapatkan dengan model Random Forest memiliki akurasi yang lebih tinggi, selanjutnya model tersebut yang akan digunakan dalam [hyperparameter-tuning](###Hyperparameter-tuning).

**Hasil training pada 3 model percobaan**

| model_names            | original_train_scores | original_test_scores |
| ---------------------- | --------------------- | -------------------- |
| RandomForestClassifier | 1.000000              | 0.685976             |
| SVC                    | 0.746565              | 0.679878             |
| LogisticRegression     | 0.609924              | 0.609756             |

### Hyperparameter-tuning
Proses ini akan menggunakan Model [**Random Forest**](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html), sebagai estimator dengan parameters yang akan menjadi percobaan sebagai berikut:

| Parameters        | Values               |
| ----------------- | -------------------- |
| n_estimators      | [100, 150, 250, 850] |
| max_depth         | [25, 35, 85, None]   |
| min_samples_split | [2, 15, 20, 30, 35]  |
| min_samples_leaf  | [1, 15, 20]          |
Contoh implementasi pada kode python
```python
params_rf = {
    'n_estimators' : [100, 150, 250, 850],
    'max_depth' : [25, 35, 85, None],
    'min_samples_split' : [2, 15, 20, 30, 35],
    'min_samples_leaf' : [1, 15, 20]
}

grid_search_rf = GridSearchCV(estimator=models_dict['rf'], 
                                      param_grid=params_rf, 
                                      verbose=3, 
                                      cv=2, 
                                      scoring='accuracy', 
                                      return_train_score=True)


grid_search_rf.fit(X_train_resampled, y_train_resampled)
```
Hasil dari pencarian grid search:
```
RandomForestClassifier(max_depth=35, n_estimators=850, random_state=42)
```

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

