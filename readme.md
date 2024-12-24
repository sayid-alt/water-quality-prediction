# Laporan Proyek Machine Learning - Sayid Muhammad Heykal

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

### Data Cleaning

**1. Menyamakan Penamaan Kolom menjadi Lower Case** <br>
Tujuannya adalah untuk konsistensi dalam penamaan kolom, sehingga lebih mudah diakses dan tidak rentan terhadap kesalahan akibat case-sensitive.

Contoh kode yang digunakan.
```python
class LowerColumnNames(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.columns = X.columns.str.lower()
        return X

train_split = LowerColumnNames().fit_transform(train_split)
```

**2. Mengisi Missing Values dengan Metode Multiple Imputer**<br>
Menggunakan teknik imputasi berbasis algoritma, seperti _Iterative Imputer_ atau _KNN Imputer_, untuk mengisi nilai yang hilang berdasarkan pola dalam data lain.


Contoh kode yang digunakan:
```python
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

class MultipleImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        imp = IterativeImputer(random_state=9122024)
        return pd.DataFrame(imp.fit_transform(X), columns=features)
        
train_cleaned = MultipleImputer().fit_transform(train_cleaned)
```

### Exploratory Data Analyst (EDA)
Melihat sebaran data yang akan diolah untuk memahami lebih dalam data tersebar pada dataset.
1. Distribusi Data

Plot distribusi data pada target (potable water & non-potable water)
<img src='https://github.com/sayid-alt/water-quality-prediction/blob/main/images/plot_target_distribution.png?raw=true' />

Plot gambar diatas menunjukkan ketidakseimbangan data target pada dataset atau disebut dengan (unbalanced data). Sehingga dibutuhkan beberapa teknik preparation dan evaluasi pada model. Seperti resampling pada untuk` _preparation_` dan teknik `_precisition_` & `_recall_` pada evaluasi.

<img src='https://github.com/sayid-alt/water-quality-prediction/blob/main/images/plot_data_distribution.png?raw=true' />

Grafik diatas menunjukkan sebaran data yang normal pada setiap fitur. walaupun terlihat normal, terlihat pada beberapa fitur yang berpotensi memiliki nilai outliers.

2. Correlation
Merupakan pengukuran korelasi antar fitur, dengan indikator jarak -1 dan 1 yang dikategorikan sebagai positif atau negatif korelasi. Berikut nilai korelasi antar fitur dengan diagram heatmap





**3. Membuat Fitur Baru dengan Tipe Kategorikal dari Fitur Numerikal** <br>
Fitur numerik diubah menjadi kategorikal berdasarkan rentang nilai tertentu, misalnya _binning_ untuk mengelompokkan data ke dalam kategori.

**4. Menskalakan Nilai Dataset antara -1 dan 1** <br>
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

**1. Logistic Regression** <br>
Kelebihan Logistic Regression adalah interpretasi hasil yang jelas, efisien untuk dataset kecil hingga menengah, dan tidak memerlukan banyak tuning. Namun, model ini kurang efektif untuk data non-linier, sensitif terhadap outlier, dan tidak fleksibel dalam menangani hubungan kompleks antar variabel.

**2. Support Vector Machine (SVM)** <br>
SVM unggul dalam menangani data berdimensi tinggi, bekerja baik dengan data non-linier berkat kernel, dan optimal untuk dataset kecil. Kekurangannya adalah proses pelatihan yang lambat untuk dataset besar, hasil sulit diinterpretasi, dan kinerja sangat bergantung pada tuning parameter.

**3. Random Forest** <br>
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


## Evaluation

Metrik evaluasi yang digunakan pada pengembangan model ini antara lain:
1. **Accuracy**
	Accuracy adalah metrik evaluasi yang mengukur sejauh mana model memprediksi kelas dengan benar. Ini dihitung dengan membandingkan jumlah prediksi benar terhadap total jumlah data. Accuracy sering digunakan sebagai metrik dasar untuk _classification tasks_, tetapi menjadi kurang efektif jika dataset memiliki kelas yang tidak seimbang, seperti pada ini.
2. **Precision**
	Precision mengukur seberapa akurat model dalam memprediksi kelas positif dari semua prediksi positif yang dibuat. Fokus utamanya adalah meminimalkan _False Positive_ (FP), yaitu prediksi salah untuk kelas positif.


### Precision
Pada kasus ini, matrik Precision akan dimaksimalkan nilainya, melihat False Positive sangat beresiko. Artinya, dalam keadaan model salah dalam memprediksi air bersifat potable menjadi resiko yang sangat fatal, `karena model akan memprediksi air kotor sebagai air bersih`.

Nilai probabilitas pada prediksi model akan dibatasi dengan `threshold=0.7`, berartikan model akan dianggap `True` apabila probabilitas mencapai diatas threshold yang ditentukan. Berikut tabel hasil klasifikasi dan confusion martix:

#### Classification Report

|              | precision | recall   | f1-score | support    |
| ------------ | --------- | -------- | -------- | ---------- |
| non potable  | 0.635634  | 0.990000 | 0.774194 | 400.000000 |
| potable      | 0.878788  | 0.113281 | 0.200692 | 256.000000 |
| accuracy     | 0.647866  | 0.647866 | 0.647866 | 0.647866   |
| macro avg    | 0.757211  | 0.551641 | 0.487443 | 656.000000 |
| weighted avg | 0.730523  | 0.647866 | 0.550388 | 656.000000 |
|              |           |          |          |            |

#### Confusion Matrix
<img src="https://github.com/sayid-alt/water-quality-prediction/blob/main/images/Pasted%20image%2020241216182324.png?raw=true"/>


---

# Refernces:
* https://link.springer.com/article/10.1007/s12403-020-00357-6
* *Ji, Y., Wu, J., Wang, Y. et al. Seasonal Variation of Drinking Water Quality and Human Health Risk Assessment in Hancheng City of Guanzhong Plain, China. Expo Health **12**, 469–485 (2020). https://doi.org/10.1007/s12403-020-00357-6*
