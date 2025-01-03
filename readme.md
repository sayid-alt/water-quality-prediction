# Water Quality Predictions

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

<img src='https://github.com/sayid-alt/water-quality-prediction/blob/main/images/plot_correlation.png?raw=true'/>

3. Outliers
Seperti yang disebutkan sebelumnya, meskipun distribusi data terlihat menyerupai *bell curve* yang menunjukkan sebaran yang normal, tetap masih terlihat beberapa data outliers pada setiap fitur.  Metode *interquartile* pada kali ini digunakan untuk mengidentifikasi data outliers, sehingga dapat kita lihat total dari data outliers yang kemudian dapat menentukan strategi yang sesuai.

<img src='https://github.com/sayid-alt/water-quality-prediction/blob/main/images/plot_outliers.png?raw=true' alt='outliers boxplot'/>

Gambar diatas menunjukkan outliers pada setiap fitur. Titik-titik diluar batas atas dan bawah merupakan data outliers. Meskipun demikian, diperlukannya kalkulasi jumlah secara tepat terkait data outliers tersebut.

Kode python berikut menghasilkan nilai output dari rangkuman mengenai data outliers

```python
# Count if the rows has an existing outliers of column
rows_with_outliers = train_split[train_split[(train_split < lower_bound) | (train_split > upper_bound)].any(axis=1)]
potability_0_outliers = rows_with_outliers[rows_with_outliers['potability'] == 0]
potability_1_outliers = rows_with_outliers[rows_with_outliers['potability'] == 1]

outliers_percentage = len(rows_with_outliers) / len(train_split) * 100
potability_0_outliers_percentage = len(potability_0_outliers) / len(train_split) * 100
potability_1_outliers_percentage = len(potability_1_outliers) / len(train_split) * 100

print(f"Number of rows with outliers: \33[33m{len(rows_with_outliers)}\33[0m")
print(f"Number of potability rows with outliers: \33[33m{len(potability_0_outliers)}\33[0m")
print(f"Number of non-potability rows with outliers: \33[33m{len(potability_1_outliers)}\33[0m\n")
print(f"Percentage of rows with outliers: \33[33m{outliers_percentage:.2f}%\33[0m")
print(f"Percentage of potability rows with outliers: \33[33m{potability_0_outliers_percentage:.2f}%\33[0m")
print(f"Percentage of non-potability with outliers: \33[33m{potability_1_outliers_percentage:.2f}%\33[0m")
```
Output
```
Number of rows with outliers: 267
Number of potability rows with outliers: 147
Number of non-potability rows with outliers: 120

Percentage of rows with outliers: 10.19%
Percentage of potability rows with outliers: 5.61%
Percentage of non-potability with outliers: 4.58%
```

**SUMMARY**:  

- The outliers are quite higher as it reaches to 10% of data has been indiceted as an outliers.
- Handling it with removal or transformation makes it poor quality of data. Which in this case related to data of water quality that requires accurate data.

**STRATEGY**:  

- Instead of removing or transforming the outliers data, we'll examine it using the robust algorithm like tree-based algorithm `(Decision Tree, Random Forest)`.

### Data Perprocessing

**1 . Membuat Fitur Baru dengan Tipe Kategorikal dari Fitur Numerikal** <br>
Fitur numerik diubah menjadi kategorikal berdasarkan rentang nilai tertentu, misalnya _binning_ untuk mengelompokkan data ke dalam kategori.

```python
class NumericalCutterAttribs(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self._columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self._columns:
            col_cut = pd.qcut(X[col], 4, labels=[0,1,2,3])
            df[f"{col}_cut"] = col_cut

        return df
```
Kode Python di atas adalah implementasi kelas yang merupakan turunan dari dua kelas utama di scikit-learn: BaseEstimator dan TransformerMixin. Berikut penjelasannya:

1. BaseEstimator:
	 BaseEstimator adalah kelas dasar (base class) di scikit-learn yang digunakan untuk membuat estimator kustom. Dengan mewarisi kelas ini, kita dapat memanfaatkan metode bawaan seperti pengelolaan parameter model (misalnya, mendapatkan atau mengatur parameter dengan get_params dan set_params).
2. TransformerMixin:
	TransformerMixin menyediakan kerangka kerja untuk membuat transformer kustom yang dapat digunakan dalam pipeline. Dengan mewarisi kelas ini, kita dapat menambahkan metode seperti fit_transform, sehingga proses transformasi data dapat diterapkan langsung dalam pipeline scikit-learn.

Jadi, implementasi ini memungkinkan kita membuat transformer atau model kustom yang dapat bekerja mulus dalam pipeline scikit-learn, baik untuk pelatihan maupun proses prediksi, tanpa harus menulis kode dari nol.

**4. Menskalakan Nilai Dataset antara -1 dan 1** <br>
	Selanjutnya, kita akan memasukkan semuanya ke dalam satu alur kerja. Berikut adalah kode cara penerapannya. Alur kerja terakhir adalah menskalakan semua nilai. Kita akan menggunakan metode StandardScaler, yang akan menormalkan semua nilai, sehingga semua nilai rata-rata akan sama dengan 0 dan simpangan baku sama dengan 1.



> ==Poin-poin diatas sebagai pipeline yang memastikan bahwa data **train** dan **test** memiliki struktur dan transformasi yang sama, sehingga kompatibel untuk digunakan dalam model pembelajaran mesin.==

Berikut contoh pada kode python dalam implementasi pipeline tersebut menggunakan library scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# columns that will cut to the categorical feture
columns_to_cut = ["ph", "hardness"]

# store the pipeline processsing.
preproc_pipe = Pipeline([
    ('columns_lowercase', LowerColumnNames()), # lowercase pipeline
    ('imputer', MultipleImputer()), # imputing the missing values
    ('numerical_cutter', NumericalCutterAttribs(columns_to_cut)), # cut numerical into categorical feature
    ('scaler', StandardScaler()) # scaling
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

### Resampling Unbalanced Data
Kita akan melihat bagaimana data didistribusikan pada level dimensi yang lebih rendah. Oleh karena itu, hal pertama yang akan kita lakukan adalah menurunkan level dimensi menjadi hanya 2 dimensi, sehingga dapat dengan mudahre diplot menjadi grafik. Dalam kasus ini, kita akan menggunakan metode pca.

<img src='https://github.com/sayid-alt/water-quality-prediction/blob/main/images/plot_target_distribution.png?raw=true' />
<img src='https://github.com/sayid-alt/water-quality-prediction/blob/main/images/target_distribution.png?raw=true' />

Plot tersebut memberi tahu kita bahwa distribusi antar kelas tercampur. Ini merupakan masalah yang cukup besar. Karena model akan lebih sulit dalam klasifikasi. Namun, bagaimanapun, kita akan mengambil sampel ulang kelas minoritas yang merupakan kelas yang dapat diminum, sehingga akan terdistribusi secara merata dengan kelas yang tidak dapat diminum.

Dalam kasus ini, kita akan menggunakan SMOTE (Synthetic Minority Over-sampling Technique) yang merupakan teknik populer untuk menangani ketidakseimbangan kelas dalam kumpulan data machine learning.

Berikut visualisasi data setelah *resampling*.

<img src="https://github.com/sayid-alt/water-quality-prediction/blob/main/images/distribution_target_distribution_barplot.png?raw=true"/>
<img src="https://github.com/sayid-alt/water-quality-prediction/blob/main/images/resampled_target_distribution.png?raw=true"/ >

Dari data yang sudah diresampling akan dilatih ulang untuk melihat kualitas dari model yang diterapkan. Berikut hasil ringkasan dari pelatihan pada data resamplling.

|     | model_names            | resampled_train_scores | resampled_test_scores |
| --- | ---------------------- | ---------------------- | --------------------- |
| 1   | RandomForestClassifier | 1.000000               | 0.64875               |
| 2   | SVC                    | 0.765957               | 0.62625               |
| 3   | LogisticRegression     | 0.509387               | 0.55250               |

Dari hasil diatas, original data menunjukkan hasil yang lebih baik pada model `Random Forest` dengan `train` dan `test` sample `1.00` dan `0.69` secara berurut. Dengan demikian, evaluasi dan hyperparameter-tuning akan menggunakan data original dan model `random forest`.

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

Kita telah melihat indikator overfit dari penilaian rangkaian kereta dan set pengujian. Di bawah ini kita melihat lebih dalam pada visualisasi makna tersebut

<img src='https://github.com/sayid-alt/water-quality-prediction/blob/main/images/learning_curve.png?raw=true'/>
### AUC
Kurva ROC digambar dengan menghitung rasio positif sebenarnya (TPR) dan rasio positif palsu (FPR) pada setiap ambang batas yang memungkinkan (dalam praktiknya, pada interval tertentu), lalu membuat grafik TPR di atas FPR.


<img src="https://github.com/sayid-alt/water-quality-prediction/blob/main/images/false_positive_rate_plot.png?raw=true"/>
AUC = 0,60 artinya kemampuan model menebak secara acak nilai positif sedikit lebih baik dari pada menebak secara acak, yang berarti AUC = 1,0 merupakan tebakan sempurna dan AUC = 0,5 merupakan tebakan acak.

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
<img src="https://github.com/sayid-alt/water-quality-prediction/blob/main/images/Confusion_Matrix_th_.png?raw=true"/>
Secara keseluruhan, model tersebut dapat menunjukkan air yang tidak layak minum dan air yang layak minum dengan lebih baik. Dan kita dapat melihat total positif palsu dan positif benar masing-masing adalah 75 dan 38. Dalam kasus kita, positif palsu lebih berisiko. Karena memprediksi air yang tidak layak minum sebagai air yang layak minum secara salah itu berbahaya. Dalam kasus tersebut, kita akan menyesuaikan ambang batas agar sesuai dengan persyaratan kita.

Bagaimanapun, kita harus meningkatkan nilai presisi, di mana positif palsu paling rendah. Oleh karena itu, kita mendefinisikan ambang batas sama dengan 0,7

Berikut Kode python yang digunakan untuk menampilkan ringkasan dari evaluasi model dengan ambang batas 0,7

```python
# Probabilities for the positive class (column 1)
y_test_proba = model.predict_proba(X_test_final)[:, 1]

# adjusting the threshold to 0.7 to be considered as a true class
threshold = 0.7

# Apply the threshold
predictions = np.where(y_test_proba > threshold, 1, 0)
target_names = ['non potable', 'potable']
report_cls = classification_report(y_test_final, 
                               predictions, 
                               target_names=target_names, 
                               output_dict=True)
print(f"th={threshold}\n")

fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
axes = axes.flatten()

# display confusion matrix
display(pd.DataFrame(report_cls).T)
display(ConfusionMatrixDisplay.from_predictions(y_test_final, predictions, cmap='rocket_r', ax=axes[0]))

# precsision recall display
display_prec_rec = PrecisionRecallDisplay.from_predictions(
    y_test_final, predictions, ax=axes[1]
)
_ = display_prec_rec.ax_.set_title("2-class Precision-Recall curve")

baseline_precision = sum(y_test_final) / len(y_test_final)  # Proportion of positives
plt.axhline(y=baseline_precision, color="red", linestyle="--", label="Chance Level")
plt.scatter(x=[report_cls['potable']['recall']],
            y=[report_cls['potable']['precision']])
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)

display(display_prec_rec)
```

Output

|              | precision | recall   | f1-score | support    |
| ------------ | --------- | -------- | -------- | ---------- |
| non potable  | 0.622465  | 0.997500 | 0.766571 | 400.000000 |
| potable      | 0.933333  | 0.054688 | 0.103321 | 256.000000 |
| accuracy     | 0.629573  | 0.629573 | 0.629573 | 0.629573   |
| macro avg    | 0.777899  | 0.526094 | 0.434946 | 656.000000 |
| weighted avg | 0.743779  | 0.629573 | 0.507742 | 656.000000 |
<img src="https://github.com/sayid-alt/water-quality-prediction/blob/main/images/eval_summary.png?raw=true"/>

Kesimpulan:
Presisinya sama dengan 0,93. Yang sebenarnya bagus untuk model yang memiliki positif palsu hanya sekitar 7% dari prediksi. Namun sebagai kompensasinya, recall sangat lemah yang nilainya hanya 0,054. Artinya model tersebut salah memprediksi air minum sebagai air tidak minum sekitar 0,95 probabilitas.

Namun, memprediksi air tidak minum sebagai air minum (positif palsu) lebih berisiko daripada sebaliknya. Dari evaluasi tersebut, kami akan menyesuaikan ambang batas sama dengan 0,7 untuk membuat presisi lebih tinggi yang berarti mengurangi jumlah positif palsu.

---

# Refernces:
* https://link.springer.com/article/10.1007/s12403-020-00357-6
* https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
* *Ji, Y., Wu, J., Wang, Y. et al. Seasonal Variation of Drinking Water Quality and Human Health Risk Assessment in Hancheng City of Guanzhong Plain, China. Expo Health **12**, 469–485 (2020). https://doi.org/10.1007/s12403-020-00357-6*
