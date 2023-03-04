# SVM_CLASSIFIER
Hafid Ahmad Adyatma(1301194235)

## Latar Belakang
Klasifikasi linier adalah alat yang berguna dalam machine learning dan data mining. Untuk beberapa
data dalam ruang dimensi yang kaya, kinerja (yaitu, akurasi pengujian) pengklasifikasi linier telah
terbukti mendekati pengklasifikasi nonlinier seperti metode kernel, tetapi kecepatan training dan testing
jauh lebih cepat.
Support Vector Machine(SVM) adalah metode pada machine learning yang dapat digunakan untuk
menganalisis data dan mengurutkannya ke dalam salah satu dari dua kategori. SVM ditemukan oleh
Vladimir N. Vapnik dan Alexey Ya. Chervonenkis pada tahun 1963. Sejak itu, SVM telah digunakan
dalam klasifikasi teks, hiperteks dan gambar. SVM dapat bekerja dengan karakter tulisan tangan dan
algoritma ini telah digunakan di laboratorium biologi untuk melakukan tugas seperti menyortir protein.
SVM bekerja untuk mencari hyperplane atau fungsi pemisah (decision boundary) terbaik untuk
memisahkan dua buah kelas atau lebih pada ruang input. Hyperplane dapat berupa line atau garis pada
dua dimensi dan dapat berupa flat plane pada multiple plane.
Berdasarkan hal di atas, maka akan dilakukan klasifikasi linear menggunakan support
vector machine dengan metode evaluasi performa menggunakan RMSE. Dengan harapan dapat
menghasilkan performansi yang baik.

Klasifikasi linear menggunakan support vector machine yang menghasilkan perfomansi yang baik
dengan melihat hasil dari RMSE. Dalam tugas besar ini menggunakan <a href = "https://drive.google.com/file/d/1jAGHMNSBxmS5CfZjfyRVhEbFDdEbTEor/view"> Dataset</a> DataClassification.


## Persiapan dan Eksplorasi Data
Pertama, import package lalu import dataset dari google drive dengan format .csv
```
# Setup setting terkait dengan plotting
import numpy as np                 # import numpy (mathematical operation)
import matplotlib.pyplot as plt      # import package untuk plotting
import pandas as pd           
from matplotlib import rcParams    # runtime configuration (rc)
rcParams['figure.figsize']    = (14,7)   # untuk membuat visualisasi lebih baik, modif parameter dibawah ini
rcParams['axes.spines.top']   = False
rcParams['axes.spines.right'] = False
```
### Import Dataset
```
!gdown --id 1jAGHMNSBxmS5CfZjfyRVhEbFDdEbTEor
```

### Melihat 10 data teratas
```
df = pd.read_csv('DataClassification.csv')
df.head(10)
```
<img width="254" alt="image" src="https://user-images.githubusercontent.com/57959734/222879983-77a51d8c-e674-4a00-8c6e-94720fc26c3b.png">

### Melihat dimensi data dan tipe data
```
df.shape

df.dtypes
```
<img width="133" alt="image" src="https://user-images.githubusercontent.com/57959734/222880074-9dbe7fd7-6f6f-43c7-90e7-30d250f602cd.png">

### Melihat statical summary
```
df.describe()
```
<img width="235" alt="image" src="https://user-images.githubusercontent.com/57959734/222880129-9a7f9f8c-cdb3-4b40-9365-267c2aadd4b8.png">

### Melihat Outliers,
Untuk melihat lebih jelas apakah terdapat oulier pada variabel di atas, disini akan dibuat boxplot untuk memvisualisasikan outliers dalam variabel di atas.
```
plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.boxplot(column='x')
fig.set_title('')
fig.set_ylabel('x')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Y')
fig.set_title('')
fig.set_ylabel('Y')
```
<img width="539" alt="image" src="https://user-images.githubusercontent.com/57959734/222880212-490bfc1f-04eb-4ba6-96fc-ac5a04752485.png">

### Membuat histogram untuk memeriksa distribusi dan untuk mengetahui apakah normal atau miring
```
plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df['x'].hist(bins=20)
fig.set_xlabel('x')
fig.set_ylabel('Number of x')


plt.subplot(2, 2, 2)
fig = df['Y'].hist(bins=20)
fig.set_xlabel('Y')
fig.set_ylabel('Number of Y')
```
<img width="540" alt="image" src="https://user-images.githubusercontent.com/57959734/222880313-ea47177f-c94b-48c4-81c9-6b1b688bcb3a.png">
Dapat dilihat pada gambar di atas, bahwa untuk kedua variabel tersebut normal

### Membuat matriks korelasi
Membuat matriks korelasi yang mengukur hubungan linier antara variabel. Matriks korelasi
dapat dibentuk dengan menggunakan fungsi corr dari library kerangka data pandas. Saya akan
menggunakan fungsi peta panas / Heat Map dari library seaborn untuk memplot matriks korelasi

```
import seaborn as sns
corr = df.corr()

plt.figure(figsize=(16,8))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':10}, cmap='Greens',)
```
<img width="347" alt="image" src="https://user-images.githubusercontent.com/57959734/222880832-066a4b90-f8bd-4a07-915c-c6ad91faab21.png">
Dapat dilihat pada gambar di atas, yang dimana mendekati 1.0 variabel tersebut memiliki
korelasi yang kuat seperti Y dengan x dan untuk korelasi yang lemah yaitu x dengan LABEL
atau Y dengan LABEL

### Split Data
Membagi dataset menjadi data train dan data test
```
from sklearn.model_selection import train_test_split #split dataset
X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
```
<img width="444" alt="image" src="https://user-images.githubusercontent.com/57959734/222881017-5f7f4933-8a33-42db-9d14-2ba2109cf682.png">

### Feature Scaling
Fungsinya yaitu agar rentang nilai antara variabel itu sama, tidak ada yang terlalu besar maupun
kecil sehingga dapat membuat analisis statistic menjadi lebih mudah
```
from sklearn.preprocessing import StandardScaler #Feature Scaling

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
```

## Pemodelan
### Support Vector Machine
<img width="209" alt="image" src="https://user-images.githubusercontent.com/57959734/222881113-e227a218-5624-4251-8211-1f96b6ff196a.png">
Support Vector Machine (SVM) adalah salah satu algoritma machine learning dengan
pendekatan supervised learning yang bekerja dengan mencari hyperplane atau fungsi pemisah
terbaik untuk memisahkan kelas. Algoritma SVM memiliki konsep dan dasarmatematis yang
mapan sehingga menjadi algoritma yang populer. Algoritma ini dapat digunakan untuk
klasifikasi (SVM classification) dan regresi (SVM regression). Dan yang kita gunakan adalah
SVM classification.

### Visualisasi 
Menggambarkan 2 data yang terpisah yaitu dengan tipe -1 (tanda +) dan tipe-2 (tanda o)
<img width="510" alt="image" src="https://user-images.githubusercontent.com/57959734/222881156-aadaf85a-5fe7-4aaf-a72d-90d335ecd215.png">

## Evaluasi
### Model Support Vector Machine ( Secara Qualitative )
Untuk SVM saya menggunakan bantuan dari library sklearn untuk memodelkan SVM dengan
parameter hyperplane yang dibuat dengan hasil sebagai berikut.

<img width="479" alt="image" src="https://user-images.githubusercontent.com/57959734/222881249-b644c6ab-16e7-42f8-a7f7-0595754a7d51.png">

Berdasarkan gambar di atas, Hyperplane yang dimodelkan oleh SVM posisinya berada ditengahtengah antara dua kelas, artinya jarak antara hyperplane dengan objek-objek data berbeda dengan
kelas yang berdekatan (terluar) yang diberi warna ungu dan kuning. Hasil hyperplane tersebut
sudah cukup optimal yang dimana tidak ada objek yang sering disebut support vector yaitu yang
paling sulit diklasifikasikan dikarenakan posisinya yang hampir tumpang tindih (overlap) dengan
kelas lain.

### Analisis Perfomansi ( Secara Quantitative )
Untuk melihat akurasi pada model SVM saya hanya menggunakan RMSE saja, didapat hasil
sebagai berikut.
```
#QUANTITATIVE validate the predicted value with testing data using RMSE

print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
```
<img width="376" alt="image" src="https://user-images.githubusercontent.com/57959734/222881366-ce1069ae-33ad-434a-9522-4458ae6f1e61.png">


## Kesimpulan
- SVM cocok untuk dataset dengan jumlah yang kecil
- Hasil yang baik saat visualisasi dengan hyperlane dimana tidak ada objek data yang
posisinya tumpang tindih (overlap) dengan kelas lain, Dan hyperplane yang dibuat
pun sudah cukup baik yaitu posisinya ditengah-tengah antara dua kelas terpisah
secara sempurna.
- Sesuai dengan hasil RMSE yang dimana besarnya tingkat kesalahan hasil prediksi itu
bisa dibilang baik. Karena, dimana semakin kecil (mendekati 0) nilai RMSE maka
hasil prediksi akan semakin akurat.


