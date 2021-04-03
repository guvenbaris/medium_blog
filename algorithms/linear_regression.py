#1.Kısım
# Gerekli kütüphaneleri import ediyoruz.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 

#2.Kısım
#Rakamlar aralarında linear bir ilişki olacak şekilde verilmiştir.
x = np.array([0,2,4,6,8,10,12,14,16,18,20,20]).reshape(-1,1)
y = np.array([1,3,7,3,12,11,16,10,20,15,13,15]).reshape(-1,1)

#3.Kısım
# Verilerimizi eğitim ve test olmak üzere ayırıyoruz. Böylece model başarısını test edebiliriz.
# test_size, test etmek için verinin ne kadarını ayırmak istediğimiz değer
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3) 

#4.Kısım
lnr_reg = LinearRegression() # LinearRegression modelimizi değişkene atıyoruz
lnr_reg.fit(x_train,y_train) # Verilerimizi eğitiriyoruz 

#5.Kısım
# test için ayırdığımız değişkenleri tahmin ediyoruz 
# x_test'i tahmin et diyoruz
y_pred = lnr_reg.predict(x_test)

#6.Kısım
#Tahmin değerleri ve Gerçek değerlerimizi karşılaştırıyoruz.
# Bunun için farklı işlemler yapılması gerekiyor aslında. RMSE,MSE,MAE gibi değerlere bakılması gerekiyor.
print("Tahmin Değerleri:\n",y_pred.reshape(1,-1))
print("Gerçek Değerler:\n",y_test.reshape(1,-1))

# Print Sonuçları:
# Tahmin Değerleri: [[18.66353678 10.78403756 13.93583725  2.90453834]]
# Gerçek Değerler:  [[13 11 10  1]]
