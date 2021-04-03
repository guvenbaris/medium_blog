# Gerekli kütüphaneleri import ediyoruz
import pandas as pd # Veriyi okumak için 
import matplotlib.pyplot as plt  # Görselleştirme için 

from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 

# Veriyi dosyamızı okuduk
data = pd.read_csv("maaslar.csv")

# Bağımsız değişkenimiz eğitim seviyesi olacak
# Bağımlı değişkenimiz maas
x = data["Egitim Seviyesi"].values
y = data["maas"].values

# x'i iki boyutlu matris haline getirdik
x = x.reshape(-1,1)

# PolynominalFeatures'dan nesne ürettik degree 4 olarak belirledik. 
poly_reg = PolynomialFeatures(degree = 4)
# değişkenimizi eğitip polynominal forma dönüştürüyoruz.
x_poly = poly_reg.fit_transform(x) 

lin_reg = LinearRegression() # LinearRegression dan nesne üretiyoruz
lin_reg.fit(x_poly,y)  # Eğitim kümemizi eğittik 

# Modelimizi tahmin etmesi için tekrar dönüştürülmüş değerlerimizi veriyoruz
y_pred = lin_reg.predict(x_poly) 

# Modelimizin başarısını ölçmek için grafik ile görselleştiriyoruz 
plt.figure(figsize=(8,6))
plt.scatter(x,y,color = "red",marker='o')
plt.plot(x,y_pred,linestyle = "--",color ='blue')
