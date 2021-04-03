# Gerekli kütüphaneleri import ediyoruz 
import pandas as pd # Veriyi okumak için 
import matplotlib.pyplot as plt  # Görselleştirme için 

from sklearn.svm import SVR 

# Veriyi dosyamızı okuduk
data = pd.read_csv("maaslar.csv")

# Bağımsız değişkenimiz eğitim seviyesi olacak
# Bağımlı değişkenimiz maas
x = data["Egitim Seviyesi"].values
y = data["maas"].values

# x'i iki boyutlu matris haline getirdik
x = x.reshape(-1,1)

svr_reg_poly =SVR(kernel = "poly",degree=4) # poly metodu seçildi ve derecesi 4 olarak ayarlandı
svr_reg_poly.fit(x,y) # poly model eğitimi

svr_reg_rbf =SVR(kernel = "rbf") # RBF metodu seçildi
svr_reg_rbf.fit(x,y) # rbf model eğitimi

svr_reg_linear =SVR(kernel = "linear") #Linear metod seçildi
svr_reg_rbf.fit(x,y) # linear model eğitimi 

# SVR Polinom grafik üzerinde gösterimi
plt.title("Poly") # başlık
plt.scatter(x,y,color='red') # kırmızı olan noktalar gerçek değerler, mavi olan çizgi grafiği tahminlerimiz
plt.plot(x,svr_reg_poly.predict(x),color='blue')

plt.show()

# SVR RBF grafik üzerinde gösterimi
plt.title("RBF") #başlık 
plt.scatter(x,y,color='red') # kırmızı olan noktalar gerçek değerler, mavi olan çizgi grafiği tahminlerimiz
plt.plot(x,svr_reg_rbf.predict(x),color='green')
plt.show()

# SVR Linear grafik üzerinde gösterimi
plt.title("Linear") # başlık
plt.scatter(x,y,color='red') # kırmızı olan noktalar gerçek değerler, mavi olan çizgi grafiği tahminlerimiz
plt.plot(x,svr_reg_rbf.predict(x),color='green') 
plt.show()
