# Gerekli kütüphaneleri import ediyoruz 
import pandas as pd # Veriyi okumak için 
import matplotlib.pyplot as plt  # Görselleştirme için 

from sklearn.tree import DecisionTreeRegressor

# Veriyi dosyamızı okuduk
data = pd.read_csv("satislar.csv")

# Bağımsız değişkenimiz Aylar olacak
# Bağımlı değişkenimiz Satislar
x = data["Aylar"].values
y = data["Satislar"].values

# x'i iki boyutlu matris haline getirdik
x = x.reshape(-1,1)

# DecisionTreeRegressor dan nesne ürettik 
dtc = DecisionTreeRegressor()
dtc.fit(x,y) # Eğitim kümemizi eğittik 

# Modelimize tahmin etmesi için tekrardan eğitim kümemizi verdik 
y_pred = dtc.predict(x)

# Burada görsel olarak daha rahat karşılaştırabilelim diye 
# pandas dataframe ine çevirildi gerçek y değeri ve tahmin edilen y değeri
real_y = pd.DataFrame(y,columns = ["Gerçek Y"],index = range(0,len(y))) # DataFrame oluşturduk Gerçek Y değerlerinden 
prediction_y = pd.DataFrame(y_pred,columns = ["Tahmin Y"],index =range(0,len(y_pred))) # Tahmin Y DataFrame
result_data = pd.concat([real_y,prediction_y],axis =1) # iki dataframe'i birleştirdik 
result_data.iloc[:10] # İlk 10 tane değeri gösterdik 

# Test ve gerçek y değerlerimizi grafik üzerinde karşılaştırdık
plt.figure(figsize=(8,6))
plt.title("Karar Ağacı Test Ve Gerçek Değer Karşılaştırma")
plt.scatter(y,y_pred) # Scatter plot kullandık yani noktalar ne kadar yakın ise o kadar iyi. 
