import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1) Veri İncelemesi
df = pd.read_csv("heart_disease_risk_dataset_earlymed.csv")
print("❯ İlk 5 satır:\n", df.head())
print("\n❯ Veri tipleri:\n", df.dtypes)
print("\n❯ Eksik değer sayıları:\n", df.isnull().sum())

# 2) Eksik varsa temizle
df = df.dropna()
print(f"\n❯ Temizlendikten sonra satır sayısı: {df.shape[0]}")

# 3) Özellik ve hedef
#    CSV’de 18 özellik var:
#    ['Chest_Pain','Shortness_of_Breath','Fatigue','Palpitations','Dizziness',
#     'Swelling','Pain_Arms_Jaw_Back','Cold_Sweats_Nausea','High_BP',
#     'High_Cholesterol','Diabetes','Smoking','Obesity','Sedentary_Lifestyle',
#     'Family_History','Chronic_Stress','Gender','Age']
X = df.drop("Heart_Risk", axis=1)
y = df["Heart_Risk"]

# 4) Ölçekleyici ile normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5) Eğitim / test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42)

# 6) Modeli eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7) Kaydet
joblib.dump(model, "kalp_hastaligi_model.pkl")
joblib.dump(scaler, "olcekleyici.pkl")
print("\n✅ Model ve ölçekleyici kaydedildi.")
