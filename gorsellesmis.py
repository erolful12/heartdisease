import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Veri setini yükle
veri = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')

# --- Veri Keşfi ---
print("### Veri Setine İlk Bakış ###")
print("İlk 5 Satır:\n", veri.head())
print("\nVeri Seti Özeti:\n", veri.info())
print("\nTemel İstatistikler:\n", veri.describe())
print("\nEksik Değerler:\n", veri.isnull().sum())

# --- Veri Ön İşleme ---
# Eksik verileri temizle
veri = veri.dropna()
print("\nEksik veriler temizlendi. Yeni veri boyutu:", veri.shape)

# Özellikleri (X) ve hedef değişkeni (y) ayır
X = veri.drop('Heart_Risk', axis=1)
y = veri['Heart_Risk']

# Özellikleri standart hale getir
olcekleyici = StandardScaler()
X_scaled = olcekleyici.fit_transform(X)

# Veriyi eğitim ve test setlerine ayır
X_egitim, X_test, y_egitim, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("\nEğitim seti boyutu:", X_egitim.shape)
print("Test seti boyutu:", X_test.shape)

# Model Eğitimi 
# 1. Lojistik Regresyon
lojistik_regresyon = LogisticRegression(random_state=42)
lojistik_regresyon.fit(X_egitim, y_egitim)
y_tahmin_loj = lojistik_regresyon.predict(X_test)
y_olasilik_loj = lojistik_regresyon.predict_proba(X_test)[:, 1]

# 2. Rastgele Orman
rastgele_orman = RandomForestClassifier(random_state=42)
rastgele_orman.fit(X_egitim, y_egitim)
y_tahmin_rf = rastgele_orman.predict(X_test)
y_olasilik_rf = rastgele_orman.predict_proba(X_test)[:, 1]

# 3. Destek Vektör Makinesi (SVM)
svm = SVC(probability=True, random_state=42)
svm.fit(X_egitim, y_egitim)
y_tahmin_svm = svm.predict(X_test)
y_olasilik_svm = svm.predict_proba(X_test)[:, 1]

# Model Değerlendirme 
def modeli_degerlendir(y_test, y_tahmin, y_olasilik):
    """Modelin performansını ölçen metrikleri hesaplar."""
    return {
        'dogruluk': accuracy_score(y_test, y_tahmin),
        'hassasiyet': precision_score(y_test, y_tahmin),
        'duyarlilik': recall_score(y_test, y_tahmin),
        'f1_skoru': f1_score(y_test, y_tahmin),
        'roc_auc': roc_auc_score(y_test, y_olasilik)
    }

# Modelleri değerlendir
sonuclar = {
    'Lojistik Regresyon': modeli_degerlendir(y_test, y_tahmin_loj, y_olasilik_loj),
    'Rastgele Orman': modeli_degerlendir(y_test, y_tahmin_rf, y_olasilik_rf),
    'SVM': modeli_degerlendir(y_test, y_tahmin_svm, y_olasilik_svm)
}

# Sonuçları Yazdır 
print("\n### Model Performans Sonuçları ###")
for model_adi, metrikler in sonuclar.items():
    print(f"\n**{model_adi}:**")
    print(f" - Doğruluk: {metrikler['dogruluk']:.4f}")
    print(f" - Hassasiyet: {metrikler['hassasiyet']:.4f}")
    print(f" - Duyarlılık: {metrikler['duyarlilik']:.4f}")
    print(f" - F1 Skoru: {metrikler['f1_skoru']:.4f}")
    print(f" - ROC AUC: {metrikler['roc_auc']:.4f}")

# En iyi modelleri belirle
max_dogruluk_modeli = max(sonuclar, key=lambda x: sonuclar[x]['dogruluk'])
max_roc_modeli = max(sonuclar, key=lambda x: sonuclar[x]['roc_auc'])
print("\n### Modellerin Karşılaştırması ###")
print(f" - En yüksek doğruluk: {max_dogruluk_modeli} ({sonuclar[max_dogruluk_modeli]['dogruluk']:.4f})")
print(f" - En yüksek ROC AUC: {max_roc_modeli} ({sonuclar[max_roc_modeli]['roc_auc']:.4f})")

# Görselleştirmeler 
# ROC Eğrisi
plt.figure(figsize=(8, 6))
for model_adi, y_olasilik in [('Lojistik Regresyon', y_olasilik_loj), 
                              ('Rastgele Orman', y_olasilik_rf), 
                              ('SVM', y_olasilik_svm)]:
    fpr, tpr, _ = roc_curve(y_test, y_olasilik)
    auc = sonuclar[model_adi]['roc_auc']
    plt.plot(fpr, tpr, label=f'{model_adi} (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Rastgele Tahmin')
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('ROC Eğrisi: Model Karşılaştırması')
plt.legend()
plt.show()

# Karmaşıklık Matrisi (Lojistik Regresyon)
cm_loj = confusion_matrix(y_test, y_tahmin_loj)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_loj, annot=True, fmt='d', cmap='Blues')
plt.title('Lojistik Regresyon - Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# Özellik Önem Dereceleri (Rastgele Orman)
onem_duzeyleri = rastgele_orman.feature_importances_
sirali_gostergeler = np.argsort(onem_duzeyleri)[::-1]
ozellikler = X.columns
plt.figure(figsize=(10, 6))
plt.title('Rastgele Orman: Özellik Önem Dereceleri')
plt.bar(range(X.shape[1]), onem_duzeyleri[sirali_gostergeler], align='center')
plt.xticks(range(X.shape[1]), ozellikler[sirali_gostergeler], rotation=90)
plt.xlabel('Özellikler')
plt.ylabel('Önem Derecesi')
plt.tight_layout()
plt.show()

#Sınıf Dağılımı 
print("\n### Sınıf Dağılımı ###")
print(y.value_counts(normalize=True))
if y.value_counts(normalize=True).min() < 0.3:
    print("Uyarı: Sınıf dengesizliği tespit edildi. 'class_weight=balanced' kullanılabilir.")

# --- Model ve Ölçekleyici Kaydet ---
joblib.dump(rastgele_orman, "kalp_hastaligi_model.pkl")
joblib.dump(olcekleyici, "olcekleyici.pkl")
print("\nModel ve ölçekleyici başarıyla kaydedildi.")
