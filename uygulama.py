import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd

# —————— Model ve Ölçekleyici Yükle ——————
model = joblib.load("kalp_hastaligi_model.pkl")
scaler = joblib.load("olcekleyici.pkl")

# —————— Haritalamalar ——————
binary_map = {"Hayır": 0.0, "Evet": 1.0}
gender_map = {"Kadın": 0.0, "Erkek": 1.0}

# —————— Türkçe Etiketlerle Özellik Sıralaması ——————
features = [
    ("Chest_Pain", "Göğüs Ağrısı"),
    ("Shortness_of_Breath", "Nefes Darlığı"),
    ("Fatigue", "Yorgunluk"),
    ("Palpitations", "Çarpıntı"),
    ("Dizziness", "Baş Dönmesi"),
    ("Swelling", "Ayaklarda Şişlik"),
    ("Pain_Arms_Jaw_Back", "Kol, Çene veya Sırtta Ağrı"),
    ("Cold_Sweats_Nausea", "Soğuk Terleme / Bulantı"),
    ("High_BP", "Yüksek Tansiyon"),
    ("High_Cholesterol", "Yüksek Kolesterol"),
    ("Diabetes", "Diyabet"),
    ("Smoking", "Sigara Kullanımı"),
    ("Obesity", "Obezite"),
    ("Sedentary_Lifestyle", "Hareketsiz Yaşam"),
    ("Family_History", "Ailede Kalp Hastalığı"),
    ("Chronic_Stress", "Kronik Stres"),
    ("Gender", "Cinsiyet"),
    ("Age", "Yaş (yıl)")
]

# —————— GUI Oluştur ——————
root = tk.Tk()
root.title("Kalp Hastalığı Risk Tahmini")
root.geometry("500x800")
root.resizable(False, False)

ttk.Label(root, text="Lütfen Semptom Bilgilerinizi Girin:", font=("Arial", 16)).pack(pady=10)

entries = {}

# Numeric giriş
def add_numeric(field, label):
    frame = ttk.Frame(root)
    frame.pack(fill="x", padx=20, pady=5)
    ttk.Label(frame, text=label, width=30).pack(side="left")
    ent = ttk.Entry(frame)
    ent.pack(side="right", fill="x", expand=True)
    entries[field] = ent

# Combobox giriş
def add_combo(field, label, options):
    frame = ttk.Frame(root)
    frame.pack(fill="x", padx=20, pady=5)
    ttk.Label(frame, text=label, width=30).pack(side="left")
    cb = ttk.Combobox(frame, values=options, state="readonly")
    cb.current(0)
    cb.pack(side="right", fill="x", expand=True)
    entries[field] = cb

# Formu oluştur
for f, label in features:
    if f == "Age":
        add_numeric(f, label)
    elif f == "Gender":
        add_combo(f, label, ["Kadın", "Erkek"])
    else:
        add_combo(f, label, ["Hayır", "Evet"])

# Tahmin fonksiyonu
def tahmin_yap():
    try:
        vals = []
        for f, _ in features:
            widget = entries[f]
            if f == "Age":
                vals.append(float(widget.get()))
            elif f == "Gender":
                vals.append(gender_map[widget.get()])
            else:
                vals.append(binary_map[widget.get()])
    except Exception:
        messagebox.showerror("Hata", "Lütfen tüm alanları eksiksiz ve doğru formatta doldurun.")
        return

    X_df = pd.DataFrame([vals], columns=[f for f, _ in features])
    X_scaled = scaler.transform(X_df)
    prob = model.predict_proba(X_scaled)[0][1]

    if prob >= 0.8:
        msg = (
            "❗ Yüksek Risk Tespit Edildi!\n\n"
            "Kalp hastalığı riski çok yüksek görünüyor.\n"
            "Lütfen acilen bir kardiyoloji uzmanına başvurun.\n\n"
            "Öneriler:\n"
            "- En kısa sürede tam kalp kontrolü yaptırın\n"
            "- Fiziksel aktivitenizi artırın\n"
            "- Tansiyon, kolesterol ve şeker düzeylerinizi kontrol ettirin\n\n"
            "Risk oranı: {:.0f}%"
        ).format(prob * 100)
    elif prob >= 0.5:
        msg = (
            "⚠️ Orta-Yüksek Risk!\n\n"
            "Kalp hastalığı riski mevcut.\n"
            "En kısa sürede doktor kontrolü önerilir.\n\n"
            "Öneriler:\n"
            "- Düzenli tansiyon ölçümü\n"
            "- Sağlıklı beslenme ve egzersiz\n"
            "- Sigara içiyorsanız bırakmayı düşünün\n\n"
            "Risk oranı: {:.0f}%"
        ).format(prob * 100)
    elif prob >= 0.2:
        msg = (
            "🔎 Düşük-Orta Risk\n\n"
            "Şu anda ciddi bir risk yok, ancak dikkatli olmalısınız.\n"
            "Sağlıklı yaşam tarzını sürdürün.\n\n"
            "Öneriler:\n"
            "- Haftada en az 150 dakika yürüyüş\n"
            "- Tuz ve işlenmiş gıdalardan kaçınma\n"
            "- Stres yönetimi alışkanlıkları\n\n"
            "Risk oranı: {:.0f}%"
        ).format(prob * 100)
    else:
        msg = (
            "✅ Düşük Risk\n\n"
            "Şu anda endişe edilecek bir durum görünmüyor.\n"
            "Yine de sağlıklı alışkanlıkları sürdürmeniz önemli.\n\n"
            "Öneriler:\n"
            "- Yılda bir rutin kalp kontrolü\n"
            "- Aktif kalmaya devam edin\n"
            "- Dengeli beslenmeye özen gösterin\n\n"
            "Risk oranı: {:.0f}%"
        ).format(prob * 100)

    messagebox.showinfo("Risk Tahmini Sonucu", msg)

# Buton
ttk.Button(root, text="Tahmin Et", command=tahmin_yap).pack(pady=20)

root.mainloop()
