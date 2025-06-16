from flask import Flask, render_template, request, session, redirect, url_for
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification  # type: ignore
import torch
import joblib
import numpy as np
import os
import pandas as pd
from chatbot_infer import chat_with_bot


from veritabani import VeritabaniYardimcisi

app = Flask(__name__)
app.secret_key = "chatbot_guvenlik_anahtari"
db = VeritabaniYardimcisi()

model_path = "bert_model"
csv_path = "chatbot_veriseti.csv"

model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
label_encoder = joblib.load(os.path.join(model_path, "label_encoder.joblib"))
df = pd.read_csv(csv_path, encoding="utf-8-sig")


@app.route("/hastalik", methods=["GET", "POST"])
def index():
    if "sohbet" not in session:
        session["sohbet"] = []

    if request.method == "POST":
        metin = request.form.get("mesaj", "").strip()
        if metin:
            session["sohbet"].append({"tip": "kullanici", "icerik": metin})
            try:
                inputs = tokenizer(metin, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_id = torch.argmax(logits, dim=1).item()
                    probs = torch.softmax(logits, dim=1)[0]
                    confidence = probs[predicted_id].item()
                    predicted_label = label_encoder.inverse_transform([predicted_id])[0]

                oneriler = df[df["hastalık"] == predicted_label]["öneri"].unique()
                ilk_oneri = oneriler[0] if len(oneriler) > 0 else "Henüz öneri bulunamadı."

                yanit = (
                    f"🤖 Tahmin Edilen Hastalık: **{predicted_label}** ({confidence*100:.2f}%)\n"
                    f"💡 Öneri: {ilk_oneri}"
                )

            except Exception as e:
                yanit = f"⚠️ Hata oluştu: {str(e)}"

            session["sohbet"].append({"tip": "bot", "icerik": yanit})

    return render_template("index.html", sohbet=session["sohbet"])


@app.route("/sifirla", methods=["POST"])
def sifirla():
    session.pop("sohbet", None)
    return redirect(url_for("index"))


@app.route("/kayit", methods=["GET", "POST"])
def kayit():
    if request.method == "POST":
        kullanici = request.form.get("kullanici")
        parola = request.form.get("parola")
        if kullanici and parola:
            if db.kayit_ol(kullanici, parola):
                session["kullanici"] = kullanici
                return redirect(url_for("index"))
            else:
                return "⚠️ Bu kullanıcı adı zaten alınmış."
    return render_template("kayit.html")


@app.route("/giris", methods=["GET", "POST"])
def giris():
    if request.method == "POST":
        kullanici = request.form.get("kullanici")
        parola = request.form.get("parola")
        if db.giris_yap(kullanici, parola):
            session["kullanici"] = kullanici
            return redirect(url_for("index"))
        else:
            return "⚠️ Kullanıcı adı veya parola hatalı."
    return render_template("giris.html")


@app.route("/cikis")
def cikis():
    session.pop("kullanici", None)
    return redirect(url_for("index"))


@app.route("/degerlendir", methods=["POST"])
def degerlendir():
    if "kullanici" not in session:
        return redirect(url_for("index"))

    soru = request.form.get("soru")
    cevap = request.form.get("cevap")
    durum = request.form.get("durum")

    if soru and cevap and durum in ["doğru", "yanlış"]:
        db.degerlendirme_ekle(session["kullanici"], soru, cevap, durum)

    return redirect(url_for("index"))


@app.route("/incelemeler")
def incelemeler():
    veriler = db.tum_degerlendirmeleri_getir()
    return render_template("doktorincelemeleri.html", veriler=veriler)


@app.route("/sil_degerlendirme", methods=["POST"])
def sil_degerlendirme():
    if "kullanici" not in session:
        return redirect(url_for("giris"))

    degerlendirme_id = request.form.get("id")
    if degerlendirme_id:
        db.degerlendirme_sil(degerlendirme_id, session["kullanici"])

    return redirect(url_for("incelemeler"))

@app.route("/", methods=["GET"])
def ana_sayfa():
    return redirect(url_for("secim"))

@app.route("/secim")
def secim():
    return render_template("secim.html")


@app.route("/sohbet", methods=["GET", "POST"])
def sohbet():
    if "sohbet2" not in session:
        session["sohbet2"] = []

    if request.method == "POST":
        mesaj = request.form.get("mesaj", "").strip()
        if mesaj:
            session["sohbet2"].append({"tip": "kullanici", "icerik": mesaj})
            try:
                yanit = chat_with_bot(mesaj)
            except Exception as e:
                yanit = f"⚠️ GPT-2 modeli çalıştırılırken hata oluştu: {str(e)}"
            session["sohbet2"].append({"tip": "bot", "icerik": yanit})

    return render_template("sohbet.html", sohbet=session["sohbet2"])



if __name__ == "__main__":
    app.run(debug=True)