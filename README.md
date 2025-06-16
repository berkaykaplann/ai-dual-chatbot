# 🧠 Türkçe Yapay Zeka Destekli Chatbot Web Uygulaması

Bu proje, kullanıcıların **GPT-2 tabanlı sohbet botu** veya **DistilBERT tabanlı hastalık tahmin botu** ile etkileşime geçebileceği bir Flask web uygulamasıdır. Kullanıcı giriş sistemi, değerlendirme kaydı ve temiz bir arayüzle entegre edilmiştir.

## 🚀 Özellikler

- 🗣️ GPT-2 tabanlı serbest Türkçe sohbet botu
- 🏥 DistilBERT ile belirtiye dayalı hastalık tahmini
- 👤 Kullanıcı girişi ve kayıt sistemi (SQLite)
- ✅ Tahmin doğruluğu kullanıcıya sorulur ve kayıt altına alınır
- 📊 Değerlendirme verileri analiz için veritabanına yazılır
- 🌐 Flask ile web tabanlı arayüz

## 📁 Proje Yapısı

proje_klasoru/
│
├── app.py # Flask ana uygulama
├── veritabani.py # Veritabanı yardımcı sınıfı
├── chatbot_infer.py # GPT-2 sohbet motoru
├── bert_tahmin.py # Hastalık tahmini fonksiyonu
├── train_model.py # GPT-2 eğitim dosyası
├── x.py # DistilBERT eğitim dosyası
├── templates/ # HTML dosyaları
├── static/ # CSS, görsel vs.
├── veritabani.db (ignore edilir)# Veritabanı (otomatik oluşur)
└── README.md # Bu dosya



## ⚙️ Kurulum

1. Depoyu klonla:

```bash
git clone https://github.com/kullanici_adin/turkish-chatbot-app.git
cd turkish-chatbot-app


Ortamı oluştur ve bağımlılıkları yükle:

python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
pip install -r requirements.txt


Uygulamayı başlat:
python app.py



🤖 Model Kullanımı
GPT-2 sohbet modeli HuggingFace’den otomatik olarak indirilecektir:
dbmdz/gpt2-turkish

Hastalık tahmin modeli için istersen chatbot_veriseti.csv verisiyle yeniden eğitip bert_model/ klasörünü oluşturabilirsin.

📦 Notlar
gpt2-turkish-chatbot/ ve bert_model/ klasörleri çok yer kapladığı için GitHub’da yer almamaktadır.

Uygulama çalıştığında bu modeller yüklenir ya da yeniden eğitilerek oluşturulur.



