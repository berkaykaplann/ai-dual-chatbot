from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import joblib
import torch

model_name = "dbmdz/distilbert-base-turkish-cased"
csv_path = "chatbot_veriseti.csv"
save_path = "bert_model"

# 1. Veriyi oku
df = pd.read_csv(csv_path, encoding="utf-8-sig")
df["hastalık"] = df["hastalık"].str.strip()
df["belirtiler"] = df["belirtiler"].str.strip()

# 2. Etiket kodla
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["hastalık"])

# 3. Eğitim/veri ayır (stratify ile dengeli dağıtım)
train_df, temp_df = train_test_split(df, test_size=0.35, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[["belirtiler", "label"]]),
    "validation": Dataset.from_pandas(val_df[["belirtiler", "label"]]),
    "test": Dataset.from_pandas(test_df[["belirtiler", "label"]])
})

# 4. Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["belirtiler"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["belirtiler"])
dataset.set_format("torch")

# 5. Model
num_labels = len(label_encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 6. Metrikler
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# 7. Eğitim ayarları (artırılmış epoch)
training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    logging_dir=os.path.join(save_path, "logs"),
    logging_steps=10,
    max_grad_norm=1.0  

)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics
)

# 9. Eğit ve kaydet
trainer.train()
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
joblib.dump(label_encoder, os.path.join(save_path, "label_encoder.joblib"))

# 10. Test metrikleri
test_result = trainer.evaluate(eval_dataset=dataset["test"])
print("\n📊 Test Sonuçları:")
for k, v in test_result.items():
    if k == "eval_accuracy":
        print(f"{k}: {v * 100:.2f}%")
    else:
        print(f"{k}: {v:.4f}")

# 11. Örnek belirti ile tahmin
print("\n🔍 Örnek Belirti ile Tahmin:")
ornek_belirti = "tırnağım şişmiş ve halsizim"

inputs = tokenizer(ornek_belirti, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
with torch.no_grad():
    outputs = model(**inputs)
    predicted_label_id = torch.argmax(outputs.logits, dim=1).item()
predicted_disease = label_encoder.inverse_transform([predicted_label_id])[0]

print(f"Belirti: {ornek_belirti}")
print(f"Tahmin Edilen Hastalık: {predicted_disease}")