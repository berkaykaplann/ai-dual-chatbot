# data_preparation.py

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "ytu-ce-cosmos/turkish-gpt2"

def prepare_data(csv_path="turkish_polite_chat.csv", max_len=128):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Padding token'ı tanımla
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("csv", data_files={"train": csv_path}, delimiter=",")

    def combine_dialogue(example):
        return {"text": f"Sen: {example['user']}\nBot: {example['chat']}"}


    dataset = dataset["train"].map(combine_dialogue)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_len)

    tokenized_dataset = dataset.map(tokenize, batched=True)
    return tokenized_dataset, tokenizer
