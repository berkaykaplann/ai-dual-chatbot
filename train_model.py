# train_model.py

from transformers import GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from data_preparation import prepare_data, MODEL_NAME

def train_model():
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    dataset, tokenizer = prepare_data()

    # 🔧 padding token tanımla
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./gpt2-turkish-chatbot",
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("./gpt2-turkish-chatbot")
    tokenizer.save_pretrained("./gpt2-turkish-chatbot")

if __name__ == "__main__":
    train_model()
