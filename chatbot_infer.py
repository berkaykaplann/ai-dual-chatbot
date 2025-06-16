# chatbot_infer.py

from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel

MODEL_PATH = "./gpt2-turkish-chatbot"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def chat_with_bot(user_input):
    prompt = f"Sen: {user_input}\nBot:"
    result = generator(prompt, max_length=100, pad_token_id=tokenizer.eos_token_id)
    return result[0]['generated_text'].split("Bot:")[1].strip()

if __name__ == "__main__":
    print("Sohbeti başlat! (Çıkmak için 'çık', 'exit' veya 'quit' yaz)\n")
    while True:
        user_input = input("Sen: ")
        if user_input.lower() in ["çık", "exit", "quit"]:
            print("Sohbet sona erdi.")
            break
        try:
            response = chat_with_bot(user_input)
            print("Bot:", response)
        except Exception as e:
            print("Bir hata oluştu:", e)
