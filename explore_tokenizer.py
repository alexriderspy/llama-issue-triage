"""Chapter 4 on screen: what the tokenizer, padding, attention mask and chat template actually do."""
import os
from transformers import AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(tokenizer("vLLM crashes when I load Llama"))
tokenizer.pad_token = tokenizer.eos_token
batch = tokenizer(["vLLM crashes when I load Llama", "Please add Gemma 4", "Typo in docs"],
                  padding=True, return_tensors="pt")
print(batch["input_ids"].shape)
print(batch["input_ids"])
print(tokenizer.batch_decode(batch["input_ids"]))
print(batch["attention_mask"])

messages = [{"role": "system", "content": "You are a cricket commentator. Answer everything like it's the last over."},
            {"role": "user", "content": "How do I install vLLM?"}]
print(tokenizer.apply_chat_template(messages, tokenize=False))
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
with_start = messages + [{"role": "assistant", "content": "And he"}]
print(tokenizer.apply_chat_template(with_start, tokenize=False))
print(tokenizer.apply_chat_template(with_start, tokenize=False, continue_final_message=True))
