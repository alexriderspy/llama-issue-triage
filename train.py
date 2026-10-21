"""Fine-tune Llama 3.2 1B Instruct with LoRA to label vLLM GitHub issues, in a plain PyTorch loop."""
import json, math, os, random
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from prompt import build_prompt

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
EPOCHS, BATCH, ACCUM, LR = 1, 2, 8, 2e-4
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32 if device == "cpu" else torch.bfloat16   # half the memory of float32
torch.manual_seed(0); random.seed(0)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

def encode(row):
    prompt_ids = tokenizer(build_prompt(tokenizer, row["title"], row["body"]), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["label"] + "<|eot_id|>", add_special_tokens=False)["input_ids"]
    # -100 tells the loss to ignore a position: we only grade the model on the answer
    return {"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids}

def collate(batch):
    # round up to a multiple of 64: GPUs (Apple's especially) are much faster when they see the same shapes again
    width = -(-max(len(x["input_ids"]) for x in batch) // 64) * 64
    pad = lambda seq, value: seq + [value] * (width - len(seq))
    return {
        "input_ids": torch.tensor([pad(x["input_ids"], tokenizer.pad_token_id) for x in batch]),
        "attention_mask": torch.tensor([pad([1] * len(x["input_ids"]), 0) for x in batch]),
        "labels": torch.tensor([pad(x["labels"], -100) for x in batch]),
    }

load = lambda name: [encode(json.loads(l)) for l in open(f"data/{name}.jsonl")]
train, val = load("train"), load("val")
train_loader = DataLoader(train, batch_size=BATCH, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val, batch_size=BATCH, collate_fn=collate)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
total_steps = EPOCHS * math.ceil(len(train_loader) / ACCUM)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

def val_loss():
    model.eval(); losses = []
    with torch.no_grad():
        for batch in val_loader:
            losses.append(model(**{k: v.to(device) for k, v in batch.items()}).loss.item())
    model.train()
    return sum(losses) / len(losses)

print("validation loss before training:", round(val_loss(), 3))
history, step = [], 0
model.train()
for epoch in range(EPOCHS):
    for i, batch in enumerate(train_loader):
        loss = model(**{k: v.to(device) for k, v in batch.items()}).loss / ACCUM
        loss.backward()
        if (i + 1) % ACCUM == 0 or i + 1 == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step += 1
            history.append({"step": step, "train_loss": loss.item() * ACCUM})
            if step % 5 == 0:
                print(f"epoch {epoch + 1} step {step}/{total_steps} loss {loss.item() * ACCUM:.3f}")
    v = val_loss()
    history.append({"step": step, "val_loss": v})
    print(f"epoch {epoch + 1} done, validation loss {v:.3f}")

model.save_pretrained("issue-triage-lora")
tokenizer.save_pretrained("issue-triage-lora")
json.dump(history, open("issue-triage-lora/history.json", "w"))
