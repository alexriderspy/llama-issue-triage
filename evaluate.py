"""Ask the model to label every test issue and compare with the maintainers' labels."""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from prompt import CATEGORIES, build_prompt

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
adapter = sys.argv[1] if len(sys.argv) > 1 else None
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32 if device == "cpu" else torch.bfloat16   # half the memory of float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
if adapter:
    model = PeftModel.from_pretrained(model, adapter)
model.eval()

def parse(text: str) -> str:
    # be generous: "**Bug**", "Category: bug" and "bug, because..." all count as bug
    text = text.lower().replace("*", "")
    found = [(text.find(c), c) for c in CATEGORIES if c in text]
    return min(found)[1] if found else "unparseable"

test = [json.loads(l) for l in open("data/test.jsonl")]
preds, raw = [], []
for row in test:
    ids = tokenizer(build_prompt(tokenizer, row["title"], row["body"]), add_special_tokens=False,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    raw.append(text); preds.append(parse(text))

gold = [r["label"] for r in test]
name = "finetuned" if adapter else "base"
print(f"{name} accuracy: {accuracy_score(gold, preds):.1%}   unparseable answers: {preds.count('unparseable')}")
print(classification_report(gold, preds, labels=CATEGORIES, zero_division=0))
print(confusion_matrix(gold, preds, labels=CATEGORIES))
json.dump([{"gold": g, "pred": p, "raw": r, "title": t["title"]} for g, p, r, t in zip(gold, preds, raw, test)],
          open(f"evals/{name}_predictions.json", "w"), indent=1)
