# Fine-tune Llama 3.2 1B to triage GitHub issues

Teach a 1B language model to read a vLLM GitHub issue (title + body) and answer with its category:
bug, feature request, usage, installation, performance, new model or documentation.
Hugging Face transformers + PEFT (LoRA), with the training loop written in plain PyTorch. Trains on a MacBook (Apple GPU) or any CUDA GPU.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login                 # accept the Llama 3.2 licence on Hugging Face first
export GITHUB_TOKEN=...               # any token, no permissions needed
python data/fetch_issues.py           # ~2,000 labelled issues
python data/prepare.py                # remove the label leak, split by date
python evals/baseline_tfidf.py        # the baseline to beat
python explore_tokenizer.py           # tokens, padding, attention mask, chat templates
python evaluate.py                    # base model, prompt only
python train.py                       # LoRA fine-tune -> issue-triage-lora/
python evaluate.py issue-triage-lora  # fine-tuned model
```
`media/terminal/` has the exact output of every step from the run in the video (made with `capture.sh`).
The video run loaded the identical weights from the `unsloth/Llama-3.2-1B-Instruct` mirror via `MODEL_ID`.

## Results (280 newest issues, never seen in training)
| Model | Accuracy |
|---|---|
| Llama 3.2 1B, prompt only | 23.9% |
| TF-IDF + logistic regression | 61.4% |
| Llama 3.2 1B + LoRA (700 issues, 1 epoch) | 68.9% |

The LoRA adapter is in `issue-triage-lora/` (43 MB).
