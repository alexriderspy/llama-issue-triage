# On-screen frames, in script order

| Frame | Chapter | Goes under the line starting |
|---|---|---|
| `03_01_terminal_setup.png` | Setup | the four setup commands (typed, not a run) |
| `04_01_vscode_fetch_issues.png` | Getting the data from GitHub | Now the data. Here's fetch_issues.py. At the top we have the repo, which is vllm-project/v… |
| `04_02_terminal.png` | Getting the data from GitHub | Let's run it. And there we go: about 300 for each label, a bit fewer for performance, new … |
| `05_01_vscode_prepare.png` | Cleaning the data, and the leak | So here's prepare.py… |
| `05_02_vscode_prepare_split.png` | Cleaning the data, and the leak | And then the split… |
| `05_03_terminal.png` | Cleaning the data, and the leak | Run it: 700 to train on… |
| `05_01_vscode_prepare.png` | Cleaning the data, and the leak | So here's prepare.py. clean_title removes anything in square brackets at the start of the … |
| `06_01_vscode_baseline_tfidf.png` | A baseline to beat | Before touching Llama, let's see what a simple model gets, so we know what fine-tuning has… |
| `06_02_terminal.png` | A baseline to beat | ```python
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
clf = Log… |
| `07_01_vscode_explore_tokenizer.png` | Tokenizers, padding and the attention mask | So let's start with the tokenizer. A model can't read text, it only understands numbers, a… |
| `07_02_terminal.png` | Tokenizers, padding and the attention mask | ```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(tokenizer("vLLM crashe… |
| `07_03_vscode_explore_tokenizer.png` | Tokenizers, padding and the attention mask | Now, when we train, we don't send one sentence at a time, we send a batch, and a batch is … |
| `07_04_terminal.png` | Tokenizers, padding and the attention mask | ```python
tokenizer.pad_token = tokenizer.eos_token
batch = tokenizer(["vLLM crashes when … |
| `08_01_vscode_train.png` | Loading the model: what a causal LM is | Now the model. If we load it and print its type, it's an AutoModelForCausalLM, and it's wo… |
| `09_01_vscode_explore_tokenizer.png` | Chat templates | The next thing I'm going to talk about is chat templates. Language models go through somet… |
| `09_02_terminal.png` | Chat templates | If we pass this to tokenizer.apply_chat_template, we can see the exact text the model sees… |
| `09_03_vscode_explore_tokenizer.png` | Chat templates | Now, if we set add_generation_prompt to True, it adds one more thing at the end: start_hea… |
| `09_04_terminal.png` | Chat templates | ```python
with_start = messages + [{"role": "assistant", "content": "And he"}]
print(token… |
| `10_01_vscode_prompt.png` | The prompt, and how the base model does | Here's prompt.py, which builds the prompt for one issue. I keep it in its own file because… |
| `10_02_vscode_evaluate.png` | The prompt, and how the base model does | Then evaluate.py. For every issue in the test set, we build the prompt, call model.generat… |
| `10_03_terminal.png` | The prompt, and how the base model does | So let's run the base model, no training at all, just the prompt. And look at this. 23.9 p… |
| `11_02_vscode_train.png` | LoRA on the whiteboard | Then in code, LoraConfig. r is 16. lora_alpha is 32, and alpha divided by r is a scaling f… |
| `11_03_terminal.png` | LoRA on the whiteboard | ```python
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",… |
| `12_01_vscode_train.png` | The training loop in PyTorch | Now the training loop. I'm writing this in plain PyTorch, because when you write the loop … |
| `12_02_vscode_train.png` | The training loop in PyTorch | Then collate, which turns a list of examples into one batch. It finds the longest example … |
| `12_03_vscode_train.png` | The training loop in PyTorch | Next, the optimizer and the schedule. AdamW is the standard optimizer for transformers, an… |
| `12_04_vscode_train.png` | The training loop in PyTorch | And the loop itself. For each batch, we move it to the GPU and call the model with the inp… |
| `13_01_terminal.png` | Training it | So let's train it. First it prints the trainable parameters, then the validation loss befo… |
| `14_01_terminal.png` | How good is it now | Now the moment of truth. Same evaluate.py, same 280 newest issues, but this time we load t… |
