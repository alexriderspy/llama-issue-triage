"""Clean the issues so the label can't be read off the template, then split by date."""
import json, re
from collections import defaultdict

NAMES = {"bug": "bug", "feature request": "feature request", "usage": "usage", "installation": "installation",
         "performance": "performance", "new-model": "new model", "documentation": "documentation"}

def clean_title(title: str) -> str:
    # "[Bug]: ..." or "[Usage] ..." gives the answer away, so it goes
    return re.sub(r"^\s*(\[[^\]]*\]\s*:?\s*)+", "", title).strip()

def clean_body(body: str) -> str:
    body = re.sub(r"```.*?```", " ", body, flags=re.S)          # code blocks and environment dumps
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)         # hidden template comments
    body = re.sub(r"<details>.*?</details>", " ", body, flags=re.S)  # collapsed environment info
    lines = [l for l in body.splitlines()
             if not l.lstrip().startswith("#")                  # template headings like "### 🐛 Describe the bug"
             and not re.match(r"\s*- \[[ xX]\]", l)]            # the "I have searched existing issues" checkboxes
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    return text[:2000]

if __name__ == "__main__":
    rows = [json.loads(l) for l in open("data/issues_raw.jsonl")]
    seen, by_label = set(), defaultdict(list)
    for r in rows:
        title, body = clean_title(r["title"]), clean_body(r["body"])
        if r["number"] in seen or len(title) < 10:
            continue
        seen.add(r["number"])
        by_label[r["label"]].append({"title": title, "body": body, "label": NAMES[r["label"]], "created": r["created"]})

    # the newest issues of each label are the test set, like issues that haven't been written yet
    splits = {"train": [], "val": [], "test": []}
    for label, items in by_label.items():
        items.sort(key=lambda x: x["created"], reverse=True)
        splits["test"] += items[:40]
        splits["val"] += items[40:60]
        splits["train"] += items[60:160]   # 100 per label keeps training short on a laptop
    for name, items in splits.items():
        with open(f"data/{name}.jsonl", "w") as f:
            for x in items:
                f.write(json.dumps(x) + "\n")
        print(name, len(items))
