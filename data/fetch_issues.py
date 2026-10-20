"""Download labelled vLLM GitHub issues: the title, the body and the label a maintainer gave it."""
import json, os, time
import requests

REPO = "vllm-project/vllm"
LABELS = ["bug", "feature request", "usage", "installation", "performance", "new-model", "documentation"]
PER_LABEL = 300
HEADERS = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}

def search(label: str, page: int) -> list[dict]:
    q = f'repo:{REPO} is:issue label:"{label}"'
    r = requests.get("https://api.github.com/search/issues", headers=HEADERS,
                     params={"q": q, "sort": "created", "order": "desc", "per_page": 100, "page": page})
    r.raise_for_status()
    return r.json()["items"]

if __name__ == "__main__":
    rows = []
    for label in LABELS:
        for page in range(1, PER_LABEL // 100 + 1):
            for issue in search(label, page):
                ours = [l["name"] for l in issue["labels"] if l["name"] in LABELS]
                if len(ours) == 1:  # skip issues with two of our labels, they're ambiguous
                    rows.append({"number": issue["number"], "title": issue["title"],
                                 "body": issue["body"] or "", "label": label,
                                 "created": issue["created_at"]})
            time.sleep(2)  # the search API allows 30 requests a minute
        print(label, sum(r["label"] == label for r in rows))
    with open("data/issues_raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("total", len(rows))
