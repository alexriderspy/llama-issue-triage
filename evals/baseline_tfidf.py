"""The old-school baseline: word counts plus logistic regression. Fine-tuning has to beat this."""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

load = lambda name: [json.loads(l) for l in open(f"data/{name}.jsonl")]
train, test = load("train"), load("test")
text = lambda rows: [r["title"] + " " + r["body"] for r in rows]

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
clf = LogisticRegression(max_iter=2000, C=5)
clf.fit(vec.fit_transform(text(train)), [r["label"] for r in train])
pred = clf.predict(vec.transform(text(test)))
print(f"TF-IDF + logistic regression accuracy: {accuracy_score([r['label'] for r in test], pred):.1%}")
