"""
POS Tagger Evaluation on Brown Corpus
So sánh hai bộ POS tagger trên Brown corpus:
  - Tagger 1: BigramTagger (với UnigramTagger backoff) - huấn luyện trên Brown
  - Tagger 2: Averaged Perceptron Tagger (pre-trained, NLTK)
"""

import sys
import random
import nltk

sys.stdout.reconfigure(encoding="utf-8")
from nltk.corpus import brown
from nltk.tag import UnigramTagger, BigramTagger, DefaultTagger
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd

# ─────────────────────────────────────────────
# 0. Tải các tài nguyên cần thiết
# ─────────────────────────────────────────────
for resource in [
    "brown", "universal_tagset",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
]:
    nltk.download(resource, quiet=True)


# ─────────────────────────────────────────────
# 1. Tải Brown corpus (universal tagset)
# ─────────────────────────────────────────────
print("=" * 60)
print("1. LOADING BROWN CORPUS")
print("=" * 60)

tagged_sents = list(brown.tagged_sents(tagset="universal"))

random.seed(42)
random.shuffle(tagged_sents)

split = int(0.8 * len(tagged_sents))
train_sents = tagged_sents[:split]
test_sents  = tagged_sents[split:]

print(f"  Total sentences : {len(tagged_sents):,}")
print(f"  Train sentences : {len(train_sents):,}  ({len(train_sents)/len(tagged_sents)*100:.0f}%)")
print(f"  Test  sentences : {len(test_sents):,}  ({len(test_sents)/len(tagged_sents)*100:.0f}%)")

total_tokens = sum(len(s) for s in tagged_sents)
test_tokens  = sum(len(s) for s in test_sents)
print(f"  Total tokens    : {total_tokens:,}")
print(f"  Test  tokens    : {test_tokens:,}")

# Nhãn vàng (gold labels) từ tập test
gold_labels = [tag for sent in test_sents for _, tag in sent]
test_word_lists = [[word for word, _ in sent] for sent in test_sents]

tag_classes = sorted(set(gold_labels))
print(f"\n  Universal POS tags ({len(tag_classes)}): {tag_classes}")


# ─────────────────────────────────────────────
# 2. Huấn luyện / chuẩn bị hai bộ POS tagger
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. SETTING UP POS TAGGERS")
print("=" * 60)

# ── Tagger 1: BigramTagger + UnigramTagger backoff ──────────────
print("\n  [Tagger 1] Training BigramTagger with UnigramTagger backoff...")
default_tagger  = DefaultTagger("NOUN")          # fallback cuối cùng
unigram_tagger  = UnigramTagger(train_sents, backoff=default_tagger)
bigram_tagger   = BigramTagger(train_sents, backoff=unigram_tagger)
print("  → Done.")

# ── Tagger 2: Averaged Perceptron Tagger (NLTK pre-trained) ─────
print("\n  [Tagger 2] Averaged Perceptron Tagger (pre-trained on Penn Treebank)")
print("             Output mapped to Universal tagset via nltk.pos_tag(..., tagset='universal')")
print("  → Ready.")


# ─────────────────────────────────────────────
# 3. Gán nhãn trên tập test
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. TAGGING TEST SET")
print("=" * 60)

# Tagger 1
print("\n  Running Tagger 1 on test set...")
bigram_preds = []
for words in test_word_lists:
    tagged = bigram_tagger.tag(words)
    bigram_preds.extend(t if t is not None else "NOUN" for _, t in tagged)
print(f"  → {len(bigram_preds):,} tokens tagged.")

# Tagger 2
print("\n  Running Tagger 2 on test set...")
perceptron_preds = []
for words in test_word_lists:
    tagged = nltk.pos_tag(words, tagset="universal")
    perceptron_preds.extend(t for _, t in tagged)
print(f"  → {len(perceptron_preds):,} tokens tagged.")


# ─────────────────────────────────────────────
# 4. Đánh giá: Precision, Recall, Macro-F1
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. EVALUATION RESULTS")
print("=" * 60)

all_labels = sorted(set(gold_labels) | set(bigram_preds) | set(perceptron_preds))

def evaluate(name, gold, pred):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(classification_report(
        gold, pred,
        labels=all_labels,
        zero_division=0,
        digits=4
    ))

evaluate("Tagger 1 : BigramTagger  (trained on Brown corpus)", gold_labels, bigram_preds)
evaluate("Tagger 2 : Averaged Perceptron Tagger (pre-trained)", gold_labels, perceptron_preds)


# ─────────────────────────────────────────────
# 5. Bảng so sánh tổng hợp
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. SUMMARY COMPARISON TABLE")
print("=" * 60)

def macro_scores(gold, pred):
    p, r, f, _ = precision_recall_fscore_support(
        gold, pred, average="macro", zero_division=0
    )
    correct = sum(g == p_ for g, p_ in zip(gold, pred))
    acc = correct / len(gold)
    return acc, p, r, f

acc1, p1, r1, f1 = macro_scores(gold_labels, bigram_preds)
acc2, p2, r2, f2 = macro_scores(gold_labels, perceptron_preds)

rows = [
    ["BigramTagger (trained)", f"{acc1:.4f}", f"{p1:.4f}", f"{r1:.4f}", f"{f1:.4f}"],
    ["Perceptron (pre-trained)", f"{acc2:.4f}", f"{p2:.4f}", f"{r2:.4f}", f"{f2:.4f}"],
]
df = pd.DataFrame(rows, columns=["Tagger", "Accuracy", "Macro-Precision", "Macro-Recall", "Macro-F1"])
df.index += 1
print(df.to_string(index=True))

# Kết luận
print("\n" + "=" * 60)
print("6. CONCLUSION")
print("=" * 60)
winner = "BigramTagger" if f1 >= f2 else "Averaged Perceptron Tagger"
better_f1 = max(f1, f2)
print(f"\n  Best Macro-F1: {better_f1:.4f}  →  {winner}")
print(f"\n  • BigramTagger   được huấn luyện trực tiếp trên Brown corpus")
print(f"    nên phù hợp với phân phối dữ liệu của corpus này.")
print(f"  • Averaged Perceptron sử dụng mô hình pre-trained trên Penn Treebank,")
print(f"    có thể có độ lệch tagset dù đã map về Universal tags.")
