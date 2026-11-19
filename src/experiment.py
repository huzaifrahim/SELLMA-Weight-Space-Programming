import json
import numpy as np
from pathlib import Path

SEED = 42
CONTEXT = 3
EPOCHS = 300
LR = 0.15
DIACRITICS = set("åäö")

np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "corpus.json"
OUT = ROOT / "experiments"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. Load data and split by sentence
# --------------------------------------------------

with open(DATA, encoding="utf-8") as f:
    corpus = json.load(f)

splits = {}

for language, sentences in corpus.items():
    sentences = list(sentences)
    rng = np.random.default_rng(SEED)
    rng.shuffle(sentences)

    if len(sentences) < 5:
        raise ValueError("At least five sentences per language are required.")

    n = len(sentences)
    n_train = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n))

    splits[language] = {
        "train": sentences[:n_train],
        "validation": sentences[n_train:n_train+n_val],
        "test": sentences[n_train+n_val:]
    }

# Shared character vocabulary
all_text = "".join(
    sentence.lower()
    for sentences in corpus.values()
    for sentence in sentences
)
vocab = sorted(set(all_text + "^$"))
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
V = len(vocab)

def make_examples(sentences):
    X, y = [], []

    for sentence in sentences:
        text = sentence.lower()
        sequence = "^" * CONTEXT + text + "$"

        for i in range(CONTEXT, len(sequence)):
            context = sequence[i-CONTEXT:i]
            target = sequence[i]

            X.append([stoi[c] for c in context])
            y.append(stoi[target])

    return np.array(X), np.array(y)

def collect(language, split):
    return make_examples(splits[language][split])

# --------------------------------------------------
# 2. One-hot context representation
# --------------------------------------------------

def encode(X):
    encoded = np.zeros((len(X), CONTEXT * V))
    for row, context in enumerate(X):
        for position, char_id in enumerate(context):
            encoded[row, position * V + char_id] = 1.0
    return encoded

# --------------------------------------------------
# 3. Softmax model
# --------------------------------------------------

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def train_model():
    W = np.random.normal(0, 0.01, (CONTEXT * V, V))
    b = np.zeros(V)

    training = [
        collect(language, "train")
        for language in splits
    ]

    X = np.concatenate([x for x, y in training])
    y = np.concatenate([y for x, y in training])
    X = encode(X)

    for epoch in range(EPOCHS):
        indices = np.random.permutation(len(y))
        X_epoch = X[indices]
        y_epoch = y[indices]

        batch_size = 128

        for start in range(0, len(y), batch_size):
            xb = X_epoch[start:start+batch_size]
            yb = y_epoch[start:start+batch_size]

            probabilities = softmax(xb @ W + b)
            probabilities[np.arange(len(yb)), yb] -= 1
            probabilities /= len(yb)

            grad_W = xb.T @ probabilities
            grad_b = probabilities.sum(axis=0)

            W -= LR * grad_W
            b -= LR * grad_b

    return W, b

# --------------------------------------------------
# 4. Evaluation
# --------------------------------------------------

def evaluate(W, b, language, split):
    X, y = collect(language, split)
    X = encode(X)

    probabilities = softmax(X @ W + b)
    predictions = np.argmax(probabilities, axis=1)

    nll = -np.log(
        np.maximum(probabilities[np.arange(len(y)), y], 1e-12)
    ).mean()

    accuracy = np.mean(predictions == y)

    diacritic_ids = [
        stoi[c] for c in DIACRITICS if c in stoi
    ]
    mask = np.isin(y, diacritic_ids)

    recall = (
        np.mean(predictions[mask] == y[mask])
        if mask.any() else None
    )

    return {
        "characters": int(len(y)),
        "next_character_accuracy": float(accuracy),
        "cross_entropy": float(nll),
        "swedish_diacritic_recall": (
            float(recall) if recall is not None else None
        )
    }

# --------------------------------------------------
# 5. Train baseline
# --------------------------------------------------

W, b = train_model()

# --------------------------------------------------
# 6. Targeted output-bias edit
# --------------------------------------------------

# The edit boosts Swedish-specific character logits.
direction = np.zeros(V)

for char in DIACRITICS:
    if char in stoi:
        direction[stoi[char]] = 1.0

# Select edit strength using validation data only.
candidates = np.linspace(0.0, 2.0, 21)
best_alpha = 0.0
best_loss = float("inf")

for alpha in candidates:
    candidate_b = b + alpha * direction

    losses = [
        evaluate(W, candidate_b, language, "validation")["cross_entropy"]
        for language in ["Swedish"]
    ]

    if losses[0] < best_loss:
        best_loss = losses[0]
        best_alpha = float(alpha)

targeted_b = b + best_alpha * direction

# --------------------------------------------------
# 7. Random edit control
# --------------------------------------------------

edit_vector = best_alpha * direction
edit_norm = np.linalg.norm(edit_vector)

random_direction = np.random.normal(size=V)
random_direction /= max(np.linalg.norm(random_direction), 1e-12)

random_b = b + edit_norm * random_direction

# --------------------------------------------------
# 8. Final held-out evaluation
# --------------------------------------------------

results = {
    "experiment": "Pilot: targeted output-bias editing",
    "seed": SEED,
    "architecture": "linear character-context softmax model",
    "context_length": CONTEXT,
    "epochs": EPOCHS,
    "vocabulary_size": V,
    "training_sentences_per_language": {
        language: len(splits[language]["train"])
        for language in splits
    },
    "validation_sentences_per_language": {
        language: len(splits[language]["validation"])
        for language in splits
    },
    "test_sentences_per_language": {
        language: len(splits[language]["test"])
        for language in splits
    },
    "targeted_edit_alpha": best_alpha,
    "targeted_edit_l2_norm": float(edit_norm),
    "conditions": {}
}

conditions = {
    "baseline": b,
    "targeted_edit": targeted_b,
    "random_edit": random_b
}

for condition, bias in conditions.items():
    results["conditions"][condition] = {}

    for language in splits:
        results["conditions"][condition][language] = evaluate(
            W, bias, language, "test"
        )

# --------------------------------------------------
# 9. Save outputs
# --------------------------------------------------

np.savez(
    OUT / "model_weights.npz",
    W=W,
    baseline_bias=b,
    targeted_bias=targeted_b,
    random_bias=random_b,
    vocabulary=np.array(vocab)
)

with open(OUT / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nEXPERIMENT COMPLETED")
print("=" * 55)

for condition, languages in results["conditions"].items():
    print(f"\n{condition.upper()}")

    for language, metrics in languages.items():
        print(
            f"{language}: "
            f"accuracy={metrics['next_character_accuracy']:.4f}, "
            f"cross_entropy={metrics['cross_entropy']:.4f}, "
            f"diacritic_recall={metrics['swedish_diacritic_recall']}"
        )

print("\nSelected targeted edit strength:", best_alpha)
print("Results saved to experiments/results.json")
print("Weights saved to experiments/model_weights.npz")
