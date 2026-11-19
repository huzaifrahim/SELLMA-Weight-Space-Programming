
import json
from pathlib import Path

# Load the dataset
data_path = Path("data/corpus.json")

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nDATASET SUMMARY")
print("-" * 40)

for language, sentences in data.items():
    text = " ".join(sentences)

    print(f"\nLanguage: {language}")
    print(f"Number of sentences: {len(sentences)}")
    print(f"Number of characters: {len(text)}")
    print(f"Unique characters: {len(set(text))}")

    if language == "Swedish":
        for char in ["å", "ä", "ö"]:
            print(f"Occurrences of {char}: {text.lower().count(char)}")
