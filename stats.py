
import re
import matplotlib.pyplot as plt

file_path = "./dailydialog/test/dialogues_test.txt"

# Load text
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Split into sentences (basic rule: ., !, ?)
sentences = re.split(r'[.!?]+', text)

# Clean sentences
sentences = [s.strip() for s in sentences if s.strip()]

# Compute sentence lengths (in words)
lengths = [len(s.split()) for s in sentences]

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(lengths, bins=50, edgecolor='black')
plt.title("Sentence Length Distribution")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.show()

plt.savefig('sentence_len')

# Split into sentences (basic rule: ., !, ?)
words = re.split(" ", text)


# Compute sentence lengths (in words)
lengths = [len(w) for w in words]

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(lengths, bins=20, edgecolor='black')
plt.title("Word Length Distribution")
plt.xlabel("Number of characters")
plt.ylabel("Frequency")
plt.show()

plt.savefig('words')