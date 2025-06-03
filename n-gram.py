import numpy as np

# Vocabulary
vocab = ["I", "hate", "like", "surgeons", "surgery", "football", "vacations", "broccoli"]
vocab_size = len(vocab)
word_to_index = {word: i for i, word in enumerate(vocab)}

# Context size
context_size = 2  # e.g., using 2 previous words to predict the next word

# Example context: ["I", "like"]
context_words = ["I", "like"]

# One-hot encoding function
def one_hot_encode(word, vocab_size, word_to_index):
    vec = np.zeros(vocab_size)
    vec[word_to_index[word]] = 1
    return vec

# Generate input vector by concatenating one-hot vectors of context words
input_vector = np.concatenate([
    one_hot_encode(word, vocab_size, word_to_index) for word in context_words
])

# Display results
print("Vocabulary Size:", vocab_size)
print("Context Size:", context_size)
print("Input Vector Shape:", input_vector.shape)
print("Input Vector:", input_vector)
