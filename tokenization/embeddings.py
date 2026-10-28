import numpy as np
import tiktoken
import gensim.downloader as api

tokenizer = tiktoken.get_encoding("cl100k_base")

# vocab_size = 50000
# embedding_size = 768

# embedding_matrix = np.random.randn(vocab_size, embedding_size) * 0.02

# print(f"Embedding matrix shape: {embedding_matrix.shape}")
# print(f"  → {vocab_size:,} tokens")
# print(f"  → {embedding_size} dimensions per token")
# print(f"  → {vocab_size * embedding_size:,} total parameters just for embeddings!")

word = 'king'

tokens = tokenizer.encode(word)

print("\n")

print("Loading word vectors... (this takes ~1 minute)")
word_vectors = api.load("glove-wiki-gigaword-100")  # 100-dimensional GloVe
print(f"Loaded! Vocabulary: {len(word_vectors):,} words")

similar_to_words = word_vectors.most_similar(word, topn=10)
print(f"Words most similar to '{word}':\n")
for similar_word, score in similar_to_words:
    print(f"  {similar_word}: {score:.3f}")



