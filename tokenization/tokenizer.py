import tiktoken
import numpy as np

tokenizer = tiktoken.get_encoding("cl100k_base")

print(f"Vocabulary size: {tokenizer.n_vocab:,} tokens")

text = 'Karan'
tokens = tokenizer.encode(text)
print(tokens)
for token in tokens:
    print(f"  {token} → '{tokenizer.decode([token])}'")
    
back_token = tokenizer.decode(tokens)
print(back_token)
