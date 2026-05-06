import numpy as np
np.random.seed(42)


## Vocabulary
SRC = ["a", "b", "c", "d", "<EOS>"]
TGT = ["a", "b", "c", "d", "<SOS>","<EOS>"]

src_to_ix = {c: i for i,c in enumerate(SRC)}
tgt_to_ix = {c:i for i,c in enumerate(TGT)}

ix_to_src = {i:c for i,c in enumerate(SRC)}
ix_to_tgt = {i:c for i,c in enumerate(TGT)}

# example sentence
src_sentence = ['a', 'b', 'c','d','<EOS>']
src_ix = [src_to_ix[c] for c in src_sentence]

print("src_ix:", src_ix)

## step 2: Embedding ##

"""
We should convert each word (index) to a vector.
'a' -> index 0 -> [0.12, -0.5, 0.3, ...]
We use a matrix to do this: E_src, shape (vocab_size, embed_dim).
'a's embedding = this matrix's 0th row.
"""

EMBED = 4 # embedding dimension

E_src = np.random.randn(len(SRC), EMBED) * 0.01

# What is the embedding of 'a'?
a_embed = E_src[src_to_ix['a']]
print("a_embed:", a_embed)

# how to get all src_ix list embeddings?
embeds = [E_src[i] for i in src_ix]
print("embeds:", embeds)

# so we should write it in matrix format :
embeds = [E_src[i].reshape(-1,1) for i in src_ix]
print("embbed[0] shape:", embeds[0].shape)


# step 3 -> Forward RNN Part
HIDDEN = 8
W_f = np.random.randn(HIDDEN, EMBED) * 0.01 # (8,4)
U_f = np.random.randn(HIDDEN, HIDDEN) * 0.01 # (8,8) # Note that U_f is the 
b_f = np.zeros((HIDDEN, 1))

# first step calculating for t=0
# formula is h_t = tanh( W_f · e(x_t)  +  U_f · h_{t-1}  +  b_f )
h_prev = np.zeros((HIDDEN,1))
e_x_0 = embeds[0]
h_0 = np.tanh(W_f @ e_x_0 + U_f @ h_prev + b_f )
print("h_0 shape:", h_0.shape)

# lets put it into for loop now
hf = []
h = np.zeros((HIDDEN,1))

for t in range(len(embeds)):
    h = np.tanh(W_f[t] @ embeds[t] + U_f @ h + b_f[t])
    hf.append(h.copy())

print("how many hidden state we produced ?:", len(hf))
print("the last hidden state shape, hf[-1]:", hf[-1].shape)
