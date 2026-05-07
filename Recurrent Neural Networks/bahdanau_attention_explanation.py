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
print("="*60)

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

print("="*60)

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
    h = np.tanh(W_f @ embeds[t] + U_f @ h + b_f)
    hf.append(h.copy())

print("how many hidden state we produced ?:", len(hf))
print("the last hidden state shape, hf[-1]:", hf[-1].shape)

print("="*60)

# step 4 -> Backward RNN Part

# introduce new weigths
W_b = np.random.randn(HIDDEN,EMBED) * 0.01
U_b = np.random.randn(HIDDEN, HIDDEN) * 0.01
b_b = np.zeros((HIDDEN,1))

# Let's put them into for loop
hb = [None] * len(embeds)
h = np.zeros((HIDDEN,1))

for t in reversed(range(len(embeds))):
    h = np.tanh(W_b @ embeds[t] + U_b @ h + b_b )
    hb[t] = h.copy()

print("hb[0] shape:", hb[0].shape) 
print("hb[4] shape:", hb[4].shape)


# What we did ? 
# Forward RNN reads data from left to right 
# t=0: h_f[0] = tanh(W_f·e(a) + U_f·0)        → only saw 'a'
# t=1: h_f[1] = tanh(W_f·e(b) + U_f·h_f[0])   → saw 'a','b'
# t=2: h_f[2] = tanh(W_f·e(c) + U_f·h_f[1])   → saw 'a','b','c'
# t=3: h_f[3] = tanh(W_f·e(d) + U_f·h_f[2])   → saw 'a','b','c','d'
# t=4: h_f[4] = tanh(W_f·e(<EOS>) + U_f·h_f[3]) → saw all

# Backward RNN reads data from right to left
# t=4: h_b[4] = tanh(W_b·e(<EOS>) + U_b·0)      → saw only '<EOS>'
#t=3: h_b[3] = tanh(W_b·e(d) + U_b·h_b[4])     → saw '<EOS>','d'
# t=2: h_b[2] = tanh(W_b·e(c) + U_b·h_b[3])     → saw '<EOS>','d','c'
# t=1: h_b[1] = tanh(W_b·e(b) + U_b·h_b[2])     → saw '<EOS>','d','c','b'
# t=0: h_b[0] = tanh(W_b·e(a) + U_b·h_b[1])     → saw all

# Now look at the position 'c'
# hf[2] # -> knows 'a', 'b', 'c' # past
# hb[2] # -> knows 'c', 'd', '<EOS>' # future

# lets union them
annotation_c = np.concatenate([hf[2] , hb[2]], axis = 0)
print("annotation 'c' shape:", annotation_c.shape)
print("annotation c :", annotation_c)

print("="*60)

# step 5 -> all annotations
annotations = []
for t in range(len(embeds)):
    h_t = np.concatenate([hf[t], hb[t]], axis = 0)
    annotations.append(h_t)

print("how many annotation ? ", len(annotations))
print("annotations[0] shape:", annotations[0].shape)

# so what is annotation here ? 
# annotation = the "special vector" which encoder produces for each source word
# our word is ['a', 'b', 'c', 'd', '<EOS>'] -> 5 words, 5 annotation

# annotation[0] -> summary for word 'a' (16,1)
# annotation[1] -> summary for word 'b' (16,1)
# annotation[2] -> summary for word 'c' (16,1)
# annotation[3] -> summary for word 'd' (16,1)
# annotation[4] -> summary for word '<EOS>' (16,1)

# Why it is called 'annotation' from Bahdanau ?
# In the classical encoder, it is taken last hidden state h[-1] normally, thus all the sentence is compressed 
# into one vector.
# Bahdanau rejected it -> said that " Let's represent all source word seperately, then decoder chooses desired one
# here "representing all words seperately is annotation", in equation 7 in article, we did it.

print("="*60)

# step 6 -> Attention
# before decoder produces a word, it asks "How much attention should I pay to each of the 5 annotations?"
# for this, it calculates score for each annotation
# e_j = v^T · tanh( W_a · s  +  U_a · h_j )

ATTN = 8
W_a = np.random.randn(ATTN, HIDDEN) * 0.01 # (8,8) - for decoder state
U_a = np.random.randn(ATTN, 2* HIDDEN) * 0.01 # (8,16) for annotation
v_a = np.random.randn(ATTN, 1) * 0.01 # (8,1)

# now, create a random decoder state
s = np.random.randn(HIDDEN,1) * 0.01 #(8,1)

scores = []
for t in range(len(embeds)):
    e_j = v_a.T @ np.tanh(W_a @ s + U_a @ annotations[t])
    scores.append(e_j)

print("how many scores ? :", len(scores))
print("scores[0].shape:", scores[0].shape)


# now convert each score to numpy array
scores = np.array([s.item() for s in scores]) # (5,) -> simple array
print("scores", scores)

# apply stable softmax
scores -= scores.max()
exp_s = np.exp(scores)
alphas = exp_s / exp_s.sum()

print("alphas:", alphas) # 5 number, and its sum must be 1
print("sum of alphas:", alphas.sum()) # 1.0

# the question is : what if alphas occurs as s = [0.02, 0.01, 0.90, 0.05, 0.02] after training ?
# this means that, in decoder step,  'c' word's annotation is weighted by %90 percent, and ignore others
# so it only focuses on 'c' !

# now last part - context vector
# c = Σ_j  α_j · h_j
# so multiply each annotation with its weight and sum them
# c = 0.02 * annotations[0] + 0.01 * annotations[1] + 0.90 * annotations[2]   ← !! dominant + 0.05 * annotations[3] + 0.02 * annotations[4]

# context vector 
c = np.zeros((2 * HIDDEN,1)) # (16,1) # why ? because c is weighted average of annotations, each annotation = [hf[t] ; hb[t]] = (8,1) + (8,1) = (16, 1)
for t in range(len(annotations)):
    c += alphas[t] * annotations[t]

print("context vector shape:", c.shape)   # (16, 1)

print("="*60)

# step 7 -> Decoder
# what we have ?
# s -> decoder hidden state 
# c -> context vector
# y_prev -> previously produced word

# Decoder works as -> s_new = tanh( W_d · e(y_prev)  +  U_d · s  +  C_d · c )
# then it produces a new word from s_new
# logits = W_o · s_new          →  score for every word
# probs  = softmax(logits)      →  probability distribution
# y      = argmax(probs)        →  maximum likelihood word

# define parameters
# Target embedding
E_tgt = np.random.randn(len(TGT), EMBED) * 0.01 #(6,4)

# Decoder RNN Weigths
W_d = np.random.randn(HIDDEN, EMBED) * 0.01
U_d = np.random.randn(HIDDEN, HIDDEN) * 0.01
C_d = np.random.randn(HIDDEN, 2* HIDDEN) * 0.01
b_d = np.zeros((HIDDEN,1))

# Output Projection
W_o = np.random.randn(len(TGT), HIDDEN) * 0.01
b_o = np.zeros((len(TGT),1))

# First decoder step: y_prev = <SOS>
y_prev_ix = tgt_to_ix['<SOS>']
e_y = E_tgt[y_prev_ix].reshape(-1,1) # (4,1)



s_new = np.tanh(W_d @ e_y + U_d @ s + C_d @ c)
print("s_new shape:", s_new.shape)

# output : which word ? 
logits = W_o @ s_new # (6,1)
logits -= logits.max()
probs = np.exp(logits) / np.exp(logits).sum() # (6,1)

y_ix = np.argmax(probs)
print("new produced word is:", ix_to_tgt[y_ix])

# Source: ['a', 'b', 'c', 'd', '<EOS>']
#          ↓
#    [Embedding]
#          ↓
#  [Forward RNN]  →  hf[0..4]
#  [Backward RNN] →  hb[0..4]
#          ↓
#  [Annotations]  →  h[t] = [hf[t]; hb[t]]  (16,1) x 5
#          ↓
#  [Attention]    →  scores → alphas → c  (16,1)
#          ↓
#  [Decoder]      →  s_new → logits → probs → word

print("="*60)


# now loop them all

output = []
s = np.random.randn(HIDDEN,1) * 0.01
y_prev_ix = tgt_to_ix['<SOS>']

for step in range(5): # max 5 steps
    # 1. embedding of y_prev
    e_y = E_tgt[y_prev_ix].reshape(-1,1)

    # 2. Attention: Scores, alphas, c
    scores = []
    for t in range(len(annotations)):
        e_j = v_a.T @ np.tanh(W_a @ s + U_a @ annotations[t])
        scores.append(e_j)
    
    scores = np.array([sc.item() for sc in scores])
    scores -= scores.max()
    alphas = np.exp(scores) / np.exp(scores).sum()
    c = np.zeros((2 * HIDDEN, 1))

    for t in range(len(annotations)):
        c += alphas[t] * annotations[t]

    # 3. Decoder Step
    s = np.tanh(W_d @ e_y + U_d @ s + C_d @ c)

    # 4. Output word
    logits = W_o @ s +  b_o
    logits -= logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    y_ix = np.argmax(probs)
    y_word = ix_to_tgt[y_ix]

    output.append(y_word)
    if y_word == '<EOS>':
        break

    # 5. Update y_prev for next step
    y_prev_ix = y_ix

print("Produced word:", output)

print("="*60)