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

# step 8 -> training
# aim : reverse training data 'abcd' to 'dcba'
src_sentence = ['a', 'b', 'c', 'd', '<EOS>']
tgt_sentence = ['d', 'c', 'b', 'a', '<EOS>']

src_ix = [src_to_ix[c] for c in src_sentence]
dec_in = [tgt_to_ix['<SOS>']] + [tgt_to_ix[c] for c in tgt_sentence[:-1]]
dec_out = [tgt_to_ix[c] for c in tgt_sentence]

print("dec_in :",dec_in)
print("dec_in :",[ix_to_tgt[i] for i in dec_in])
print("dec_out :",[ix_to_tgt[i] for i in dec_out])

# Why we seperated dec_in and dec_out ? 
# Decoder works t-1 time manner
# In every step, model does :
# I see: dec_in[i] -> previous word was it
# I should produce -> dec_out[i] -> next word should be it
# So it should produce 'd' when it sees '<SOS>' at step 0, and at step 1 it sees 'd' and should produce 'c' ....

# Decoder step:   0        1        2        3        4
#                 ↓        ↓        ↓        ↓        ↓
# dec_in :      <SOS>      'd'      'c'      'b'      'a'      ← input (What I see ?)
# dec_out:       'd'       'c'      'b'      'a'     <EOS>     ← target (What should I produce?)



# loss calculation
# run encoder first, (I re-write again)
embeds = [E_src[i].reshape(-1,1) for i in src_ix]

hf = []
h = np.zeros((HIDDEN,1))
for t in range(len(embeds)):
    h = np.tanh(W_f @ embeds[t] + U_f @ h + b_f)
    hf.append(h.copy())

hb = [None] * len(embeds)
h = np.zeros((HIDDEN,1))
for t in reversed(range(len(embeds))):
    h = np.tanh(W_b @ embeds[t] + U_b @ h + b_b)
    hb[t] = h.copy()

annotations = [np.concatenate([hf[t], hb[t]], axis = 0) for t in range(len(embeds))]

# decoder + loss
s = np.zeros((HIDDEN,1))
loss = 0.0

for i in range(len(dec_in)):
    e_y = E_tgt[dec_in[i]].reshape(-1,1)

    # attention
    scores = []
    for t in range(len(annotations)):
        e_j = v_a.T @ np.tanh(W_a @ s + U_a @ annotations[t])
        scores.append(e_j)

    scores = np.array([sc.item() for sc in scores])
    scores -= scores.max()
    alphas = np.exp(scores) / np.exp(scores).sum()
    c = sum(alphas[t] * annotations[t] for t in range(len(annotations)))

    # decoder step
    s = np.tanh(W_d @ e_y + U_d @ s + C_d @ c + b_d)

    # output
    logits = W_o @ s + b_o
    logits -= logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()

    loss += -np.log(probs[dec_out[i],0] + 1e-9)

print(f"Loss: {loss:.4f}")

print("="*60)

# Backward: output layer
# d_logits = probs - one_hot(target)
d_logits = probs.copy() # (6,1)
d_logits[dec_out[i],0] -= 1.0

# W_o gradient
dW_o = d_logits @ s.T # (6,1) @ (1,8) = (6,8)

# s gradient
d_s = W_o.T @ d_logits # (8,6) @ (6,1) = (8,1)

print("="*60)
print("Test for one step")
# only for last step i=4 -> gradient test
d_logits = probs.copy()
d_logits[dec_out[4],0] -= 1.0

dW_o = d_logits @ s.T
d_s =  W_o.T @ d_logits

print("dW_o shape:", dW_o.shape) # (6,8)
print("d_s shape:", d_s.shape) # (8,1)

print("="*60)

# now loop with backward pass
s = np.zeros((HIDDEN,1))
loss = 0.0

# lists to be saved
all_s = [s.copy()] # all_s[0] = s0, all_s[1] = s1 ...
all_probs = []
all_c = []
all_alphas = []
all_ey = []

for i in range(len(dec_in)):
    e_y = E_tgt[dec_in[i]].reshape(-1,1)

    # attention
    scores = []
    for t in range(len(annotations)):
        e_j = v_a.T @ np.tanh(W_a @ s + U_a @ annotations[t])
        scores.append(e_j)

    scores = np.array([sc.item() for sc in scores])
    scores -= scores.max()
    alphas = np.exp(scores) / np.exp(scores).sum()
    c = sum(alphas[t] * annotations[t] for t in range(len(annotations)))

    # decoder step
    s = np.tanh(W_d @ e_y + U_d @ s + C_d @ c + b_d)

    # output 
    logits = W_o @ s + b_o
    logits -= logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()

    loss += -np.log(probs[dec_out[i],0] + 1e-9)

    # save
    all_s.append(s.copy())
    all_probs.append(probs.copy())
    all_c.append(c.copy())
    all_alphas.append(alphas.copy())
    all_ey.append(e_y.copy())

print(f"Loss: {loss:.4f}")
print(f"all_s length: {len(all_s)}") # must be 6 (s0 + 5 steps)
print(f"all_probs length: {len(all_probs)}") # must be 5

print("="*60)

# backward pass
# gradient accumulators - same shape for every parameter
dW_o = np.zeros_like(W_o)
db_o = np.zeros_like(b_o)
dW_d = np.zeros_like(W_d)
dU_d = np.zeros_like(U_d)
dC_d = np.zeros_like(C_d)
db_d = np.zeros_like(b_d)
dE_tgt = np.zeros_like(E_tgt)
dW_a     = np.zeros_like(W_a)
dU_a     = np.zeros_like(U_a)
dv_a_grad = np.zeros_like(v_a)

# the gradient comes from decoder recurrence (it will come from next step)
d_s_next = np.zeros((HIDDEN,1))

for i in reversed(range(len(dec_in))):
    s_i = all_s[i + 1] # this step's hidden state
    s_prev = all_s[i] # previous hidden state
    probs = all_probs[i]
    c = all_c[i]
    e_y = all_ey[i]

    # 1. Output layer Gradient
    d_logits = probs.copy()
    d_logits[dec_out[i],0] -= 1.0 #(6,1)

    dW_o += d_logits @ s_i.T #(6,8)
    db_o += d_logits #(6,1)

    # the gradient comes to s_i: output + next step
    d_s = W_o.T @ d_logits + d_s_next #(8,1)



    # 2. Decoder RNN: s_i = tanh(...)
    # derivation of tanh
    d_pre = (1 -s_i **2) * d_s #(8,1)

    dW_d += d_pre @ e_y.T #(8,4)
    dU_d += d_pre @ s_prev.T #(8,8)
    dC_d += d_pre @ c.T #(8,16)
    db_d += d_pre #(8,1)
    dE_tgt[dec_in[i]] += (W_d.T @ d_pre).squeeze()

    # previous gradient comes to s
    #d_s_next = U_d.T @ d_pre #(8,1)
    d_c_i = C_d.T @ d_pre
    alphas_i = all_alphas[i]
    d_alpha = np.array([(annotations[t].T @ d_c_i).item() for t in range(len(annotations))])
    dot = np.dot(alphas_i, d_alpha)
    d_scores = alphas_i * (d_alpha - dot)

    d_s_from_attn = np.zeros((HIDDEN, 1))

    for j in range(len(annotations)):
        tv = np.tanh(W_a @ s_prev + U_a @ annotations[j])
        d_tanh = v_a * d_scores[j]
        dv_a_grad += tv * d_scores[j]
        d_pre_a = (1- tv**2) * d_tanh
        dW_a += d_pre_a @ s_prev.T
        dU_a += d_pre_a @ annotations[j].T
        d_s_from_attn += W_a.T @ d_pre_a

    # d_s_next is updated - from now it comes from both decoder and attention
    d_s_next = U_d.T @ d_pre + d_s_from_attn

print("dW_a shape:", dW_a.shape) #(8,8)
print("dU_a shape:", dU_a.shape) # (8,16)

print("="*60)

# ── PARAMETER UPDATE (SGD) ──
LR = 0.01
W_o  -= LR * dW_o
b_o  -= LR * db_o
W_d  -= LR * dW_d
U_d  -= LR * dU_d
C_d  -= LR * dC_d
b_d  -= LR * db_d
W_a  -= LR * dW_a
U_a  -= LR * dU_a
v_a  -= LR * dv_a_grad
E_tgt -= LR * dE_tgt



print("="*60)
# now loop everything to train !!

# Multiple reverse pairs so the model cannot memorize one fixed length pattern
# without using the source; attention must learn which source position matters.
# Long reverse pair repeated so its gradient is not drowned by short pairs.
LONG = (['a', 'b', 'c', 'd', '<EOS>'], ['d', 'c', 'b', 'a', '<EOS>'])
DATASET = (
    [LONG] * 6
    + [
        (['a', 'b', '<EOS>'], ['b', 'a', '<EOS>']),
        (['b', 'c', 'd', '<EOS>'], ['d', 'c', 'b', '<EOS>']),
        (['c', 'd', '<EOS>'], ['d', 'c', '<EOS>']),
        (['a', 'c', '<EOS>'], ['c', 'a', '<EOS>']),
        (['b', 'd', '<EOS>'], ['d', 'b', '<EOS>']),
        (['a', 'b', 'c', '<EOS>'], ['c', 'b', 'a', '<EOS>']),
        (['a', 'd', '<EOS>'], ['d', 'a', '<EOS>']),
    ]
)


def encode_sentence(src_ix_local):
    """BiRNN encoder for one source index list -> list of annotation vectors."""
    embeds_local = [E_src[i].reshape(-1, 1) for i in src_ix_local]
    hf_local = []
    h = np.zeros((HIDDEN, 1))
    for t in range(len(embeds_local)):
        h = np.tanh(W_f @ embeds_local[t] + U_f @ h + b_f)
        hf_local.append(h.copy())
    hb_local = [None] * len(embeds_local)
    h = np.zeros((HIDDEN, 1))
    for t in reversed(range(len(embeds_local))):
        h = np.tanh(W_b @ embeds_local[t] + U_b @ h + b_b)
        hb_local[t] = h.copy()
    ann = [np.concatenate([hf_local[t], hb_local[t]], axis=0) for t in range(len(embeds_local))]
    return ann


# Per-example SGD + clipping (long pair oversampled in DATASET).
LR = 0.08
CLIP = 4.0
N_EPOCHS = 2200
N_PAIRS = len(DATASET)

for epoch in range(N_EPOCHS):
    order = np.random.permutation(N_PAIRS)
    epoch_loss = 0.0

    for k in order:
        src_ch, tgt_ch = DATASET[int(k)]
        src_ix_local = [src_to_ix[c] for c in src_ch]
        dec_in_local = [tgt_to_ix['<SOS>']] + [tgt_to_ix[c] for c in tgt_ch[:-1]]
        dec_out_local = [tgt_to_ix[c] for c in tgt_ch]

        annotations = encode_sentence(src_ix_local)

        # 2. FORWARD + LOSS
        s = np.zeros((HIDDEN, 1))
        loss = 0.0
        all_s = [s.copy()]
        all_probs = []
        all_c = []
        all_alphas = []
        all_ey = []

        for i in range(len(dec_in_local)):
            e_y = E_tgt[dec_in_local[i]].reshape(-1, 1)

            scores = []
            for t in range(len(annotations)):
                e_j = v_a.T @ np.tanh(W_a @ s + U_a @ annotations[t])
                scores.append(e_j)
            scores = np.array([sc.item() for sc in scores])
            scores -= scores.max()
            alphas = np.exp(scores) / np.exp(scores).sum()
            c = sum(alphas[t] * annotations[t] for t in range(len(annotations)))

            s = np.tanh(W_d @ e_y + U_d @ s + C_d @ c + b_d)

            logits = W_o @ s + b_o
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()

            loss += -np.log(probs[dec_out_local[i], 0] + 1e-9)
            all_s.append(s.copy())
            all_probs.append(probs.copy())
            all_c.append(c.copy())
            all_alphas.append(alphas.copy())
            all_ey.append(e_y.copy())

        epoch_loss += loss

        # 3. BACKWARD
        dW_o = np.zeros_like(W_o)
        db_o = np.zeros_like(b_o)
        dW_d = np.zeros_like(W_d)
        dU_d = np.zeros_like(U_d)
        dC_d = np.zeros_like(C_d)
        db_d = np.zeros_like(b_d)
        dE_tgt = np.zeros_like(E_tgt)
        dW_a = np.zeros_like(W_a)
        dU_a = np.zeros_like(U_a)
        dv_a_grad = np.zeros_like(v_a)
        d_s_next = np.zeros((HIDDEN, 1))

        for i in reversed(range(len(dec_in_local))):
            s_i = all_s[i + 1]
            s_prev = all_s[i]
            probs = all_probs[i]
            c = all_c[i]
            e_y = all_ey[i]

            d_logits = probs.copy()
            d_logits[dec_out_local[i], 0] -= 1.0
            dW_o += d_logits @ s_i.T
            db_o += d_logits
            d_s = W_o.T @ d_logits + d_s_next
            d_pre = (1 - s_i**2) * d_s
            dW_d += d_pre @ e_y.T
            dU_d += d_pre @ s_prev.T
            dC_d += d_pre @ c.T
            db_d += d_pre
            dE_tgt[dec_in_local[i]] += (W_d.T @ d_pre).squeeze()

            d_c_i = C_d.T @ d_pre
            alphas_i = all_alphas[i]
            d_alpha = np.array([(annotations[t].T @ d_c_i).item() for t in range(len(annotations))])
            dot = np.dot(alphas_i, d_alpha)
            d_scores = alphas_i * (d_alpha - dot)
            d_s_from_attn = np.zeros((HIDDEN, 1))
            for j in range(len(annotations)):
                tv = np.tanh(W_a @ s_prev + U_a @ annotations[j])
                d_tanh = v_a * d_scores[j]
                dv_a_grad += tv * d_scores[j]
                d_pre_a = (1 - tv**2) * d_tanh
                dW_a += d_pre_a @ s_prev.T
                dU_a += d_pre_a @ annotations[j].T
                d_s_from_attn += W_a.T @ d_pre_a
            d_s_next = U_d.T @ d_pre + d_s_from_attn

        # clip then update (one step per training pair)
        for g in (dW_o, db_o, dW_d, dU_d, dC_d, db_d, dE_tgt, dW_a, dU_a, dv_a_grad):
            np.clip(g, -CLIP, CLIP, out=g)

        W_o -= LR * dW_o
        b_o -= LR * db_o
        W_d -= LR * dW_d
        U_d -= LR * dU_d
        C_d -= LR * dC_d
        b_d -= LR * db_d
        W_a -= LR * dW_a
        U_a -= LR * dU_a
        v_a -= LR * dv_a_grad
        E_tgt -= LR * dE_tgt

    if epoch % 200 == 0:
        # metric 1: train loss
        train_loss = epoch_loss

        # metric 2: predicted sequence on a fixed demo input
        demo_src_ix = [src_to_ix[c] for c in ['a', 'b', 'c', 'd', '<EOS>']]
        demo_annotations = encode_sentence(demo_src_ix)
        s_eval = np.zeros((HIDDEN, 1))
        y_prev_eval = tgt_to_ix['<SOS>']
        pred_seq = []
        for _ in range(6):
            e_y_eval = E_tgt[y_prev_eval].reshape(-1, 1)
            scores_eval = []
            for t in range(len(demo_annotations)):
                e_j_eval = v_a.T @ np.tanh(W_a @ s_eval + U_a @ demo_annotations[t])
                scores_eval.append(e_j_eval)
            scores_eval = np.array([sc.item() for sc in scores_eval])
            scores_eval -= scores_eval.max()
            alphas_eval = np.exp(scores_eval) / np.exp(scores_eval).sum()
            c_eval = sum(alphas_eval[t] * demo_annotations[t] for t in range(len(demo_annotations)))
            s_eval = np.tanh(W_d @ e_y_eval + U_d @ s_eval + C_d @ c_eval + b_d)
            logits_eval = W_o @ s_eval + b_o
            logits_eval -= logits_eval.max()
            probs_eval = np.exp(logits_eval) / np.exp(logits_eval).sum()
            y_eval = int(np.argmax(probs_eval))
            pred_word = ix_to_tgt[y_eval]
            pred_seq.append(pred_word)
            if pred_word == '<EOS>':
                break
            y_prev_eval = y_eval

        print(f"Epoch {epoch:4d} | train loss: {train_loss:.4f} | predicted sequence: {pred_seq}")




print("="*60)

## What is alignment ?
# After training, alphas are being calculated for each decoder step.
# step 0 -> while 'd' is produced, which word that the source looked ?
# step 1 -> while 'c' is produced, which word that the source looked ?
# ...

# if you get all "alphas" in to one matrix, the result is:
#              a     b     c     d    <EOS>
# out[0]='d'  0.02  0.01  0.03  0.91  0.03   ← looks 'd'
# out[1]='c'  0.01  0.02  0.94  0.02  0.01   ← looks 'c'
# out[2]='b'  0.02  0.93  0.02  0.02  0.01   ← looks 'b'
# out[3]='a'  0.94  0.02  0.02  0.01  0.01   ← looks 'a'



# Test after Training — re-encode the demo source so annotations match this sentence
demo_src_chars = ['a', 'b', 'c', 'd', '<EOS>']
demo_tgt_chars = ['d', 'c', 'b', 'a', '<EOS>']
src_ix_demo = [src_to_ix[c] for c in demo_src_chars]
annotations = encode_sentence(src_ix_demo)
src_sentence = demo_src_chars  # column labels for alignment matrix

s = np.zeros((HIDDEN, 1))
y_prev_ix = tgt_to_ix['<SOS>']
output = []
alignment = [] # we will save every steps' alignments

for step in range(6):
    e_y = E_tgt[y_prev_ix].reshape(-1, 1)

    scores = []
    for t in range(len(annotations)):
        e_j = v_a.T @ np.tanh(W_a @ s + U_a @ annotations[t])
        scores.append(e_j)
    scores = np.array([sc.item() for sc in scores])
    scores -= scores.max()
    alphas = np.exp(scores) / np.exp(scores).sum()
    c = sum(alphas[t] * annotations[t] for t in range(len(annotations)))

    s = np.tanh(W_d @ e_y + U_d @ s + C_d @ c + b_d)
    logits = W_o @ s + b_o
    logits -= logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    y_ix = np.argmax(probs)
    y_word = ix_to_tgt[y_ix]

    alignment.append(alphas.copy())
    output.append(y_word)

    if y_word == '<EOS>':
        break
    y_prev_ix = y_ix


# alignment matrix yazdir (soft alpha = paper'daki Figure 3 benzeri)
print("\nAlignment Matrix (soft weights alpha_ij):")
print("        " + "  ".join(f"{c:>6}" for c in src_sentence))
for i, (word, weights) in enumerate(zip(output, alignment)):
    print(f"out[{i}]={word}  " + "  ".join(f"{w:>6.3f}" for w in weights))

# Softmax dagilimi duz olsa bile argmax hangi kaynak pozisyonuna en cok baktigini gosterir
print("\nArgmax source position per decoder step (0=a ... 4=<EOS>):")
for i, weights in enumerate(alignment):
    j = int(np.argmax(weights))
    print(f"  step {i} ({output[i]:>5}) -> index {j} ({src_sentence[j]})")

print("\nNote: Correct translation with nearly-uniform alphas can happen because the "
      "decoder RNN can carry most of the 'what to say next' signal; sharper alphas "
      "often appear when the task forces reordering or when the encoder is trained too.")

print(f"\nInput:  {src_sentence}")
print(f"Output: {output}")
print(f"Expected: {demo_tgt_chars}")