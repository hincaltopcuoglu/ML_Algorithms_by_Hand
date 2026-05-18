import numpy as np
np.random.seed(42)

# Toy task: "I am a student" -> "Je suis etudiant"
# (same spirit as Bahdanau demo, but with Transformer)

SRC = ["I", "am", "a", "student", "<EOS>"]
TGT = ["Je", "suis", "etudiant", "<SOS>", "<EOS>"]

src_to_ix = {w:i for i,w in enumerate(SRC)}
tgt_to_ix = {w:i for i,w in enumerate(TGT)}

ix_to_src = {i:w for i,w in enumerate(SRC)}
ix_to_tgt = {i:w for i,w in enumerate(TGT)}

print(src_to_ix)
print(tgt_to_ix)
print("="*60)
print(ix_to_src)
print(ix_to_tgt)
src_ix = [src_to_ix[w] for w in ["I", "am", "a", "student", "<EOS>"]]
print("src_ix:", src_ix)

print("="*60)

# Question : how transformer knows words order since there is no RNN ?
# Answer: Positional Encoding, Position information is added to each word's embedding

# "I"       → embedding[I]       + pos_encoding[0]
# "am"      → embedding[am]      + pos_encoding[1]
# "a"       → embedding[a]       + pos_encoding[2]
# "student" → embedding[student] + pos_encoding[3]

# positional encoding is calculated with sin / cosine functions
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

print("="*60)
print("=== Step 1: Embbedding + Positional Encodding ===")

D_MODEL = 8 # dimension of embeddig, in the paper it is 512, we hold it small here.

E_src = np.random.randn(len(SRC), D_MODEL) * 0.01 # (5,8)

print("=== Calculating Positional Encoding ===")
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len,d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / 10000 ** (2*i / d_model))
            PE[pos, i+1] = np.cos(pos / 10000 ** (2*i /d_model))
    return PE

pe = positional_encoding(len(SRC), D_MODEL)
print(pe)
print("PE shape:", pe.shape) #(5,8)

# Embbedding + positional Embedding
embeddings = E_src[src_ix] + pe #(5,8)
print(embeddings)
print("embeddings shape:", embeddings.shape)

print("="*60)

print("=== Self Attention ===")
# Self Attention step
# What is self attention ?
# In Bahdanau paper, every decoder asks to encoder annotations
# In Self Attention, every word, ask other words in its own sentences

# "studen" -> "Which words are related to me ?" -> "I": weak , "am": medium , "a": "weak" relationship

# Q = X · W_Q   ← "What I am seeking?" (Query)
# K = X · W_K   ← "What am I?"   (Key)
# V = X · W_V   ← "What is my information?"   (Value)

# Attention(Q, K, V) = softmax( Q · K^T / sqrt(d_k) ) · V

D_K = 4 # query / key dimension. In paper d_k = d_model / num_heads

print("=== W_Q, W_K, W_V Weight Matrices ===")
W_Q = np.random.randn(D_MODEL, D_K) * 0.01   # (8, 4)
W_K = np.random.randn(D_MODEL, D_K) * 0.01   # (8, 4)
W_V = np.random.randn(D_MODEL, D_K) * 0.01   # (8, 4)

print(W_Q)
print(W_K)
print(W_V)

# Calculate Q, K, V
Q = embeddings @ W_Q #(5,4) 
K = embeddings @ W_K #(5,4)
V = embeddings @ W_V #(5,4)

print("Q:", Q)
print("Q shape:", Q.shape)
print("K:", K)
print("K shape:", K.shape)
print("V:", V)
print("V shape:", V.shape)


#          "I"   "am"   "a"  "student"  "<EOS>"
# "I"      [  .     .     .      .         .  ]
# "am"     [  .     .     .      .         .  ]
# "a"      [  .     .     .      .         .  ]
# "student"[  .     .     .      .         .  ]
# "<EOS>"  [  .     .     .      .         .  ]

# (i,j) cell : "how i. word is related with j. word ?"

scores = Q @ K.T / np.sqrt(D_K) #(5,5)
print("scores shape:", scores.shape)

# apply softmax to each row seperately
scores -= scores.max(axis=1, keepdims = True) # stable softmax
attn_weights = np.exp(scores) / np.exp(scores).sum(axis = 1, keepdims = True)

print("attn_weights shape:", attn_weights.shape)   # (5, 5)
print("satir toplamlari:", attn_weights.sum(axis=1))  # all of them must be 1.0 

# output = attn_weights @ V
attn_output = attn_weights @ V #(5,4)
print("attn_output shape:", attn_output.shape)


print("=== Multi Head Attention ===")
## Multi-Head Attention
# Single head does only one-way relationship
# Multi-head does multiple-way relationship

# MultiHead(Q,K,V) = Concat(head_1, head_2, ..., head_h) @ W_O
# head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)

NUM_HEADS = 2 

# for every head, there are seperate W_Q, W_K, W_V
heads_output = []
for h in range(NUM_HEADS):
    W_Q_h = np.random.randn(D_MODEL, D_K) * 0.01
    W_K_h = np.random.randn(D_MODEL, D_K) * 0.01
    W_V_h = np.random.randn(D_MODEL, D_K) * 0.01

    Q_h = embeddings @ W_Q_h #(5,4) 
    K_h = embeddings @ W_K_h #(5,4)
    V_h = embeddings @ W_V_h 

    sc = Q_h @ K_h.T / np.sqrt(D_K)
    sc -= sc.max(axis= 1, keepdims = True)
    aw = np.exp(sc) / np.exp(sc).sum(axis = 1, keepdims = True)
    head_out = aw @ V_h
    heads_output.append(head_out)

# concat
concat = np.concatenate(heads_output, axis = 1) #(5,8)
print("concat shape:",concat.shape) #(5,8) = (5, D_MODEL)



# MultiHead(Q,K,V) = Concat(head_1, ..., head_h) @ W_O

# Linear projection after Multi-head attention
W_O = np.random.randn(D_MODEL, D_MODEL) * 0.01 #(8,8)
mha_output = concat @ W_O
print("Multi Head Attention Output Shape", mha_output.shape)



print("=== Feed Forward Network (FFN) ===")
### Feed Forward Network (FFN)
# it is two layered network applied for each token
# FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
# In paper it is 2048 but we will make it 32, (small sample)

# STEP 4: FFN
D_FF = 32
W1 = np.random.randn(D_MODEL, D_FF) * 0.01 #(8,32)
b1 = np.zeros((D_FF,)) 
W2 = np.random.randn(D_FF, D_MODEL) * 0.01 #(32, 8)
b2 = np.zeros((D_MODEL,))

ffn_hidden = mha_output @ W1 + b1 # (5,32)
ffn_hidden = np.maximum(0, ffn_hidden) # RELU
ffn_output = ffn_hidden @ W2 + b2 # (5,8)

print("FFN output shape:", ffn_output.shape)

print("=== Add & Norm ===")
# In the paper, there is a Layer Normalization at the end of each sublayer
# output = LayerNorm(x + sublayer(x))
# x original input, sublayer(x) = output of MHA AND FFN, It is residual conntection for not losing gradients

# LayerNorm is normalization of each token on itself
# LN(x) = (x - mean) / (std + ε) * γ + β

# step 5: Add & Norm
eps = 1e-6

# after MHA: residual + layer norm
x1 = embeddings + mha_output # residual (5,8)
mean1 = x1.mean(axis=1, keepdims=True)              # (5, 1)
std1  = x1.std(axis=1, keepdims=True)               # (5, 1)
x1_norm = (x1 - mean1) / (std1 + eps)              # (5, 8)
print("After MHA + Norm shape:", x1_norm.shape)

# after FFN: residual + layer norm
x2 = x1_norm + ffn_output # residual(5,8)
mean2 = x2.mean(axis=1, keepdims=True)
std2  = x2.std(axis=1, keepdims=True)
x2_norm = (x2 - mean2) / (std2 + eps)              # (5, 8)
print("After FFN + Norm shape:", x2_norm.shape)

print("=== Encoder Step is done. What we did: Input → Embedding + PE → Multi-Head Self-Attention → Add&Norm → FFN → Add&Norm")

print("="*60)
print("=== Decoder Step ===")
# first step Decoder input embedding
# == DECODER ==
D_MODEL_TGT = D_MODEL # same dimension

E_tgt = np.random.randn(len(TGT), D_MODEL) * 0.01

tgt_input = ["<SOS>", "Je", "suis", "etudiant"]
tgt_ix = [tgt_to_ix[w] for w in tgt_input]

pe_tgt = positional_encoding(len(tgt_ix), D_MODEL)
tgt_embeddings = E_tgt[tgt_ix] + pe_tgt #(4,8)
print("Target embeddings shape:", tgt_embeddings.shape)

print("=== Masked Self Attention")
# In normal self attention, each token can look to everywhere, but decoder has a rule:
# the token at position t, only can look at 0...t, can not see future tokens
# we do it with causal mask

# === Decoder Step 1: Masked Self-Attention ===
T = len(tgt_ix)  # 4
W_Q_dec = np.random.randn(D_MODEL, D_K) * 0.01
W_K_dec = np.random.randn(D_MODEL, D_K) * 0.01
W_V_dec = np.random.randn(D_MODEL, D_K) * 0.01

Q_dec = tgt_embeddings @ W_Q_dec #(4,4)
K_dec = tgt_embeddings @ W_K_dec #(4,4)
V_dec = tgt_embeddings @ W_V_dec #(4,4)

scores_dec = Q_dec @ K_dec.T / np.sqrt(D_K) #(4,4)

# CAUSAL MASK : UPPER TRIANGLE -inf
mask = np.triu(np.ones((T,T)), k=1) * -1e9
scores_dec = scores_dec + mask

scores_dec -= scores_dec.max(axis = 1, keepdims = True)
aw_dec = np.exp(scores_dec) / np.exp(scores_dec).sum(axis = 1, keepdims = True)
masked_attn_out = aw_dec @ V_dec #(4,4)

print("Masked attn weights:\n", np.round(aw_dec, 3))
print("Masked attn output shape:", masked_attn_out.shape)

# Linear projection: masked_attn_out -> D_MODEL
W_O_dec = np.random.randn(D_K, D_MODEL) * 0.01   # (4, 8)
masked_out_proj = masked_attn_out @ W_O_dec        # (4, 8)

# Add & Norm after masked self-attention
x_dec1 = tgt_embeddings + masked_out_proj
mean_d1 = x_dec1.mean(axis=1, keepdims=True)
std_d1  = x_dec1.std(axis=1, keepdims=True)
x_dec1_norm = (x_dec1 - mean_d1) / (std_d1 + eps)   # (4, 8)
print("After Masked Attn + Norm shape:", x_dec1_norm.shape)


print("=== Cross Attention ===")
# in the paper : "the queries come from the previous decoder layer, and the keys and values come from the output of the encoder"
# Think it like in Bahdanau paper:
# Decoder looks every hidden state of encoder at each step, so it asks "which word is more important ?"
# In Transformer, it does same thing but diferent name:

# Bahdanau	                  Transformer Cross-Attention
# Decoder hidden state s      Query Q (decoder'dan)
# Encoder annotations h_j     Key K (encoder'dan)
# Encoder annotations h_j     Value V (encoder'dan)
# Context vector c            cross_attn_out

# cross attention does : while I am producing the word "Je", which words should I pay attention to in source sentence ?
# it comes from Q decoder

# === Decoder Step 2: Cross-Attention ===
W_Q_cross = np.random.randn(D_MODEL, D_K) * 0.01
W_K_cross = np.random.randn(D_MODEL, D_K) * 0.01
W_V_cross = np.random.randn(D_MODEL, D_K) * 0.01

Q_cross = x_dec1_norm  @ W_Q_cross   # (4, 4)  <- from decoder
K_cross = x2_norm @ W_K_cross           # (5, 4)  <- from encoder
V_cross = x2_norm @ W_V_cross           # (5, 4)  <- from encoder

scores_cross = Q_cross @ K_cross.T / np.sqrt(D_K)  # (4, 5)
scores_cross -= scores_cross.max(axis=1, keepdims=True)
aw_cross = np.exp(scores_cross) / np.exp(scores_cross).sum(axis=1, keepdims=True)
cross_attn_out = aw_cross @ V_cross     # (4, 4)

print("Cross-attn weights shape:", aw_cross.shape)
print("Cross-attn output shape:", cross_attn_out.shape)



print("=== Add & Norm after Cross Attention ===")

# === Add & Norm after Cross-Attention ===
W_O_cross = np.random.randn(D_K, D_MODEL) * 0.01   # (4, 8)
cross_out_proj = cross_attn_out @ W_O_cross          # (4, 8)

x_dec2 = x_dec1_norm + cross_out_proj
mean_d2 = x_dec2.mean(axis=1, keepdims=True)
std_d2  = x_dec2.std(axis=1, keepdims=True)
x_dec2_norm = (x_dec2 - mean_d2) / (std_d2 + eps)   # (4, 8)
print("After Cross-Attn + Norm shape:", x_dec2_norm.shape)

# === Decoder FFN ===
W_FF1_dec = np.random.randn(D_MODEL, D_FF) * 0.01
b_FF1_dec = np.zeros((D_FF,))
W_FF2_dec = np.random.randn(D_FF, D_MODEL) * 0.01
b_FF2_dec = np.zeros((D_MODEL,))

ffn_dec = x_dec2_norm @ W_FF1_dec + b_FF1_dec
ffn_dec = np.maximum(0, ffn_dec)
ffn_dec = ffn_dec @ W_FF2_dec + b_FF2_dec            # (4, 8)

x_dec3 = x_dec2_norm + ffn_dec
mean_d3 = x_dec3.mean(axis=1, keepdims=True)
std_d3  = x_dec3.std(axis=1, keepdims=True)
x_dec3_norm = (x_dec3 - mean_d3) / (std_d3 + eps)   # (4, 8)
print("Decoder FFN + Norm shape:", x_dec3_norm.shape)

print("=== Final Step: Linear + Softmax ===")

# === Final: Linear + Softmax ===
W_out = np.random.randn(D_MODEL, len(TGT)) * 0.01   # (8, 5)
b_out = np.zeros((len(TGT),))

logits = x_dec3_norm @ W_out + b_out                 # (4, 5)
logits -= logits.max(axis=1, keepdims=True)
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # (4, 5)

predicted_ix = np.argmax(probs, axis=1)              # (4,)
predicted_words = [ix_to_tgt[i] for i in predicted_ix]

print("Logits shape:", logits.shape)
print("Probs shape:", probs.shape)
print("Predicted tokens:", predicted_words) # !!! there is no training at this step !!!!



# SRC: "I am a student <EOS>"
#      ↓
# Embedding + Positional Encoding
#      ↓
# Multi-Head Self-Attention + W_O
#      ↓
# Add & Norm
#      ↓
# Feed-Forward Network
#      ↓
# Add & Norm  →  encoder_output (x2_norm)
#      ↓
# ════ DECODER ════
# TGT: "<SOS> Je suis etudiant"
#      ↓
# Embedding + Positional Encoding
#      ↓
# Masked Multi-Head Self-Attention
#      ↓
# Add & Norm
#      ↓
# Cross-Attention (Q←decoder, K,V←encoder)
#      ↓
# Add & Norm
#      ↓
# Feed-Forward Network
#      ↓
# Add & Norm
#      ↓
# Linear + Softmax  →  predicted tokens

print("=== Traning Loop ===")

# Training loop
# first calculate cross-entropy
# then calculate loss
# In Decoder, the sentence is tgt_input = ["<SOS>", "Je", "suis", "etudiant"] 
# Model should predict word in every position

# Pozition 0 (<SOS>)      → "Je" should be predicted
# Pozition 1 (Je)         → "suis" should be predicted
# Pozition 2 (suis)       → "etudiant" should be predicted
# Pozition 3 (etudiant)   → "<EOS>" should be predicted


# === Cross-Entropy Loss ===
tgt_output = ["Je", "suis", "etudiant", "<EOS>"]
tgt_out_ix = [tgt_to_ix[w] for w in tgt_output]   # [0, 1, 2, 4]

# === Training Loop (W_out only) ===
LR = 0.1
N_EPOCHS = 200

for epoch in range(N_EPOCHS):
    # forward (last layer)
    logits = x_dec3_norm @ W_out + b_out
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    # loss
    loss = 0.0
    for t in range(len(tgt_out_ix)):
        loss += -np.log(probs[t, tgt_out_ix[t]] + 1e-9)
    loss /= len(tgt_out_ix)

    # backward (W_out gradient)
    dlogits = probs.copy()
    for t in range(len(tgt_out_ix)):
        dlogits[t, tgt_out_ix[t]] -= 1
    dlogits /= len(tgt_out_ix)

    dW_out = x_dec3_norm.T @ dlogits   # (8, 5)
    db_out = dlogits.sum(axis=0)       # (5,)

    W_out -= LR * dW_out
    b_out -= LR * db_out

    if epoch % 50 == 0:
        pred = [ix_to_tgt[i] for i in np.argmax(probs, axis=1)]
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Pred: {pred}")