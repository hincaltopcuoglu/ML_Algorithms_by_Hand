from typing import Any
import numpy as np
np.random.seed(42)
# ── Toy EN → FR corpus ──────────────────────────────────────────
raw_pairs = [
    (["I",   "love",  "you"],          ["je",   "t'aime"]),
    (["I",   "am",    "happy"],        ["je",   "suis",  "heureux"]),
    (["she", "is",    "kind"],         ["elle", "est",   "gentille"]),
    (["we",  "learn", "together"],     ["nous", "apprenons", "ensemble"]),
    (["he",  "reads", "books"],        ["il",   "lit",   "des", "livres"]),
    (["I",   "speak", "French"],       ["je",   "parle", "francais"]),
    (["you", "are",   "welcome"],      ["vous", "etes",  "bienvenus"]),
    (["she", "loves", "music"],        ["elle", "aime",  "la", "musique"]),
]


# vocabulary

def build_vocab(pairs, side, specials):
    words = set()
    for pair in pairs:
        for w in pair[side]:
            words.add(w)
    vocab = specials + sorted(words)
    return {w: i for i,w in enumerate(vocab)}, {i:w for i,w in enumerate(vocab)}

en_to_ix, ix_to_en = build_vocab(raw_pairs, 0, ["<PAD>", "<EOS>"])
fr_to_ix, ix_to_fr = build_vocab(raw_pairs, 1, ["<PAD>", "<SOS>", "<EOS>"])

EN_SIZE = len(en_to_ix)
FR_SIZE = len(fr_to_ix)

print("English words:", en_to_ix)
print("French words:", fr_to_ix)
print("EN vocab size:", EN_SIZE)
print("FR vocab size:", FR_SIZE)


# ── Data preparation ─────────────────────────────────────────────

def make_pair(en_words, fr_words):
    src = [en_to_ix[w] for w in en_words] + [en_to_ix["<EOS>"]]
    tgt_in  = [fr_to_ix["<SOS>"]] + [fr_to_ix[w] for w in fr_words]
    tgt_out = [fr_to_ix[w] for w in fr_words] + [fr_to_ix["<EOS>"]]
    return src, tgt_in, tgt_out
dataset = [make_pair(en, fr) for en, fr in raw_pairs]
# test
src, tgt_in, tgt_out = dataset[0]
print("src     :", src)
print("tgt_in  :", tgt_in)
print("tgt_out :", tgt_out)



# -- Hyperparameters ----------------------------------------------

D_MODEL = 16 # Embedding Dimension
D_K = 8 # query/key dimension per head
D_FF = 32 # feed forwawrd inner dimension
N_HEADS = 2 # Number of attention heads
LR = 0.01 
EPOCHS = 300

class Transformer:
    def __init__(self,src_vocab_size, tgt_vocab_size, d_model, d_k, d_ff, n_heads):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.d_k = d_k
        self.d_ff = d_ff
        self.n_heads = n_heads

        # Embeddings 
        self.E_src = np.random.randn(src_vocab_size, d_model) * 0.01 # <- English word embeddings
        self.E_tgt = np.random.randn(tgt_vocab_size, d_model) * 0.01 # <- French word embeddings

        # Encoder Self-Attention
        self.W_Q_enc = [np.random.randn(d_model, d_k) * 0.01 for _ in range(n_heads)]
        self.W_K_enc = [np.random.randn(d_model, d_k) * 0.01 for _ in range(n_heads)]
        self.W_V_enc = [np.random.randn(d_model, d_k) * 0.01 for _ in range(n_heads)]
        self.W_O_enc = np.random.randn(n_heads * d_k, d_model) * 0.01

        # Encoder FFN (2 MATRICES + 2 BIASES)S
        self.W1_enc = np.random.randn(d_model, d_ff) * 0.01
        self.b1_enc = np.zeros((d_ff,))
        self.W2_enc = np.random.randn(d_ff, d_model) * 0.01
        self.b2_enc = np.zeros((d_model,))

        # Decoder masked Self-Attention
        self.W_Q_dec = [np.random.randn(d_model, d_k) * 0.01 for _ in range(n_heads)]
        self.W_K_dec = [np.random.randn(d_model, d_k) * 0.01 for _ in range(n_heads)]
        self.W_V_dec = [np.random.randn(d_model, d_k) * 0.01 for _ in range(n_heads)]
        self.W_O_dec = np.random.randn(n_heads * d_k, d_model) * 0.01

        # Decoder cross-attention (Q from decoder , K/V from encoder)
        self.W_Q_cross = [np.random.randn(d_model, d_k) * 0.01 for _ in range(n_heads)]
        self.W_K_cross = [np.random.randn(d_model, d_k) * 0.01 for _ in range(n_heads)]
        self.W_V_cross = [np.random.randn(d_model, d_k) * 0.01 for _ in range(n_heads)]
        self.W_O_cross = np.random.randn(n_heads * d_k, d_model) * 0.01

        # Decoder FFN (2 MATRICES + 2 BIASES)
        self.W1_dec = np.random.randn(d_model, d_ff) * 0.01
        self.b1_dec = np.zeros((d_ff,))
        self.W2_dec = np.random.randn(d_ff, d_model) * 0.01
        self.b2_dec = np.zeros((d_model,))

        # Output layer
        self.W_out = np.random.randn(d_model, tgt_vocab_size) * 0.01
        self.b_out = np.zeros((tgt_vocab_size,))


    # positional encoding
    def positional_encoding(self, seq_len):
        PE = np.zeros((seq_len, self.d_model))
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                PE[pos, i]   = np.sin(pos / 10000 ** (2*i / self.d_model))
                PE[pos, i+1] = np.cos(pos / 10000 ** (2*i / self.d_model))
        return PE

    # softmax
    def softmax(self, x):
        x = x - x.max(axis = -1 , keepdims = True) # numerical stability
        return np.exp(x) / np.exp(x).sum(axis = -1, keepdims = True)

    # layer norm.
    def layer_norm(self, x, eps = 1e-6):
        mean = x.mean(axis = -1, keepdims = True)
        std = x.std(axis = -1, keepdims = True)
        return (x - mean) / (std + eps)

    # multihead attention
    def multihead_attention(self, Q_in, K_in, V_in, W_Qs, W_Ks, W_Vs, W_O, mask=None):
        heads = []
        for h in range(self.n_heads):
            Q = Q_in @ W_Qs[h]
            K = K_in @ W_Ks[h]
            V = V_in @ W_Vs[h]
            scores = Q @ K.T / np.sqrt(self.d_k) # (seq, seq)
            if mask is not None:
                scores = scores + mask
            attn = self.softmax(scores)
            heads.append(attn @ V)
        concat = np.concatenate(heads, axis = 1) # (seq, n_heads * d_k)
        return concat @ W_O # (seq, d_model)

    def encode(self, src_ix):
        # step 1: embedding + positional encoding
        x = self.E_src[src_ix]                                          # (src_len, d_model)
        x = x + self.positional_encoding(len(src_ix))

        # step 2: Multihead Self-Attention
        attn = self.multihead_attention(x, x, x, self.W_Q_enc, self.W_K_enc, self.W_V_enc, self.W_O_enc)

        # step 3: Add & Norm
        x = self.layer_norm(x + attn)

        # step 4: FFN  — save intermediates for backward
        x_enc_before_ffn = x
        enc_ffn_hidden = np.maximum(0, x @ self.W1_enc + self.b1_enc)   # ReLU
        ffn = enc_ffn_hidden @ self.W2_enc + self.b2_enc

        # step 5: Add & Norm
        x = self.layer_norm(x + ffn)

        enc_cache = (x_enc_before_ffn, enc_ffn_hidden, src_ix)
        return x, enc_cache                                              # encoder output + cache

    def decode(self, tgt_ix, enc_out):
        # step 1: target embedding + positional encoding
        x = self.E_tgt[tgt_ix] # (tgt_len, d_model)
        x = x + self.positional_encoding(len(tgt_ix)) # (tgt_len, d_model)

        # step 2: masked multi-head self-attention
        # build causal mask so position t cannot see t+1, t+2....
        T = len(tgt_ix)
        mask = np.triu(np.ones((T,T)), k=1) * -1e9 # upper triangle = -inf

        attn = self.multihead_attention(x, x, x, self.W_Q_dec, self.W_K_dec, self.W_V_dec, self.W_O_dec, mask=mask)

        # step 3: Add & Norm
        x = self.layer_norm(x + attn)

        Q_dec  = x
        K_enc  = enc_out
        V_enc  = enc_out

        # step 4: cross-attention — explicit loop to save attn weights & V per head
        # Q comes from decoder ("what am I looking for?")
        # K and V come from encoder ("what does each source word contain?")
        cross_heads      = []
        cross_attn_ws    = []   # attention weights per head - needed for backward
        cross_Vs         = []   # V vectors per head        - needed for backward
        for h in range(self.n_heads):
            Q_h  = Q_dec @ self.W_Q_cross[h]           # (T_tgt, d_k)
            K_h  = K_enc @ self.W_K_cross[h]           # (T_src, d_k)
            V_h  = V_enc @ self.W_V_cross[h]           # (T_src, d_k)
            sc   = Q_h @ K_h.T / np.sqrt(self.d_k)    # (T_tgt, T_src)
            aw   = self.softmax(sc)                    # (T_tgt, T_src)
            cross_attn_ws.append(aw)
            cross_Vs.append(V_h)
            cross_heads.append(aw @ V_h)
        cross = np.concatenate(cross_heads, axis=1) @ self.W_O_cross

        # STEP 5: Add & Norm
        x = self.layer_norm(x + cross)

        x_before_ffn = x
        # step 6: FFN
        ffn_hidden = np.maximum(0, x @ self.W1_dec + self.b1_dec) # RELU
        ffn = ffn_hidden @ self.W2_dec + self.b2_dec

        # step 7: Add & Norm
        x = self.layer_norm(x + ffn)
        
        cache = (x_before_ffn, ffn_hidden, cross_attn_ws, cross_Vs)
        
        return x, cache


    def forward(self, src_ix, tgt_ix):
        # 1. encode the source sentence
        enc_out, enc_cache = self.encode(src_ix)            # (src_len, d_model)

        # 2. Decode using encoder output
        dec_out, dec_cache = self.decode(tgt_ix, enc_out)  # (tgt_len, d_model)

        # 3. project to vocabulary scores
        logits = dec_out @ self.W_out + self.b_out          # (tgt_len, tgt_vocab_size)
        probs  = self.softmax(logits)                        # (tgt_len, tgt_vocab_size)

        return probs, dec_out, dec_cache, enc_cache
        

    def train_step(self, src_ix, tgt_ix, tgt_out_ix, lr):
        T = len(tgt_out_ix)

        # forward pass
        probs, dec_out, dec_cache, enc_cache = self.forward(src_ix, tgt_ix)
        x_before_ffn, ffn_hidden, cross_attn_ws, cross_Vs = dec_cache
        x_enc_before_ffn, enc_ffn_hidden, src_ix_saved = enc_cache

        # loss (cross-entropy)
        loss = sum(-np.log(probs[t, tgt_out_ix[t]] + 1e-9) for t in range(T)) / T

        # BACKWARD PASS
        # Step 1: dL/d(logits) = probs - one_hot(target)
        d_logits = probs.copy()
        for t, ix in enumerate(tgt_out_ix):
            d_logits[t,ix] -= 1
        d_logits /= T

        # STEP 2: Gradients for W_out and b_out 
        dW_out = dec_out.T @ d_logits # (d_model, tgt_vocab_size)
        db_out = d_logits.sum(axis = 0) # (tgt_vocab_size,)

        # step 3: Gradient flowing back into dec_out
        d_dec_out = d_logits @ self.W_out.T #(T, d_model)

        # step 4: Through layer_norm (approximate as identity)
        d_ffn = d_dec_out

        # step 5 Gradient through W2_dec
        dW2_dec = ffn_hidden.T @ d_ffn # (d_ff, d_model)
        db2_dec = d_ffn.sum(axis = 0) # (d_model,)

        # step 6: Back through ReLU and W1_dec
        d_ffn_hidden = d_ffn @ self.W2_dec.T
        d_ffn_hidden[ffn_hidden <= 0] = 0              # ReLU mask
        dW1_dec = x_before_ffn.T @ d_ffn_hidden        # (d_model, d_ff)
        db1_dec = d_ffn_hidden.sum(axis=0)             # (d_ff,)

        # step 7: Gradient into target embeddings
        # d_ffn_hidden flows back through x_before_ffn into E_tgt
        d_emb = d_ffn_hidden @ self.W1_dec.T           # (T_tgt, d_model)
        dE_tgt = np.zeros_like(self.E_tgt)
        for t, ix in enumerate(tgt_ix):
            dE_tgt[ix] += d_emb[t]

        # step 8: Gradient through cross-attention to enc_out (V path)
        # d_emb is approximately the gradient w.r.t. cross_attn_output
        d_concat = d_emb @ self.W_O_cross.T            # (T_tgt, n_heads * d_k)
        d_enc_out = np.zeros((len(src_ix_saved), self.d_model))
        for h in range(self.n_heads):
            d_head = d_concat[:, h*self.d_k:(h+1)*self.d_k]  # (T_tgt, d_k)
            d_V_h  = cross_attn_ws[h].T @ d_head              # (T_src, d_k)
            d_enc_out += d_V_h @ self.W_V_cross[h].T          # (T_src, d_model)

        # step 9: Gradient through encoder FFN
        d_enc_ffn_hidden = d_enc_out @ self.W2_enc.T          # (T_src, d_ff)
        d_enc_ffn_hidden[enc_ffn_hidden <= 0] = 0              # ReLU mask
        d_enc_x = d_enc_ffn_hidden @ self.W1_enc.T            # (T_src, d_model)

        # step 10: Gradient into source embeddings E_src
        dE_src = np.zeros_like(self.E_src)
        for t, ix in enumerate(src_ix_saved):
            dE_src[ix] += d_enc_x[t]

        # Update Weights
        self.W_out  -= lr * dW_out
        self.b_out  -= lr * db_out
        self.W2_dec -= lr * dW2_dec
        self.b2_dec -= lr * db2_dec
        self.W1_dec -= lr * dW1_dec
        self.b1_dec -= lr * db1_dec
        self.E_tgt  -= lr * dE_tgt
        self.E_src  -= lr * dE_src                             # update source embeddings

        return loss


# ── Training Loop ────────────────────────────────────────────────
model = Transformer(EN_SIZE, FR_SIZE, D_MODEL, D_K, D_FF, N_HEADS)

for epoch in range(EPOCHS):
    total_loss = 0.0
    for src, tgt_in, tgt_out in dataset:
        loss = model.train_step(src, tgt_in, tgt_out, LR)
        total_loss += loss
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(dataset):.4f}")



# ── Translation Test ─────────────────────────────────────────────
def translate(model, src_words, max_len=10):
    src_ix = [en_to_ix[w] for w in src_words] + [en_to_ix["<EOS>"]]
    enc_out = model.encode(src_ix)

    tgt_ix = [fr_to_ix["<SOS>"]]
    for _ in range(max_len):
        probs, _, _, _ = model.forward(src_ix, tgt_ix)
        next_ix = np.argmax(probs[-1])         # take last position's prediction
        if next_ix == fr_to_ix["<EOS>"]:
            break
        tgt_ix.append(next_ix)

    return [ix_to_fr[i] for i in tgt_ix[1:]]  # skip <SOS>

print("\n--- Translation Test ---")
for en, fr in raw_pairs[:4]:
    pred = translate(model, en)
    print(f"EN: {en}")
    print(f"FR expected : {fr}")
    print(f"FR predicted: {pred}")
    print()