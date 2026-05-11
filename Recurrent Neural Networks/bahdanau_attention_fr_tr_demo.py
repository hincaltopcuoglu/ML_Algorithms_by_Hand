import numpy as np

np.random.seed(42)


def build_vocab(sentences, extra_tokens):
    vocab = list(extra_tokens)
    seen = set(vocab)
    for sent in sentences:
        for tok in sent:
            if tok not in seen:
                seen.add(tok)
                vocab.append(tok)
    return vocab


def make_pair(src_tokens, tgt_tokens, src_to_ix, tgt_to_ix):
    src_ix = [src_to_ix[t] for t in src_tokens] + [src_to_ix["<EOS>"]]
    dec_in = [tgt_to_ix["<SOS>"]] + [tgt_to_ix[t] for t in tgt_tokens]
    dec_out = [tgt_to_ix[t] for t in tgt_tokens] + [tgt_to_ix["<EOS>"]]
    return src_ix, dec_in, dec_out


class BahdanauSeq2Seq:
    def __init__(self, src_vocab_size, tgt_vocab_size, embed=32, hidden=48, attn_h=48, clip=5.0, lr=0.002):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed = embed
        self.hidden = hidden
        self.attn_h = attn_h
        self.clip = clip
        self.lr = lr

        g = lambda r, c: np.random.randn(r, c) * 0.01

        self.E_src = g(src_vocab_size, embed)
        self.W_ef = g(hidden, embed)
        self.U_ef = g(hidden, hidden)
        self.b_ef = np.zeros((hidden, 1))
        self.W_eb = g(hidden, embed)
        self.U_eb = g(hidden, hidden)
        self.b_eb = np.zeros((hidden, 1))
        self.W_s0 = g(hidden, hidden)
        self.W_a = g(attn_h, hidden)
        self.U_a = g(attn_h, 2 * hidden)
        self.v_a = g(attn_h, 1)
        self.E_tgt = g(tgt_vocab_size, embed)
        self.W_d = g(hidden, embed)
        self.U_d = g(hidden, hidden)
        self.C_d = g(hidden, 2 * hidden)
        self.b_d = np.zeros((hidden, 1))
        self.W_o = g(tgt_vocab_size, hidden)
        self.b_o = np.zeros((tgt_vocab_size, 1))

        self.params = [
            "E_src", "W_ef", "U_ef", "b_ef", "W_eb", "U_eb", "b_eb",
            "W_s0", "W_a", "U_a", "v_a", "E_tgt", "W_d", "U_d", "C_d",
            "b_d", "W_o", "b_o"
        ]
        self.m = {n: np.zeros_like(getattr(self, n)) for n in self.params}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in self.params}
        self.t = 0

    def encode(self, src_ix):
        Tx = len(src_ix)
        emb = [self.E_src[i].reshape(-1, 1) for i in src_ix]
        hf, hb = [], [None] * Tx
        h = np.zeros((self.hidden, 1))
        for t in range(Tx):
            h = np.tanh(self.W_ef @ emb[t] + self.U_ef @ h + self.b_ef)
            hf.append(h.copy())
        h = np.zeros((self.hidden, 1))
        for t in reversed(range(Tx)):
            h = np.tanh(self.W_eb @ emb[t] + self.U_eb @ h + self.b_eb)
            hb[t] = h.copy()
        annots = [np.concatenate([hf[t], hb[t]], axis=0) for t in range(Tx)]
        s0 = np.tanh(self.W_s0 @ hb[0])
        return annots, s0, hf, hb, emb

    def attend(self, s_prev, annots):
        Tx = len(annots)
        Ws = self.W_a @ s_prev
        scores, tanh_vals = np.zeros(Tx), []
        for j in range(Tx):
            tv = np.tanh(Ws + self.U_a @ annots[j])
            tanh_vals.append(tv)
            scores[j] = (self.v_a.T @ tv).item()
        scores -= scores.max()
        exp_s = np.exp(scores)
        alphas = exp_s / exp_s.sum()
        c_i = sum(alphas[j] * annots[j] for j in range(Tx))
        return c_i, alphas, tanh_vals

    def decode_step(self, y_prev_ix, s_prev, c_i):
        e_y = self.E_tgt[y_prev_ix].reshape(-1, 1)
        pre_s = self.W_d @ e_y + self.U_d @ s_prev + self.C_d @ c_i + self.b_d
        s_i = np.tanh(pre_s)
        logits = self.W_o @ s_i + self.b_o
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        return s_i, probs, e_y

    def forward(self, src_ix, dec_in, dec_out):
        annots, s0, hf, hb, emb = self.encode(src_ix)
        s = s0
        all_s, all_c, all_alphas, all_probs, all_tanh, all_ey = [s0], [], [], [], [], []
        loss = 0.0
        for i in range(len(dec_in)):
            c_i, alphas, tanh_vals = self.attend(s, annots)
            s_new, probs, e_y = self.decode_step(dec_in[i], s, c_i)
            loss += -np.log(probs[dec_out[i], 0] + 1e-9)
            all_s.append(s_new)
            all_c.append(c_i)
            all_alphas.append(alphas)
            all_probs.append(probs)
            all_tanh.append(tanh_vals)
            all_ey.append(e_y)
            s = s_new
        cache = {
            "src_ix": src_ix, "dec_in": dec_in, "dec_out": dec_out, "annots": annots, "hf": hf, "hb": hb, "emb": emb,
            "all_s": all_s, "all_c": all_c, "all_alphas": all_alphas, "all_probs": all_probs, "all_tanh": all_tanh, "all_ey": all_ey
        }
        return loss, cache

    def backward(self, cache):
        src_ix, dec_in, dec_out = cache["src_ix"], cache["dec_in"], cache["dec_out"]
        annots, hf, hb, emb = cache["annots"], cache["hf"], cache["hb"], cache["emb"]
        all_s, all_c, all_alphas = cache["all_s"], cache["all_c"], cache["all_alphas"]
        all_probs, all_tanh, all_ey = cache["all_probs"], cache["all_tanh"], cache["all_ey"]
        Tx, Ty = len(src_ix), len(dec_in)

        grads = {n: np.zeros_like(getattr(self, n)) for n in self.params}
        d_annots = [np.zeros((2 * self.hidden, 1)) for _ in range(Tx)]
        d_s_next = np.zeros((self.hidden, 1))

        for i in reversed(range(Ty)):
            s_prev, s_i = all_s[i], all_s[i + 1]
            c_i, alphas, probs, tanh_v, e_y = all_c[i], all_alphas[i], all_probs[i], all_tanh[i], all_ey[i]
            d_logits = probs.copy()
            d_logits[dec_out[i]] -= 1.0
            grads["W_o"] += d_logits @ s_i.T
            grads["b_o"] += d_logits
            d_s_i = self.W_o.T @ d_logits + d_s_next
            d_pre = (1.0 - s_i ** 2) * d_s_i
            grads["W_d"] += d_pre @ e_y.T
            grads["U_d"] += d_pre @ s_prev.T
            grads["C_d"] += d_pre @ c_i.T
            grads["b_d"] += d_pre
            grads["E_tgt"][dec_in[i]] += (self.W_d.T @ d_pre).squeeze()
            d_s_next = self.U_d.T @ d_pre
            d_c_i = self.C_d.T @ d_pre

            d_alpha = np.array([(annots[j].T @ d_c_i).item() for j in range(Tx)])
            for j in range(Tx):
                d_annots[j] += alphas[j] * d_c_i
            dot = np.dot(alphas, d_alpha).item()
            d_scores = alphas * (d_alpha - dot)
            d_s_from_attn = np.zeros((self.hidden, 1))
            for j in range(Tx):
                d_tanh = self.v_a * d_scores[j]
                grads["v_a"] += tanh_v[j] * d_scores[j]
                d_pre_a = (1.0 - tanh_v[j] ** 2) * d_tanh
                grads["W_a"] += d_pre_a @ s_prev.T
                grads["U_a"] += d_pre_a @ annots[j].T
                d_s_from_attn += self.W_a.T @ d_pre_a
                d_annots[j] += self.U_a.T @ d_pre_a
            d_s_next += d_s_from_attn

        d_pre_s0 = (1.0 - all_s[0] ** 2) * d_s_next
        grads["W_s0"] += d_pre_s0 @ hb[0].T
        d_hb_extra_0 = self.W_s0.T @ d_pre_s0
        d_hf = [d_annots[t][:self.hidden, :] for t in range(Tx)]
        d_hb = [d_annots[t][self.hidden:, :] for t in range(Tx)]
        d_hb[0] += d_hb_extra_0

        d_src_emb = [np.zeros((self.embed, 1)) for _ in range(Tx)]
        d_h_carry = np.zeros((self.hidden, 1))
        for t in reversed(range(Tx)):
            dh = d_hf[t] + d_h_carry
            h_prev_f = hf[t - 1] if t > 0 else np.zeros((self.hidden, 1))
            d_pre = (1.0 - hf[t] ** 2) * dh
            grads["W_ef"] += d_pre @ emb[t].T
            grads["U_ef"] += d_pre @ h_prev_f.T
            grads["b_ef"] += d_pre
            d_src_emb[t] += self.W_ef.T @ d_pre
            d_h_carry = self.U_ef.T @ d_pre

        d_h_carry = np.zeros((self.hidden, 1))
        for t in range(Tx):
            dh = d_hb[t] + d_h_carry
            h_next_b = hb[t + 1] if t < Tx - 1 else np.zeros((self.hidden, 1))
            d_pre = (1.0 - hb[t] ** 2) * dh
            grads["W_eb"] += d_pre @ emb[t].T
            grads["U_eb"] += d_pre @ h_next_b.T
            grads["b_eb"] += d_pre
            d_src_emb[t] += self.W_eb.T @ d_pre
            d_h_carry = self.U_eb.T @ d_pre

        for t in range(Tx):
            grads["E_src"][src_ix[t]] += d_src_emb[t].squeeze()
        for g in grads.values():
            np.clip(g, -self.clip, self.clip, out=g)
        return grads

    def update(self, grads):
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.t += 1
        for n in self.params:
            g = grads[n]
            self.m[n] = b1 * self.m[n] + (1 - b1) * g
            self.v[n] = b2 * self.v[n] + (1 - b2) * g ** 2
            m_hat = self.m[n] / (1 - b1 ** self.t)
            v_hat = self.v[n] / (1 - b2 ** self.t)
            p = getattr(self, n)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def translate(self, src_ix, tgt_to_ix, ix_to_tgt, max_len=12):
        annots, s, _, _, _ = self.encode(src_ix)
        y_prev = tgt_to_ix["<SOS>"]
        out, attn = [], []
        for _ in range(max_len):
            c_i, alphas, _ = self.attend(s, annots)
            s, probs, _ = self.decode_step(y_prev, s, c_i)
            y_ix = int(np.argmax(probs))
            attn.append(alphas)
            if y_ix == tgt_to_ix["<EOS>"]:
                break
            out.append(ix_to_tgt[y_ix])
            y_prev = y_ix
        return out, attn


# Small FR -> TR demo corpus (toy size)
raw_pairs = [
    (["bonjour"], ["merhaba"]),
    (["merci"], ["tesekkurler"]),
    (["oui"], ["evet"]),
    (["non"], ["hayir"]),
    (["salut"], ["selam"]),
    (["je", "vais", "bien"], ["iyiyim"]),
    (["comment", "ca", "va"], ["nasilsin"]),
    (["je", "suis", "etudiant"], ["ben", "ogrenciyim"]),
    (["je", "suis", "fatigue"], ["ben", "yorgunum"]),
]

fr_vocab = build_vocab([p[0] for p in raw_pairs], ["<PAD>", "<EOS>"])
tr_vocab = build_vocab([p[1] for p in raw_pairs], ["<PAD>", "<SOS>", "<EOS>"])
fr_to_ix = {w: i for i, w in enumerate(fr_vocab)}
ix_to_fr = {i: w for i, w in enumerate(fr_vocab)}
tr_to_ix = {w: i for i, w in enumerate(tr_vocab)}
ix_to_tr = {i: w for i, w in enumerate(tr_vocab)}
data = [make_pair(fr, tr, fr_to_ix, tr_to_ix) for fr, tr in raw_pairs]

model = BahdanauSeq2Seq(
    src_vocab_size=len(fr_vocab),
    tgt_vocab_size=len(tr_vocab),
    embed=40,
    hidden=64,
    attn_h=64,
    clip=5.0,
    lr=0.0025,
)

print("=" * 72)
print("Bahdanau Attention FR -> TR toy training")
print("=" * 72)
for epoch in range(1, 3201):
    total = 0.0
    order = np.random.permutation(len(data))
    for k in order:
        src_ix, dec_in, dec_out = data[int(k)]
        loss, cache = model.forward(src_ix, dec_in, dec_out)
        grads = model.backward(cache)
        model.update(grads)
        total += loss
    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | train loss: {total / len(data):.4f}")

print("\nSample translations:")
tests = [
    ["bonjour"],
    ["je", "suis", "etudiant"],
    ["comment", "ca", "va"],
    ["je", "suis", "fatigue"],
]
for fr_tokens in tests:
    src_ix = [fr_to_ix[t] for t in fr_tokens] + [fr_to_ix["<EOS>"]]
    out, attn = model.translate(src_ix, tr_to_ix, ix_to_tr)
    print(f"FR: {' '.join(fr_tokens):<24} -> TR: {' '.join(out)}")

print("\nAttention (FR='je suis etudiant'):")
demo = ["je", "suis", "etudiant"]
demo_ix = [fr_to_ix[t] for t in demo] + [fr_to_ix["<EOS>"]]
pred, attn = model.translate(demo_ix, tr_to_ix, ix_to_tr)
header = "           " + "".join(f"{tok:>12}" for tok in (demo + ["<EOS>"]))
print(header)
print("           " + "-" * (12 * (len(demo) + 1)))
for i, (tok, w) in enumerate(zip(pred, attn)):
    print(f"out[{i}]={tok:<8}" + "".join(f"{x:>12.3f}" for x in w))
