import numpy as np

# ===========================================================================
# BAHDANAU ATTENTION — Implemented from Scratch (NumPy only)
# Paper: "Neural Machine Translation by Jointly Learning to Align and Translate"
#        Bahdanau, Cho & Bengio  —  ICLR 2015
#
# ARCHITECTURE (exactly as in the paper):
#
#   ENCODER  (Sec 3.2 / Appendix A.2.1):
#       →h_t  = tanh( W_ef · E[x_t]  +  U_ef · →h_{t-1} )
#       ←h_t  = tanh( W_eb · E[x_t]  +  U_eb · ←h_{t+1} )
#       h_t   = [ →h_t ; ←h_t ]          ← annotation for position t   (Eq.7)
#
#   ATTENTION (Sec 3.1 / Appendix A.1.2):
#       e_ij  = v_a^T · tanh( W_a · s_{i-1}  +  U_a · h_j )             (align)
#       α_ij  = exp(e_ij) / Σ_k exp(e_ik)                               (Eq.6)
#       c_i   = Σ_j  α_ij · h_j                                         (Eq.5)
#
#   DECODER  (Sec 3.1 / Appendix A.2.2):
#       s_i   = tanh( W_d · E[y_{i-1}]  +  U_d · s_{i-1}  +  C_d · c_i )
#       p(y_i) = softmax( W_o · s_i )
#       s_0   = tanh( W_s0 · ←h_1 )                 ← init from encoder
#
# TOY TASK: Sequence Reversal
#   Input:  [ a, b, c ]  →  Output: [ c, b, a ]
#
#   Why this task?  The attention MUST learn to look BACKWARDS:
#   - when generating output[0]='c'  →  attend to input position 2 (c)
#   - when generating output[1]='b'  →  attend to input position 1 (b)
#   - when generating output[2]='a'  →  attend to input position 0 (a)
#   The resulting attention matrix should be ANTI-DIAGONAL — perfectly
#   demonstrating that the model has learned to align source & target.
# ===========================================================================


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 ─ VOCABULARY & DATA
# ─────────────────────────────────────────────────────────────────────────────

SRC_CHARS = ['<PAD>', 'a', 'b', 'c', 'd', '<EOS>']
TGT_CHARS = ['<PAD>', 'a', 'b', 'c', 'd', '<SOS>', '<EOS>']

src_to_ix = {c: i for i, c in enumerate(SRC_CHARS)}
ix_to_src = {i: c for i, c in enumerate(SRC_CHARS)}
tgt_to_ix = {c: i for i, c in enumerate(TGT_CHARS)}
ix_to_tgt = {i: c for i, c in enumerate(TGT_CHARS)}

SRC_V = len(SRC_CHARS)   # source vocabulary size (Kx in paper)
TGT_V = len(TGT_CHARS)   # target vocabulary size (Ky in paper)


def make_pair(src_chars, tgt_chars):
    """
    Converts raw character lists into index sequences.
    Source : x_1 ... x_Tx  <EOS>
    Dec-in : <SOS>  y_1 ... y_{Ty-1}   (fed to decoder at each step)
    Dec-out: y_1 ... y_Ty  <EOS>        (what the decoder must predict)
    """
    src_ix  = [src_to_ix[c] for c in src_chars] + [src_to_ix['<EOS>']]
    dec_in  = [tgt_to_ix['<SOS>']] + [tgt_to_ix[c] for c in tgt_chars]
    dec_out = [tgt_to_ix[c] for c in tgt_chars] + [tgt_to_ix['<EOS>']]
    return src_ix, dec_in, dec_out


RAW_PAIRS = [
    (['a', 'b', 'c'], ['c', 'b', 'a']),
    (['a', 'b'],       ['b', 'a']),
    (['b', 'c', 'd'], ['d', 'c', 'b']),
    (['a', 'd'],       ['d', 'a']),
    (['c', 'd'],       ['d', 'c']),
    (['a', 'b', 'd'], ['d', 'b', 'a']),
    (['b', 'c'],       ['c', 'b']),
    (['a', 'c'],       ['c', 'a']),
]

DATA = [make_pair(s, t) for s, t in RAW_PAIRS]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 ─ HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

EMBED   = 12    # m  in paper  — embedding dimension
HIDDEN  = 24    # n  in paper  — encoder/decoder hidden size
ATTN_H  = 24    # n' in paper  — attention hidden size
EPOCHS  = 6000
CLIP    = 5.0

# Adam optimiser settings (faster convergence than plain SGD for this model)
LR      = 0.003
BETA1   = 0.9
BETA2   = 0.999
EPS_OPT = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 ─ MODEL
# ─────────────────────────────────────────────────────────────────────────────

class BahdanauSeq2Seq:
    """
    Full Bahdanau (2015) seq2seq model with additive attention.
    Uses plain tanh RNNs (same as Eq. 1 in paper, GRU variant in Appendix A).
    """

    def __init__(self):
        np.random.seed(42)
        g = lambda r, c: np.random.randn(r, c) * 0.01  # tiny random init

        # ── Source word embedding  E ∈ R^{Kx × m} ──
        self.E_src = g(SRC_V, EMBED)

        # ── Encoder forward RNN  →h_t = tanh(W_ef · e(x_t) + U_ef · →h_{t-1}) ──
        self.W_ef = g(HIDDEN, EMBED)    # (n, m)
        self.U_ef = g(HIDDEN, HIDDEN)   # (n, n)
        self.b_ef = np.zeros((HIDDEN, 1))

        # ── Encoder backward RNN  ←h_t = tanh(W_eb · e(x_t) + U_eb · ←h_{t+1}) ──
        self.W_eb = g(HIDDEN, EMBED)
        self.U_eb = g(HIDDEN, HIDDEN)
        self.b_eb = np.zeros((HIDDEN, 1))

        # ── Decoder initial state  s_0 = tanh(W_s0 · ←h_1) ──
        self.W_s0 = g(HIDDEN, HIDDEN)   # (n, n)

        # ── Attention (alignment model)  e_ij = v_a^T · tanh(W_a·s + U_a·h_j) ──
        self.W_a  = g(ATTN_H, HIDDEN)           # (n', n)   — aligns decoder state
        self.U_a  = g(ATTN_H, 2 * HIDDEN)       # (n', 2n)  — aligns annotation
        self.v_a  = g(ATTN_H, 1)                # (n', 1)

        # ── Target word embedding  E_tgt ∈ R^{Ky × m} ──
        self.E_tgt = g(TGT_V, EMBED)

        # ── Decoder RNN  s_i = tanh(W_d·e(y_{i-1}) + U_d·s_{i-1} + C_d·c_i) ──
        self.W_d  = g(HIDDEN, EMBED)            # (n, m)
        self.U_d  = g(HIDDEN, HIDDEN)           # (n, n)
        self.C_d  = g(HIDDEN, 2 * HIDDEN)       # (n, 2n)
        self.b_d  = np.zeros((HIDDEN, 1))

        # ── Output projection  logit = W_o · s_i ──
        self.W_o  = g(TGT_V, HIDDEN)            # (Ky, n)
        self.b_o  = np.zeros((TGT_V, 1))

        # Collect all parameters (for Adam)
        self._param_names = [
            'E_src', 'W_ef', 'U_ef', 'b_ef',
            'W_eb', 'U_eb', 'b_eb',
            'W_s0',
            'W_a', 'U_a', 'v_a',
            'E_tgt', 'W_d', 'U_d', 'C_d', 'b_d',
            'W_o', 'b_o',
        ]
        # Adam first & second moment estimates
        self._m  = {n: np.zeros_like(getattr(self, n)) for n in self._param_names}
        self._v  = {n: np.zeros_like(getattr(self, n)) for n in self._param_names}
        self._t  = 0   # Adam step counter

    # ─────────────────────────────────────────────────────────────────────────
    # 3a. ENCODER
    # Paper Sec 3.2 / Appendix A.2.1
    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, src_ix):
        """
        src_ix : list[int]  length Tx

        Forward RNN  (left → right):
            →h_t = tanh( W_ef · E[x_t]  +  U_ef · →h_{t-1} )

        Backward RNN  (right → left):
            ←h_t = tanh( W_eb · E[x_t]  +  U_eb · ←h_{t+1} )

        Annotation (Eq.7):
            h_j = [ →h_j ; ←h_j ]    shape (2n, 1)

        Returns
        -------
        annots   : list of (2n, 1) arrays, length Tx
        s0       : (n, 1)  initial decoder state
        hf       : list of (n, 1) forward hidden states
        hb       : list of (n, 1) backward hidden states
        emb      : list of (m, 1) source embeddings
        """
        Tx = len(src_ix)
        emb = [self.E_src[i].reshape(-1, 1) for i in src_ix]  # (m,1) each

        # ── Forward pass  left → right ──
        hf = []
        h  = np.zeros((HIDDEN, 1))
        for t in range(Tx):
            h = np.tanh(self.W_ef @ emb[t] + self.U_ef @ h + self.b_ef)
            hf.append(h.copy())

        # ── Backward pass  right → left ──
        hb = [None] * Tx
        h  = np.zeros((HIDDEN, 1))
        for t in reversed(range(Tx)):
            h = np.tanh(self.W_eb @ emb[t] + self.U_eb @ h + self.b_eb)
            hb[t] = h.copy()

        # ── Concatenate → annotation  h_j = [→h_j ; ←h_j] ──
        annots = [np.concatenate([hf[t], hb[t]], axis=0) for t in range(Tx)]

        # ── Initial decoder state  s_0 = tanh(W_s0 · ←h_1) ──
        # Paper uses the backward hidden state at the FIRST source position
        s0 = np.tanh(self.W_s0 @ hb[0])

        return annots, s0, hf, hb, emb

    # ─────────────────────────────────────────────────────────────────────────
    # 3b. ATTENTION
    # Paper Sec 3.1 / Appendix A.1.2  (Eqs. 5, 6)
    # ─────────────────────────────────────────────────────────────────────────

    def attend(self, s_prev, annots):
        """
        Computes the context vector c_i for decoder step i.

        Alignment model (energy score):
            e_ij = v_a^T · tanh( W_a · s_{i-1}  +  U_a · h_j )

        Attention weights:
            α_ij = softmax( e_ij )   over all j

        Context vector (Eq.5):
            c_i = Σ_j  α_ij · h_j

        Parameters
        ----------
        s_prev  : (n, 1)        decoder hidden state at step i-1
        annots  : list of (2n, 1)  encoder annotations

        Returns
        -------
        c_i       : (2n, 1)  context vector
        alphas    : (Tx,)    attention weights  α_ij
        tanh_vals : list of (n',1)  stored for backward pass
        """
        Tx = len(annots)

        # Pre-compute W_a · s_{i-1} once — reused for every j
        Ws = self.W_a @ s_prev                          # (n', 1)

        tanh_vals = []
        scores    = np.zeros(Tx)
        for j in range(Tx):
            tv         = np.tanh(Ws + self.U_a @ annots[j])   # (n', 1)
            tanh_vals.append(tv)
            scores[j]  = (self.v_a.T @ tv).item()              # scalar e_ij

        # Stable softmax → α_ij  (Eq. 6)
        scores -= scores.max()
        exp_s   = np.exp(scores)
        alphas  = exp_s / exp_s.sum()                   # (Tx,)

        # Context vector  c_i = Σ_j α_ij · h_j  (Eq. 5)
        c_i = sum(alphas[j] * annots[j] for j in range(Tx))    # (2n, 1)

        return c_i, alphas, tanh_vals

    # ─────────────────────────────────────────────────────────────────────────
    # 3c. DECODER STEP
    # Paper Sec 3.1 / Appendix A.2.2
    # ─────────────────────────────────────────────────────────────────────────

    def decode_step(self, y_prev_ix, s_prev, c_i):
        """
        One step of the decoder RNN.

        Decoder hidden state:
            s_i = tanh( W_d · e(y_{i-1})  +  U_d · s_{i-1}  +  C_d · c_i )

        Output distribution:
            p(y_i) = softmax( W_o · s_i )

        Parameters
        ----------
        y_prev_ix : int        index of previous target token y_{i-1}
        s_prev    : (n, 1)
        c_i       : (2n, 1)

        Returns
        -------
        s_i   : (n, 1)        new decoder hidden state
        probs : (Ky, 1)       output probability distribution
        e_y   : (m, 1)        target embedding   (saved for backprop)
        pre_s : (n, 1)        pre-tanh value     (saved for backprop)
        """
        e_y   = self.E_tgt[y_prev_ix].reshape(-1, 1)       # (m, 1)
        pre_s = (self.W_d @ e_y
                 + self.U_d @ s_prev
                 + self.C_d @ c_i
                 + self.b_d)                                 # (n, 1)
        s_i   = np.tanh(pre_s)

        logits = self.W_o @ s_i + self.b_o                  # (Ky, 1)
        logits -= logits.max()
        exp_l  = np.exp(logits)
        probs  = exp_l / exp_l.sum()                         # (Ky, 1)

        return s_i, probs, e_y, pre_s

    # ─────────────────────────────────────────────────────────────────────────
    # 3d. FULL FORWARD PASS
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, src_ix, dec_in, dec_out):
        """
        Full encoder → attention → decoder forward pass.

        src_ix  : list[int]   source indices  x_1 … x_Tx
        dec_in  : list[int]   decoder inputs  <SOS> y_1 … y_{Ty-1}
        dec_out : list[int]   decoder targets y_1 … y_Ty <EOS>

        Returns
        -------
        loss  : scalar  cross-entropy loss
        cache : dict    all intermediate values needed for backward
        """
        annots, s0, hf, hb, emb = self.encode(src_ix)

        Ty = len(dec_in)
        s  = s0

        # Storage for backward pass
        all_s      = [s0]   # all_s[i] = s_{i-1}  (all_s[0]=s0, all_s[Ty]=s_Ty)
        all_c      = []
        all_alphas = []
        all_probs  = []
        all_tanh   = []     # attention tanh values per step
        all_ey     = []     # target embeddings
        all_pres   = []     # pre-tanh decoder states

        loss = 0.0
        for i in range(Ty):
            c_i, alphas, tanh_vals = self.attend(s, annots)
            s_new, probs, e_y, pre_s = self.decode_step(dec_in[i], s, c_i)

            loss += -np.log(probs[dec_out[i], 0] + 1e-9)

            all_c.append(c_i)
            all_alphas.append(alphas)
            all_probs.append(probs)
            all_tanh.append(tanh_vals)
            all_ey.append(e_y)
            all_pres.append(pre_s)

            s = s_new
            all_s.append(s_new)

        cache = dict(
            src_ix=src_ix, dec_in=dec_in, dec_out=dec_out,
            annots=annots, hf=hf, hb=hb, emb=emb,
            all_s=all_s, all_c=all_c, all_alphas=all_alphas,
            all_probs=all_probs, all_tanh=all_tanh,
            all_ey=all_ey, all_pres=all_pres,
        )
        return loss, cache

    # ─────────────────────────────────────────────────────────────────────────
    # 3e. BACKWARD PASS  (BPTT through Decoder + Attention + Encoder)
    # ─────────────────────────────────────────────────────────────────────────

    def backward(self, cache):
        """
        Back-propagates the cross-entropy loss through the entire model.

        Gradient flow summary:
          Loss → Output layer → Decoder RNN → { Attention → Encoder }
                                            ↘
                                             Decoder recurrence (BPTT)

        All intermediate activations are taken from `cache`.
        """
        src_ix     = cache['src_ix']
        dec_in     = cache['dec_in']
        dec_out    = cache['dec_out']
        annots     = cache['annots']
        hf         = cache['hf']
        hb         = cache['hb']
        emb        = cache['emb']
        all_s      = cache['all_s']
        all_c      = cache['all_c']
        all_alphas = cache['all_alphas']
        all_probs  = cache['all_probs']
        all_tanh   = cache['all_tanh']
        all_ey     = cache['all_ey']
        all_pres   = cache['all_pres']

        Tx = len(src_ix)
        Ty = len(dec_in)

        # ── Gradient accumulators ──
        dE_src = np.zeros_like(self.E_src)
        dW_ef  = np.zeros_like(self.W_ef)
        dU_ef  = np.zeros_like(self.U_ef)
        db_ef  = np.zeros_like(self.b_ef)
        dW_eb  = np.zeros_like(self.W_eb)
        dU_eb  = np.zeros_like(self.U_eb)
        db_eb  = np.zeros_like(self.b_eb)
        dW_s0  = np.zeros_like(self.W_s0)
        dW_a   = np.zeros_like(self.W_a)
        dU_a   = np.zeros_like(self.U_a)
        dv_a   = np.zeros_like(self.v_a)
        dE_tgt = np.zeros_like(self.E_tgt)
        dW_d   = np.zeros_like(self.W_d)
        dU_d   = np.zeros_like(self.U_d)
        dC_d   = np.zeros_like(self.C_d)
        db_d   = np.zeros_like(self.b_d)
        dW_o   = np.zeros_like(self.W_o)
        db_o   = np.zeros_like(self.b_o)

        # Annotation gradients — accumulated across all decoder steps
        d_annots = [np.zeros((2 * HIDDEN, 1)) for _ in range(Tx)]

        # Gradient flowing backward through decoder recurrence
        # d_s_next holds  dL/d(all_s[i])  as we iterate backward
        d_s_next = np.zeros((HIDDEN, 1))

        # ── Backward through decoder steps (reverse order) ──
        for i in reversed(range(Ty)):
            s_prev = all_s[i]        # s_{i-1}
            s_i    = all_s[i + 1]    # s_i
            c_i    = all_c[i]
            alphas = all_alphas[i]
            probs  = all_probs[i]
            tanh_v = all_tanh[i]     # list of (n',1) — one per source position
            e_y    = all_ey[i]
            pre_s  = all_pres[i]

            # ── 1. Output layer: softmax cross-entropy gradient ──
            # d_logits = p - one_hot(target)
            d_logits              = probs.copy()
            d_logits[dec_out[i]]  -= 1.0          # (Ky, 1)

            dW_o += d_logits @ s_i.T              # (Ky, n)
            db_o += d_logits                       # (Ky, 1)

            # Gradient into decoder hidden state s_i
            d_s_i  = self.W_o.T @ d_logits        # (n, 1)
            d_s_i += d_s_next                      # + recurrence from step i+1

            # ── 2. Decoder RNN backward: s_i = tanh(pre_s) ──
            d_pre = (1.0 - s_i ** 2) * d_s_i     # through tanh

            dW_d += d_pre @ e_y.T                  # (n, m)
            dU_d += d_pre @ s_prev.T               # (n, n)
            dC_d += d_pre @ c_i.T                  # (n, 2n)
            db_d += d_pre                           # (n, 1)

            dE_tgt[dec_in[i]] += (self.W_d.T @ d_pre).squeeze()   # embedding

            # Gradient flowing to s_{i-1} via U_d recurrence
            d_s_next = self.U_d.T @ d_pre          # reset for previous step

            # Gradient to context vector c_i
            d_c_i = self.C_d.T @ d_pre             # (2n, 1)

            # ── 3. Attention backward ──
            # c_i = Σ_j α_j · h_j
            # d_α_j = h_j^T · d_c_i  (contribution per annotation)
            d_alpha = np.array([
                (annots[j].T @ d_c_i).item() for j in range(Tx)
            ])                                      # (Tx,)

            # Gradient to each annotation from context vector
            for j in range(Tx):
                d_annots[j] += alphas[j] * d_c_i   # (2n, 1)

            # Softmax backward:  d_score_j = α_j (d_α_j - Σ_k α_k d_α_k)
            dot      = np.dot(alphas, d_alpha).item()
            d_scores = alphas * (d_alpha - dot)     # (Tx,)

            # Alignment model backward:
            # score_j = v_a^T · tanh_v[j]
            # tanh_v[j] = tanh(W_a · s_{i-1}  +  U_a · h_j)
            d_s_from_attn = np.zeros((HIDDEN, 1))
            for j in range(Tx):
                # d_tanh = v_a * d_score_j
                d_tanh   = self.v_a * d_scores[j]                    # (n', 1)
                dv_a    += tanh_v[j] * d_scores[j]                   # (n', 1)
                # through tanh
                d_pre_a  = (1.0 - tanh_v[j] ** 2) * d_tanh          # (n', 1)

                dW_a += d_pre_a @ s_prev.T                            # (n', n)
                dU_a += d_pre_a @ annots[j].T                         # (n', 2n)

                d_s_from_attn += self.W_a.T @ d_pre_a                # (n, 1)
                d_annots[j]   += self.U_a.T @ d_pre_a                # (2n, 1)

            # Combine: s_prev receives gradient from RNN recurrence + attention
            d_s_next += d_s_from_attn

        # ── Gradient through  s_0 = tanh(W_s0 · ←h_1) ──
        # After the decoder loop, d_s_next = dL/ds_0
        d_pre_s0 = (1.0 - all_s[0] ** 2) * d_s_next                 # (n, 1)
        dW_s0   += d_pre_s0 @ hb[0].T                                 # (n, n)
        d_hb_extra_0 = self.W_s0.T @ d_pre_s0                        # gradient → ←h_1

        # ── Split annotation gradients into forward & backward components ──
        d_hf = [d_annots[t][:HIDDEN, :]  for t in range(Tx)]   # upper half
        d_hb = [d_annots[t][HIDDEN:, :]  for t in range(Tx)]   # lower half
        d_hb[0] += d_hb_extra_0                                       # add s0 path

        # ── Backward through forward encoder RNN (right → left) ──
        d_src_emb = [np.zeros((EMBED, 1)) for _ in range(Tx)]
        d_h_carry = np.zeros((HIDDEN, 1))
        for t in reversed(range(Tx)):
            dh         = d_hf[t] + d_h_carry
            h_prev_f   = hf[t - 1] if t > 0 else np.zeros((HIDDEN, 1))
            d_pre      = (1.0 - hf[t] ** 2) * dh

            dW_ef      += d_pre @ emb[t].T
            dU_ef      += d_pre @ h_prev_f.T
            db_ef      += d_pre
            d_src_emb[t] += self.W_ef.T @ d_pre
            d_h_carry   = self.U_ef.T @ d_pre

        # ── Backward through backward encoder RNN (left → right in index) ──
        # Backward RNN processes t = Tx-1 ... 0 in forward.
        # Its recurrence: hb[t] depends on hb[t+1].
        # Backprop order: t = 0, 1, ..., Tx-1 (reverse of forward processing).
        d_h_carry = np.zeros((HIDDEN, 1))
        for t in range(Tx):
            dh         = d_hb[t] + d_h_carry
            h_next_b   = hb[t + 1] if t < Tx - 1 else np.zeros((HIDDEN, 1))
            d_pre      = (1.0 - hb[t] ** 2) * dh

            dW_eb      += d_pre @ emb[t].T
            dU_eb      += d_pre @ h_next_b.T
            db_eb      += d_pre
            d_src_emb[t] += self.W_eb.T @ d_pre
            d_h_carry   = self.U_eb.T @ d_pre

        # ── Backward through source embeddings ──
        for t in range(Tx):
            dE_src[src_ix[t]] += d_src_emb[t].squeeze()

        # ── Gradient clipping ──
        grads = dict(
            E_src=dE_src, W_ef=dW_ef, U_ef=dU_ef, b_ef=db_ef,
            W_eb=dW_eb, U_eb=dU_eb, b_eb=db_eb,
            W_s0=dW_s0,
            W_a=dW_a, U_a=dU_a, v_a=dv_a,
            E_tgt=dE_tgt, W_d=dW_d, U_d=dU_d, C_d=dC_d, b_d=db_d,
            W_o=dW_o, b_o=db_o,
        )
        for g in grads.values():
            np.clip(g, -CLIP, CLIP, out=g)

        return grads

    # ─────────────────────────────────────────────────────────────────────────
    # 3f. ADAM OPTIMIZER UPDATE
    # ─────────────────────────────────────────────────────────────────────────

    def update(self, grads):
        self._t += 1
        t = self._t
        for name in self._param_names:
            g          = grads[name]
            self._m[name] = BETA1 * self._m[name] + (1 - BETA1) * g
            self._v[name] = BETA2 * self._v[name] + (1 - BETA2) * g ** 2
            m_hat      = self._m[name] / (1 - BETA1 ** t)
            v_hat      = self._v[name] / (1 - BETA2 ** t)
            param      = getattr(self, name)
            param     -= LR * m_hat / (np.sqrt(v_hat) + EPS_OPT)

    # ─────────────────────────────────────────────────────────────────────────
    # 3g. GREEDY INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def translate(self, src_ix, max_len=10):
        """
        Greedy decoding: at each step predict the most probable target token.

        Returns
        -------
        output  : list of predicted character strings
        attn    : list of attention weight arrays (Ty × Tx) for visualisation
        """
        annots, s0, _, _, _ = self.encode(src_ix)
        s      = s0
        y_prev = tgt_to_ix['<SOS>']
        output = []
        attn   = []

        for _ in range(max_len):
            c_i, alphas, _ = self.attend(s, annots)
            s, probs, _, _ = self.decode_step(y_prev, s, c_i)
            y_ix           = int(np.argmax(probs))
            attn.append(alphas)
            if y_ix == tgt_to_ix['<EOS>']:
                break
            output.append(ix_to_tgt[y_ix])
            y_prev = y_ix

        return output, attn


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 ─ TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

model = BahdanauSeq2Seq()

print("=" * 65)
print("  Bahdanau Attention Seq2Seq — Sequence Reversal Task")
print(f"  Encoder: BiRNN  |  EMBED={EMBED}  HIDDEN={HIDDEN}  ATTN={ATTN_H}")
print("=" * 65)

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for src_ix, dec_in, dec_out in DATA:
        loss, cache = model.forward(src_ix, dec_in, dec_out)
        grads       = model.backward(cache)
        model.update(grads)
        total_loss += loss

    if epoch % 500 == 0:
        avg = total_loss / len(DATA)
        print(f"  Epoch {epoch:5d}  |  avg loss: {avg:.4f}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 ─ TRANSLATION TEST
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  Translation Results")
print("=" * 65)

test_cases = [
    (['a', 'b', 'c'], 'cba'),    # seen
    (['b', 'c', 'd'], 'dcb'),    # seen
    (['a', 'd'],       'da'),    # seen
    (['c', 'b', 'a'], 'abc'),    # reversed → unseen order
    (['d', 'c', 'b'], 'bcd'),    # unseen
]

all_correct = 0
for src_chars, expected in test_cases:
    src_ix = [src_to_ix[c] for c in src_chars] + [src_to_ix['<EOS>']]
    output, _ = model.translate(src_ix)
    result     = ''.join(output)
    ok         = result == expected
    all_correct += int(ok)
    mark       = "OK" if ok else "FAIL"
    print(f"  [{mark}]  {''.join(src_chars)} -> {result}   (expected: {expected})")

print(f"\n  Accuracy: {all_correct}/{len(test_cases)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 ─ ATTENTION MATRIX VISUALISATION
# Paper Fig. 3: rows = decoder output steps, cols = encoder input positions
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("  Attention Weights  alpha_ij  for  'abc' -> 'cba'")
print("  rows = decoder output steps  |  cols = encoder input positions")
print("  High weight (near 1.0) -> model is ATTENDING to that source word")
print("=" * 65)

demo_src = ['a', 'b', 'c']
demo_ix  = [src_to_ix[c] for c in demo_src] + [src_to_ix['<EOS>']]
out_tok, attn_matrix = model.translate(demo_ix)

src_labels = demo_src + ['<EOS>']
header     = "          " + "".join(f"{c:>8}" for c in src_labels)
print(header)
print("          " + "-" * (8 * len(src_labels)))

for i, (tok, weights) in enumerate(zip(out_tok, attn_matrix)):
    bar_row = ""
    for w in weights:
        blocks = int(round(w * 7))
        bar_row += f"{'#' * blocks:>7} "
    print(f"  out[{i}]={tok}  {bar_row}")
    print("          " + "".join(f"{w:>8.3f}" for w in weights))
    print()

print("Expected (ideal anti-diagonal):")
print("  out[0]=c  ->  strong weight on 'c' (position 2)")
print("  out[1]=b  ->  strong weight on 'b' (position 1)")
print("  out[2]=a  ->  strong weight on 'a' (position 0)")

print()
print("=" * 65)
print("  Attention Weights  alpha_ij  for  'abcd' reversed  (out-of-vocab test)")
print("=" * 65)

demo_src2 = ['a', 'b', 'c', 'd']
demo_ix2  = [src_to_ix[c] for c in demo_src2] + [src_to_ix['<EOS>']]
out_tok2, attn2 = model.translate(demo_ix2)
src_labels2 = demo_src2 + ['<EOS>']

print("          " + "".join(f"{c:>8}" for c in src_labels2))
print("          " + "-" * (8 * len(src_labels2)))
for i, (tok, weights) in enumerate(zip(out_tok2, attn2)):
    print(f"  out[{i}]={tok}  " + "".join(f"{w:>8.3f}" for w in weights))
print(f"\n  Full output: {''.join(out_tok2)}")
