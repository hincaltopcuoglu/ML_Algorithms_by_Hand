"""
This code is a implementation of the encoder-decoder architecture proposed 
in the paper "On the Properties of Neural Machine Translation: Encoder-Decoder Approaches" by Cho, Merrienboer, 
Bahdanaue and Bengio 2014.

The paper proposes a encoder-decoder architecture for neural machine translation with GRU units.

We will implement the encoder-decoder architecture with GRU units and show the difrence with Vanilla RNN and GRU units.

This approach is reminiscent to the LSTM's architecture.

We will write it in Numpy only.
"""

import numpy as np

hidden_size = 256
learning_rate = 0.001

SRC  = ["growth", "has", "slowed", "the", "economy", "<START>", "<END>"]
TGT = ["croissance", "a", "ralenti", "la", "economie", "<START>", "<END>"]


src_to_ix = {c: i for i,c in enumerate(SRC)}
tgt_to_ix = {c:i for i,c in enumerate(TGT)}

ix_to_src = {i:c for i,c in enumerate(SRC)}
ix_to_tgt = {i:c for i,c in enumerate(TGT)}

src_vocab_size = len(SRC)
tgt_vocab_size = len(TGT)

# GRU Unit

# Matrix	For	                        Shape

# W_r     reset gate, input side      (hidden_size, input_size)

# U_r     reset gate, hidden side     (hidden_size, hidden_size)

# W_z     update gate, input side     (hidden_size, input_size)

# U_z     update gate, hidden side    (hidden_size, hidden_size)

# W_h     candidate, input side       (hidden_size, input_size)

# U_h     candidate, hidden side      (hidden_size, hidden_size)

class GRU:
    def __init__(self, input_size, hidden_size):
        # Reset Gate Weights
        self.W_r = np.random.randn(hidden_size, input_size) * 0.01
        self.U_r = np.random.randn(hidden_size, hidden_size) * 0.01

        self.W_z = np.random.randn(hidden_size, input_size) * 0.01
        self.U_z = np.random.randn(hidden_size, hidden_size) * 0.01

        self.W_h = np.random.randn(hidden_size, input_size) * 0.01
        self.U_h = np.random.randn(hidden_size, hidden_size) * 0.01

        # Gradient accumulators (reset before each training step)
        self.dW_r = np.zeros_like(self.W_r)
        self.dU_r = np.zeros_like(self.U_r)
        self.dW_z = np.zeros_like(self.W_z)
        self.dU_z = np.zeros_like(self.U_z)
        self.dW_h = np.zeros_like(self.W_h)
        self.dU_h = np.zeros_like(self.U_h)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def forward(self, x_t, h_prev):
        r_t = self.sigmoid(self.W_r @ x_t + self.U_r @ h_prev)
        z_t = self.sigmoid(self.W_z @ x_t + self.U_z @ h_prev)
        h_hat_t = np.tanh(self.W_h @ x_t + self.U_h @ (r_t * h_prev))
        h_t = (1 - z_t) * h_prev + z_t * h_hat_t

        cache = (x_t, h_prev, r_t, z_t, h_hat_t)

        return h_t, cache

    def backward(self, dh_t, cache):
        x_t, h_prev, r_t, z_t, h_hat_t = cache

        # step 1: Through the final interpoliation h_t = (1-z)*h_prev + z*h_hat
        dh_hat = dh_t * z_t
        dz = dh_t * (h_hat_t - h_prev)
        dh_prev = dh_t * (1 - z_t)

        # step 2: through tanh (candidate hidden state)
        d_tanh = dh_hat  * (1 - h_hat_t ** 2)
        self.dW_h += d_tanh @ x_t.T
        self.dU_h += d_tanh @ (r_t * h_prev).T
        dr = (self.U_h.T @ d_tanh) * h_prev
        dh_prev += (self.U_h.T @ d_tanh) * r_t

        # Step 3: Through sigmoid (update gate z)
        d_sig_z = dz * z_t * (1 - z_t)
        self.dW_z += d_sig_z @ x_t.T
        self.dU_z += d_sig_z @ h_prev.T
        dh_prev += self.U_z.T @ d_sig_z

        # Step 4: Through sigmoid (reset gate r)
        d_sig_r = dr * r_t * (1 - r_t)
        self.dW_r += d_sig_r @ x_t.T
        self.dU_r += d_sig_r @ h_prev.T
        dh_prev += self.U_r.T @ d_sig_r

        return dh_prev



class Encoder:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.gru = GRU(input_size, hidden_size) # Encoder owns a GRU

    def forward(self, inputs):
        # inputs = list of one-hot vectors, one per word

        h = np.zeros((self.hidden_size,1)) # start with empty memory
        caches = []

        for x_t in inputs:
            h, cache = self.gru.forward(x_t, h)
            caches.append(cache)
        
        return h, caches  # this is the context vector c

    # what it does ?
    # Word 1: "growth"  → GRU → h₁
    # Word 2: "has"     → GRU → h₂   (remembers "growth")
    # Word 3: "slowed"  → GRU → h₃   (remembers "growth has")
                              # ↓
                       # return h₃ = context vector c



class Decoder:
    # What is input_size for the decoder's GRU? (Hint: the input is a target language word, 
    # encoded as one-hot of size output_size)
    # What shape is W_out?
    # What shape is b_out?

    # Input size for decoder's GRU: The decoder feeds target words as input, 
    # each word is a one-hot vector of size output_size. So  → input_size = output_size 

    # Shape of W_out: It maps hidden_state → word scores:

    # Input: hidden state of size hidden_size
    # Output: score for each word in target vocab, size output_size
    # So shape = (output_size, hidden_size)

    def __init__(self, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.gru = GRU(output_size, hidden_size)
        self.W_out = np.random.randn(output_size, hidden_size) * 0.01
        self.b_out = np.zeros((output_size,1))



    def forward(self, context, output_size, max_len=10):
        # Start:  h = context vector c  (from encoder)
        # x = one-hot of <START> token

        # Step 1: h = GRU(x, h)
        # scores = W_out @ h + b_out
        # probs = softmax(scores)
        # predicted_word = argmax(probs)
        # x = one-hot of predicted_word   ← feed output back as next input

        # Step 2: h = GRU(x, h)
        # ... repeat until predicted_word == <END>

        # context = encoder's final h (shape: hidden_size, 1)
        # output_size = size of target vocab
        # max_len = stop after this many words (safety limit)

        h = context # decoder starts from encoder's context vector

        # Create one-hot vector for <START> token
        x = np.zeros((output_size,1))
        x[tgt_to_ix["<START>"]] = 1

        output_ids = [] # we'll collect predicted word indices here

        for _ in range(max_len):

            # Step 1: Run the GRU
            h, cache = self.gru.forward(x,h)

            # Step 2: Compute word scores
            scores = self.W_out @ h + self.b_out
            
            # Step 3: Softmax -> probabilities
            exp_s = np.exp(scores - np.max(scores))
            probs = exp_s / np.sum(exp_s)

            # Step 4 : Pick the most likeliy word
            predicted_ix = np.argmax(probs)
            output_ids.append(predicted_ix)

            # Step 5: Stop if <END> is predicted
            if predicted_ix == tgt_to_ix["<END>"]:
                break

            # Steap 6: Feed predicted word as next input
            x = np.zeros((output_size, 1))
            x[predicted_ix] = 1
        
        return output_ids


# --- Putting it all together ---

# 1. Create encoder and decoder
# encoder = Encoder(src_vocab_size, hidden_size)
# decoder = Decoder(hidden_size, tgt_vocab_size)

# 2. Prepare input sentence as list of one-hot vector
# sentence = ['the', 'economy', 'has', 'slowed']

# inputs = []
# for word in sentence:
#     x = np.zeros((src_vocab_size,1))
#     x[src_to_ix[word]] = 1
#     inputs.append(x)

# 3.Encode the sentence -> get context vector
# context = encoder.forward(inputs)

# 4. Decode the context vector -> get translation
# output_ids = decoder.forward(context, tgt_vocab_size)

# 5. Conver output indices back to words
# output_words = [ix_to_tgt[ix] for ix in output_ids]
# print("Input: ", sentence)
# print("Output: ", output_words)


#################### -------------------------- ##################



# Training loop :
# For each epoch:
#     For each (english_sentence, french_sentence) pair:

#         # ENCODER forward
#         context = encoder.forward(src_inputs)

#         # DECODER forward with teacher forcing
#         loss = 0
#         for each target word:
#             h = gru(correct_previous_word, h)
#             scores → softmax → probabilities
#             loss += cross_entropy(probs, correct_next_word)

#         # BACKWARD PASS
#         backprop through decoder
#         backprop through encoder

#         # UPDATE WEIGHTS
#         all_weights -= learning_rate * gradients


# Training data: (english_sentence, french_sentence)
train_data = [
    (["the", "economy", "has", "slowed"],
     ["<START>", "la", "economie", "a", "ralenti", "<END>"]),

    (["growth", "has", "slowed"],
     ["<START>", "croissance", "a", "ralenti", "<END>"]),
]

def words_to_onehots(words, word_to_ix, vocab_size):
    result = []
    for w in words:
        x = np.zeros((vocab_size, 1))
        x[word_to_ix[w]] = 1
        result.append(x)
    return result

# Traning loop

encoder = Encoder(src_vocab_size, hidden_size)
decoder = Decoder(hidden_size, tgt_vocab_size)

for epoch in range(500):
    total_loss = 0

    for src_words, tgt_words in train_data:

        # Prepare inputs
        src_inputs = words_to_onehots(src_words, src_to_ix, src_vocab_size)
        # decoder input: <START> to second to last word
        dec_inputs = words_to_onehots(tgt_words[:-1], tgt_to_ix, tgt_vocab_size)
        # targets :second word to <END>
        dec_targets = [tgt_to_ix[w] for w in tgt_words[1:]]

        # -- Encoder forward --
        context, enc_caches = encoder.forward(src_inputs)

        # -- Decoder forward with forcing --
        h = context
        dec_hs = [h]
        dec_caches = []
        prob_list = []

        for x_t in dec_inputs:
            h, cache = decoder.gru.forward(x_t, h)
            dec_hs.append(h)
            dec_caches.append(cache)
            scores = decoder.W_out @ h + decoder.b_out
            exp_s = np.exp(scores - np.max(scores))
            probs = exp_s / np.sum(exp_s)
            prob_list.append(probs)

        
        # compute loss 
        loss = 0
        for t, target_ix in enumerate(dec_targets):
            loss += -np.log(prob_list[t][target_ix, 0] + 1e-9)
        total_loss += loss

        # -- decoder backward --
        # reset decoder GRU gradients
        decoder.gru.dW_r[:] = 0; decoder.gru.dU_r[:] = 0
        decoder.gru.dW_z[:] = 0; decoder.gru.dU_z[:] = 0
        decoder.gru.dW_h[:] = 0; decoder.gru.dU_h[:] = 0
        dW_out = np.zeros_like(decoder.W_out)
        db_out = np.zeros_like(decoder.b_out)
        
        dh = np.zeros((hidden_size, 1))

        for t in reversed(range(len(dec_targets))):
            dy = np.copy(prob_list[t])
            dy[dec_targets[t]] -= 1
            dW_out += dy @ dec_hs[t+1].T
            db_out += dy
            dh_from_loss = decoder.W_out.T @ dy
            dh = dh + dh_from_loss
            dh = decoder.gru.backward(dh, dec_caches[t])

        
        # --- ENCODER BACKWARD ---
        encoder.gru.dW_r[:] = 0; encoder.gru.dU_r[:] = 0
        encoder.gru.dW_z[:] = 0; encoder.gru.dU_z[:] = 0
        encoder.gru.dW_h[:] = 0; encoder.gru.dU_h[:] = 0

        dh = dh  # gradient flows from decoder into encoder's last hidden state
        for cache in reversed(enc_caches):
            dh = encoder.gru.backward(dh, cache)
        # --- CLIP GRADIENTS ---
        for grad in [encoder.gru.dW_r, encoder.gru.dU_r,
                     encoder.gru.dW_z, encoder.gru.dU_z,
                     encoder.gru.dW_h, encoder.gru.dU_h,
                     decoder.gru.dW_r, decoder.gru.dU_r,
                     decoder.gru.dW_z, decoder.gru.dU_z,
                     decoder.gru.dW_h, decoder.gru.dU_h,
                     dW_out, db_out]:
            np.clip(grad, -5, 5, out=grad)
        # --- UPDATE WEIGHTS ---
        lr = learning_rate
        encoder.gru.W_r -= lr * encoder.gru.dW_r
        encoder.gru.U_r -= lr * encoder.gru.dU_r
        encoder.gru.W_z -= lr * encoder.gru.dW_z
        encoder.gru.U_z -= lr * encoder.gru.dU_z
        encoder.gru.W_h -= lr * encoder.gru.dW_h
        encoder.gru.U_h -= lr * encoder.gru.dU_h
        decoder.gru.W_r -= lr * decoder.gru.dW_r
        decoder.gru.U_r -= lr * decoder.gru.dU_r
        decoder.gru.W_z -= lr * decoder.gru.dW_z
        decoder.gru.U_z -= lr * decoder.gru.dU_z
        decoder.gru.W_h -= lr * decoder.gru.dW_h
        decoder.gru.U_h -= lr * decoder.gru.dU_h
        decoder.W_out -= lr * dW_out
        decoder.b_out -= lr * db_out
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")



# --- Translation Test ---
def translate(encoder, decoder, src_words, max_len=10):
    src_inputs = words_to_onehots(src_words, src_to_ix, src_vocab_size)
    context, _ = encoder.forward(src_inputs)
    output_ids = decoder.forward(context, tgt_vocab_size, max_len)
    return [ix_to_tgt[ix] for ix in output_ids]

print("\n--- Translation Test After Training ---")
test1 = ["the", "economy", "has", "slowed"]
test2 = ["growth", "has", "slowed"]

print(f"Input:    {test1}")
print(f"Output:   {translate(encoder, decoder, test1)}")
print()
print(f"Input:    {test2}")
print(f"Output:   {translate(encoder, decoder, test2)}")