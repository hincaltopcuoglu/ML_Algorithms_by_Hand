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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def forward(self, x_t, h_prev):
        r_t = self.sigmoid(self.W_r @ x_t + self.U_r @ h_prev)
        z_t = self.sigmoid(self.W_z @ x_t + self.U_z @ h_prev)
        h_hat_t = np.tanh(self.W_h @ x_t + self.U_h @ (r_t * h_prev))
        h_t = (1 - z_t) * h_prev + z_t * h_hat_t 

        return h_t



class Encoder:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.gru = GRU(input_size, hidden_size) # Encoder owns a GRU

    def forward(self, inputs):
        # inputs = list of one-hot vectors, one per word

        h = np.zeros((self.hidden_size,1)) # start with empty memory

        for x_t in inputs:
            h = self.gru.forward(x_t, h)
        
        return h  # this is the context vector c

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
            h = self.gru.forward(x,h)

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
encoder = Encoder(src_vocab_size, hidden_size)
decoder = Decoder(hidden_size, tgt_vocab_size)

# 2. Prepare input sentence as list of one-hot vector
sentence = ['the', 'economy', 'has', 'slowed']

inputs = []
for word in sentence:
    x = np.zeros((src_vocab_size,1))
    x[src_to_ix[word]] = 1
    inputs.append(x)

# 3.Encode the sentence -> get context vector
context = encoder.forward(inputs)

# 4. Decode the context vector -> get translation
output_ids = decoder.forward(context, tgt_vocab_size)

# 5. Conver output indices back to words
output_words = [ix_to_tgt[ix] for ix in output_ids]
print("Input: ", sentence)
print("Output: ", output_words)