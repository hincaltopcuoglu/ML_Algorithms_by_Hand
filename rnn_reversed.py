import numpy as np

# Research: Training RNN in Reverse Order
# Purpose: Compare standard sequential training vs. reverse order training

# --- Task 1: Environment Setup ---
# 1. Import numpy
# 2. Define the training string "hello"
# 3. Create the character mapping dictionaries (char_to_ix, ix_to_char) from the reversed string "olleh"

data = "hello"[::-1]


reversed_chars = list(set(data))

vocab_size = len(reversed_chars)

char_to_ix = {ch:i for i,ch in enumerate(reversed_chars)}
ix_to_chars = {i:ch for i,ch in enumerate(reversed_chars)}

hidden_size = 100
learning_rate = 0.1

class ReverseRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01

        self.bh = np.zeros((hidden_size,1))
        self.by = np.zeros((vocab_size,1))

    def forward(self, inputs, h_prev):
        xs, hs, ps = {}, {}, {}
        hs[-1] = np.copy(h_prev)

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.Wxh.shape[1], 1))
            xs[t][inputs[t]] = 1

            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)

            # raw scores (logits)
            y = np.dot(self.Why, hs[t]) + self.by

            # softmax
            exp_y = np.exp(y - np.max(y))
            ps[t] = exp_y / np.sum(exp_y)
        
        return xs, hs, ps
    

    def backward(self,inputs, targets, xs, hs, ps):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(hs[0])
        
        loss = 0
        for t in reversed(range(len(inputs))):
            loss += -np.log(ps[t][targets[t],0])

            dy = np.copy(ps[t])
            dy[targets[t]] -= 1

            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            dh = np.dot(self.Why.T,dy) + dh_next

            dz = (1- hs[t] * hs[t]) * dh
            dWxh += np.dot(dz, xs[t].T)
            dWhh += np.dot(dz , hs[t-1].T)

            dh_next = np.dot(self.Whh.T,dz)
        
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        return loss, dWxh, dWhh, dWhy, dbh, dby
    



    def sample(self, start_char_ix, h_prev, length):
        x = np.zeros((self.Wxh.shape[1],1))
        x[start_char_ix] = 1
        h = h_prev
        y = np.dot(self.Why, h) + self.by
        ixes = []
        
        for t in range(length):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h)+ self.bh)
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)

            ix = np.random.choice(range(vocab_size), p=p.ravel())

            x = np.zeros_like(x)
            x[ix] = 1

            ixes.append(ix)

        return ixes


rnn  = ReverseRNN(vocab_size, hidden_size, vocab_size)
h_prev = np.zeros((hidden_size,1))
inputs = [char_to_ix[c] for c in data[:-1]]
targets = [char_to_ix[c] for c in data[1:]]

for epoch in range(1001):
    # step 1 forward pass
    xs, hs, ps = rnn.forward(inputs,h_prev)
 
    # step 2: BPTT
    loss, dWxh, dWhh, dWhy, dbh, dby = rnn.backward(inputs, targets, xs, hs, ps)

    # step 3 : update weights (gradient descent)
    rnn.Wxh -= learning_rate * dWxh
    rnn.Whh -= learning_rate * dWhh
    rnn.Why -= learning_rate * dWhy
    rnn.bh -= learning_rate * dbh
    rnn.by -= learning_rate * dby

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Final Prediction Test for Reversed RNN
print("\nReverse RNN Prediction (starting with 'o'):")
h = np.zeros((hidden_size, 1))
start_char = 'o'
print(start_char, end="")
# Sample 4 more characters
result_indices = rnn.sample(char_to_ix[start_char], h, 4)
for idx in result_indices:
    print(ix_to_chars[idx], end="")
print()