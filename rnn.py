import numpy as np

data = "hello"

chars = list(set(data))

char_to_ix = {ch: i for i,ch in enumerate(chars)}

ix_to_char = {i:ch for i,ch in enumerate(chars)}

vocab_size = len(chars)
hidden_size = 100
seq_length = 5 # length of the sequence "hello"
learning_rate = 1e-1

class SimpleRNN:
    def __init__(self,input_size, hidden_size, output_size):
        # weigths
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01

        #biases
        self.bh = np.zeros((hidden_size,1)) # bias for hidden
        self.by = np.zeros((output_size,1)) # bias for output

    def forward(self,inputs, h_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev) # store the initial state

        for t in range(len(inputs)):
            # 1. One-hot encode input character at time t
            xs[t] = np.zeros((self.Wxh.shape[1],1))
            xs[t][inputs[t]] = 1

            # 2. Update hidden state (memory)
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)

            # 3. compute output (unnormalized log probs)
            ys[t] = np.dot(self.Why, hs[t]) + self.by

            # 4. Compute Softmax (probabilities)
            # exp(y) / sum(exp(y))
            exp_y = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = exp_y / np.sum(exp_y)

        return xs, hs, ps

    def backward(self, inputs, targets, hs, ps, xs):
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])
        loss = 0

        # 2. iterate backwards through time
        for t in reversed(range(len(inputs))):
            # calculate total loss (log probability of correct character)
            loss += -np.log(ps[t][targets[t],0])

            # output gradient (dy = prediction - target)
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1

            # hidden to output weights gradient
            # Formula: dWhy += dy * hs[t]^T
            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            # Gradient into the hidden state (ht)
            # It comes from the output (Why^T * dy) AND the next step (dh_next)
            dh = np.dot(self.Why.T, dy) + dh_next
            
            # backpropagate througt the tanh nonlinearity
            dz = (1 - hs[t] * hs[t]) * dh

            # biases and hidden gradients
            dbh += dz
            dWxh += np.dot(dz, xs[t].T)
            dWhh += np.dot(dz, hs[t-1].T)

            # prepare dh_next for the previous time step
            # Formula: dh_next = Whh^T * dz
            dh_next = np.dot(self.Whh.T, dz)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return loss, dWxh, dWhh, dWhy, dbh, dby

    def sample(self, start_char_ix, h_prev, length):
        # Create one-hot vector for the starting character
        x = np.zeros((self.Wxh.shape[1], 1))
        x[start_char_ix] = 1
        h = h_prev
        ixes = []
        for t in range(length):
            # Forward pass for a single step
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            # Stable Softmax
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)
            # Pick next char
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            # Update for next step
            x = np.zeros_like(x)
            x[ix] = 1
            ixes.append(ix)
        return ixes
            

               
rnn  = SimpleRNN(vocab_size, hidden_size, vocab_size)
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


# Final Prediction Test
# We start with 'h' and want the next 4 letters to potentially get "hello"
sample_ixes = rnn.sample(char_to_ix['h'], h_prev, 4)
txt = 'h' + ''.join(ix_to_char[ix] for ix in sample_ixes)
print(f"\nFinal Prediction: {txt}")
