import numpy as np

# Research Task: Bidirectional Recurrent Neural Network (BiRNN)
# Goal: Combine a Forward RNN and a Backward RNN to predict a "Missing Character" (Gap Filling)

# --- Task 1: Environment & Hyperparameters ---
# 1. Define the word "hidden"
# 2. Create char_to_ix and ix_to_char mappings
# 3. Set hidden_size = 64
# 4. Define vocab_size


word = "hidden"
chars = list(set(word))

char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

hidden_size = 64
learning_rate = 0.1

class BidirectionalRNN:
    def __init__(self,input_size, hidden_size, output_size):
        self.Wxh_f = np.random.rand(hidden_size,input_size) * 0.01 
        self.Whh_f = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh_f = np.zeros((hidden_size,1))
        
        self.Wxh_b = np.random.randn(hidden_size,input_size) * 0.01
        self.Whh_b = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh_b = np.zeros((hidden_size,1))


        self.Why = np.random.randn(output_size, 2 * hidden_size) * 0.01
        self.by = np.zeros((output_size,1))


    def forward(self,inputs, h_prev_f, h_prev_b):
        xs, hs_f, hs_b, ps = {}, {}, {}, {}
        hs_f[-1] = np.copy(h_prev_f)
        hs_b[len(inputs)] = np.copy(h_prev_b)

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.Wxh_f.shape[1],1))
            xs[t][inputs[t]] = 1

            hs_f[t] = np.tanh(np.dot(self.Wxh_f, xs[t]) + np.dot(self.Whh_f, hs_f[t-1]) + self.bh_f)

        for t in reversed(range(len(inputs))):
            xs[t] = np.zeros((self.Wxh_b.shape[1],1))
            xs[t][inputs[t]] = 1

            hs_b[t] = np.tanh(np.dot(self.Wxh_b, xs[t]) + np.dot(self.Whh_b, hs_b[t+1]) + self.bh_b)

        for t in range(len(inputs)):
            h_combined = np.concatenate((hs_f[t], hs_b[t]), axis=0)

            y = np.dot(self.Why, h_combined) + self.by
            exp_y = np.exp(y - np.max(y))
            ps[t] = exp_y / np.sum(exp_y)

        return xs, hs_f, hs_b, ps
    

    def backward(self, inputs, targets, xs, hs_f, hs_b, ps):
        # --- Step 1: Initializing Gradients ---
        # We need space to store the sum of gradients for each weight matrix.
        # Think of these like 'accumulation buckets'.
        dWxh_f, dWhh_f, dbh_f = np.zeros_like(self.Wxh_f), np.zeros_like(self.Whh_f), np.zeros_like(self.bh_f)
        dWxh_b, dWhh_b, dbh_b = np.zeros_like(self.Wxh_b), np.zeros_like(self.Whh_b), np.zeros_like(self.bh_b)
        dWhy, dby = np.zeros_like(self.Why), np.zeros_like(self.by)
        
        # In BiRNN, the error from the output 'splits' into the Forward and Backward layers.
        # Let's save the split errors for each time step first.
        dh_f_from_output = {}
        dh_b_from_output = {}

        # --- Step 2: Gradients from the Output (Softmax) ---
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # p - y (target)
            
            # The output layer (Why) uses BOTH hidden states [hs_f[t] ; hs_b[t]]
            h_combined = np.concatenate((hs_f[t], hs_b[t]), axis=0)
            dWhy += np.dot(dy, h_combined.T)
            dby += dy

            # Now, split the error signal to send it back to the hidden layers
            # dh = Why^T * dy
            dh_total = np.dot(self.Why.T, dy)
            dh_f_from_output[t] = dh_total[:hidden_size, :]   # First half goes to Forward RNN
            dh_b_from_output[t] = dh_total[hidden_size:, :]   # Second half goes to Backward RNN

        # --- Step 3: Backprop through the Forward RNN (T down to 0) ---
        dh_next_f = np.zeros((hidden_size, 1))
        for t in reversed(range(len(inputs))):
            # Total error for Forward RNN = (Error from Output) + (Error from next timestep)
            dh = dh_f_from_output[t] + dh_next_f
            
            # tanh derivative: (1 - h^2)
            raw_h = (1 - hs_f[t]**2) * dh
            
            # Update gradients (accumulation)
            dbh_f += raw_h
            dWxh_f += np.dot(raw_h, xs[t].T)
            dWhh_f += np.dot(raw_h, hs_f[t-1].T)
            
            # Send error back to the "previous" hidden state (h_t-1)
            dh_next_f = np.dot(self.Whh_f.T, raw_h)

        # --- Step 4: Backprop through the Backward RNN (0 up to T) ---
        dh_next_b = np.zeros((hidden_size, 1))
        for t in range(len(inputs)):
            # Total error for Backward RNN = (Error from Output) + (Error from "next in sequence" but "previous in time")
            dh = dh_b_from_output[t] + dh_next_b
            
            # tanh derivative: (1 - h^2)
            raw_h = (1 - hs_b[t]**2) * dh
            
            # Update gradients (accumulation)
            dbh_b += raw_h
            dWxh_b += np.dot(raw_h, xs[t].T)
            dWhh_b += np.dot(raw_h, hs_b[t+1].T) # Note: t+1 because we are going backwards in time
            
            # Send error back to the "previous" (which is index t+1 for the backward layer)
            dh_next_b = np.dot(self.Whh_b.T, raw_h)

        # --- Step 5: Gradient Clipping ---
        for dparam in [dWxh_f, dWhh_f, dbh_f, dWxh_b, dWhh_b, dbh_b, dWhy, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return dWxh_f, dWhh_f, dbh_f, dWxh_b, dWhh_b, dbh_b, dWhy, dby

    def update(self, gradients, learning_rate):
        # We unpack the tuple we returned from backward()
        dWxh_f, dWhh_f, dbh_f, dWxh_b, dWhh_b, dbh_b, dWhy, dby = gradients
        
        # Standard SGD Update: weight = weight - (learning_rate * gradient)
        self.Wxh_f -= learning_rate * dWxh_f
        self.Whh_f -= learning_rate * dWhh_f
        self.bh_f -= learning_rate * dbh_f
        
        self.Wxh_b -= learning_rate * dWxh_b
        self.Whh_b -= learning_rate * dWhh_b
        self.bh_b -= learning_rate * dbh_b
        
        self.Why -= learning_rate * dWhy
        self.by -= learning_rate * dby

# --- Task 2: Training Loop & Gap Filling Test ---

data = "hidden"
inputs = [char_to_ix[ch] for ch in data]
# For character-level prediction, targets are actually the same as inputs here 
# if we want to learn the representation of the word
targets = inputs 

rnn = BidirectionalRNN(len(chars), hidden_size, len(chars))

print("Training Bidirectional RNN on word 'hidden'...")
for epoch in range(1000):
    h_prev_f = np.zeros((hidden_size, 1))
    h_prev_b = np.zeros((hidden_size, 1))
    
    # Forward Pass
    xs, hs_f, hs_b, ps = rnn.forward(inputs, h_prev_f, h_prev_b)
    
    # Loss (Cross Entropy)
    loss = 0
    for t in range(len(inputs)):
        loss += -np.log(ps[t][targets[t], 0])
    
    # Backward Pass
    grads = rnn.backward(inputs, targets, xs, hs_f, hs_b, ps)
    
    # Update Weights
    rnn.update(grads, learning_rate)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- Task 3: Demonstration: Gap Filling ---
# Let's say we have "h_dden" and we want to fill the gap.
print("\nGap Filling Test: 'h_dden' (index 1 is missing)")
test_word = "h_dden"
# In a real gap-filling scenario, we would use a special [MASK] token.
# Here, we'll just see what the model predicts at index 1 given the context.
test_inputs = [char_to_ix[ch] if ch != '_' else 0 for ch in test_word] # placeholder for mask

_, _, _, ps_test = rnn.forward(inputs, np.zeros((hidden_size, 1)), np.zeros((hidden_size, 1)))
predicted_ix = np.argmax(ps_test[1])
print(f"Prediced character for index 1: '{ix_to_char[predicted_ix]}' (Expected: 'i')")
