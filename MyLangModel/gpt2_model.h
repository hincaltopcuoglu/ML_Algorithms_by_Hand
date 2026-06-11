#define GPT2_MODEL_H

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

// ====================================================================
// Embedding Katmanı
// ====================================================================
// Convert tokens to dense vector like a lookup table

struct Embedding {
    std::vector<std::vector<float>> weights;  // vocab_size x embedding_dim
    std::vector<std::vector<float>> weight_gradients;
    int vocab_size, embedding_dim;

    Embedding(int vocab_size, int embedding_dim) 
        : vocab_size(vocab_size), embedding_dim(embedding_dim) {
        weights.assign(vocab_size, std::vector<float>(embedding_dim, 0.0f));
        weight_gradients.assign(vocab_size, std::vector<float>(embedding_dim, 0.0f));
        
        // Xavier initialization: Small random values
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(embedding_dim));
        
        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                weights[i][j] = dist(gen);
            }
        }
    }


    // Convert token_id into embedding vector
    std::vector<float> embed(int token_id) const {
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error("Invalid token ID");
        }
        return weights[token_id];

    }

    // ========= BACKWARD ===========
    // input_grad: gradients in embedding space (seq_len x embedding_dim)
    // token_ids : input tokens
    // output_grad: gradients in input space (seq_len x embedding_dim)
    std::vector<std::vector<float>> backward(
        const std::vector<std::vector<float>>& input_grad,
        const std::vector<int>& token_ids) {
        
        // Calculate Weight gradients: grad_w[token_id] += grad_embedding
        for (size_t i = 0; i < token_ids.size(); i++) {
            int token_id = token_ids[i];
            for(int j = 0; j < embedding_dim; j++) {
                weight_gradients[token_id][j] += input_grad[i][j];
            }
        }

        // Input gradients: grad_token = grad @ W^T
        // The gradient returs to embedding case is same
        std::vector<std::vector<float>> output_grad = input_grad;

        return output_grad;
    }

    // Weight update: w = w -learning_rate * grad_w
    void update_weights(float learning_rate) {
        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                weights[i][j] -= learning_rate * weight_gradients[i][j];
                weight_gradients[i][j] = 0.0f;
            }
        }
    }
};

// ============================================================================
// AKTIVASYON FONKSİYONLARI
// ============================================================================

// ReLU: max(0,x)
inline float relu(float x) {
    return std::max(0.0f, x);
}

// Softmax: normlize exponential values
std::vector<float> softmax(const std::vector<float>& scores) {
    if (scores.empty()) {
        throw std::runtime_error("Cannot compute softmax of empty vector");
    }
    
    // Numerical stability: minus max value
    float max_score = *std::max_element(scores.begin(), scores.end());
    
    std::vector<float> exp_scores(scores.size());
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < scores.size(); i++) {
        exp_scores[i] = std::exp(scores[i] - max_score);
        sum_exp += exp_scores[i];
    }
    
    // Division by zero check
    if (sum_exp == 0.0f || !std::isfinite(sum_exp)) {
        // Eğer sayısal stabilite sorunu varsa, uniform distribution dön
        std::vector<float> uniform(scores.size(), 1.0f / scores.size());
        return uniform;
    }
    
    std::vector<float> probs(scores.size());
    for (size_t i = 0; i < scores.size(); i++) {
        probs[i] = exp_scores[i] / sum_exp;
    }
    
    return probs;
}


// ============================================================================
// SELF-ATTENTION MECHANISM
// ============================================================================
// Transformer: each position can look another position

struct SelfAttention {
    int embedding_dim;
    std::vector<std::vector<float>> W_q, W_k, W_v; // Query, Key, Value matrices
    std::vector<std::vector<float>> W_q_grad, W_k_grad, W_v_grad;
    float scale;

    SelfAttention(int embedding_dim)
        : embedding_dim(embedding_dim), scale(1.0f / std::sqrt(embedding_dim)) {
        
        W_q.assign(embedding_dim, std::vector<float>(embedding_dim, 0.0f));
        W_k.assign(embedding_dim, std::vector<float>(embedding_dim, 0.0f));
        W_v.assign(embedding_dim, std::vector<float>(embedding_dim, 0.0f));
        W_q_grad.assign(embedding_dim, std::vector<float>(embedding_dim, 0.0f));
        W_k_grad.assign(embedding_dim, std::vector<float>(embedding_dim, 0.0f));
        W_v_grad.assign(embedding_dim, std::vector<float>(embedding_dim, 0.0f)); 
        
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.01f);

        for (int i = 0; i < embedding_dim; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                W_q[i][j] = dist(gen);
                W_k[i][j] = dist(gen);
                W_v[i][j] = dist(gen);
            }
        } 
    }

    // forward + intermedite values storage
    struct AttentionOutput {
        std::vector<std::vector<float>> output;
        std::vector<std::vector<float>> queries, keys, values;
        std::vector<std::vector<float>> attention_weights;
        std::vector<std::vector<float>> input;
    };

    AttentionOutput forward_with_cache(
        const std::vector<std::vector<float>>& input) const{
        
        AttentionOutput cache;
        cache.input = input;

        int seq_len = input.size();

        // Step 1: Q, K, V calculation
        cache.queries.assign(seq_len, std::vector<float>(embedding_dim, 0.0f));
        cache.keys.assign(seq_len, std::vector<float>(embedding_dim, 0.0f));
        cache.values.assign(seq_len, std::vector<float>(embedding_dim, 0.0f));


        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                for (int k = 0; k < embedding_dim; k++) {
                    cache.queries[i][j] += input[i][k] * W_q[k][j];
                    cache.keys[i][j] += input[i][k] * W_k[k][j];
                    cache.values[i][j] += input[i][k] * W_v[k][j];
                }
            }
        }

        // Step 2: Attention scores
        std::vector<std::vector<float>> scores(seq_len, std::vector<float>(seq_len, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int k = 0; k < embedding_dim; k++) {
                    score += cache.queries[i][k] * cache.keys[j][k];
                }
                scores[i][j] = score * scale;
            }
        }

        // Step 3: Causal masking + Softmax
        cache.attention_weights.assign(seq_len, std::vector<float>(seq_len, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            std::vector<float> row_scores;
            for (int j = 0; j <= i; j++) {
                row_scores.push_back(scores[i][j]);
            }

            auto probs = softmax(row_scores);
            for (int j = 0; j <= i; j++) {
                cache.attention_weights[i][j] = probs[j];
            }
        }
        
        // Step 4: Output = attention @ V
        cache.output.assign(seq_len, std::vector<float>(embedding_dim, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                for (int k = 0; k < i; k++) {
                    cache.output[i][j] += cache.attention_weights[i][k] * cache.values[k][j];
                }
            }
        }
        
        return cache;
    }

    // ===== BACKWARD =====
    // output_grad: output gradients (seq_len x embedding_dim)
    // cache: forward pass values

    std::vector<std::vector<float>> backward(
        const std::vector<std::vector<float>>& output_grad,
        const AttentionOutput& cache) {

            int seq_len = output_grad.size();

            // Gradient w.r.t. values: grad_V = attention_weights^T @ grad_output
            std::vector<std::vector<float>> grad_values(seq_len, std::vector<float>(embedding_dim, 0.0f));
            for (int i = 0; i < seq_len; i++) {
                for (int k = 0; k <= i; k++) {
                    for (int j = 0; j < embedding_dim; j++) {
                        grad_values[k][j] += cache.attention_weights[i][k] * output_grad[i][j];
                    }
                }
            }
            
            // Gradient w.r.t. attention_weights: grad_attn = grad_output @ V^T
            std::vector<std::vector<float>> grad_attn(seq_len, std::vector<float>(seq_len, 0.0f));
            for (int i = 0; i < seq_len; i++) {
                for (int k = 0; k <= i; k++) {
                    for (int j = 0; j < embedding_dim; j++) {
                        grad_attn[i][k] += output_grad[i][j] * cache.values[k][j];
                    }
                }
            }

            // Gradient w.r.t. scores (softmax backward)
            std::vector<std::vector<float>> grad_scores(seq_len, std::vector<float>(seq_len, 0.0f));
            for (int i = 0; i < seq_len; i ++) {
                for (int j = 0; j <= i; j++) {
                    for (int k = 0; k <= i; k++) {
                        float softmax_grad = cache.attention_weights[i][j] *
                            (grad_attn[i][k] - (j == k ? cache.attention_weights[i][k] : 0.0f));
                        grad_scores[i][j] += softmax_grad;
                    }
                }
            }

            // Gradient w.r.t. keys: grad_K = grad_scores @ Q
            std::vector<std::vector<float>> grad_keys(seq_len, std::vector<float>(embedding_dim, 0.0f));
            for (int i = 0; i < seq_len; i++) {
                for (int k = 0; k <= i; k++) {
                    for (int j = 0; j < embedding_dim; j++) {
                        grad_keys[k][j] += grad_scores[i][k] * scale * cache.queries[i][j];
                    }
                }
            }
            
            // Gradient w.r.t. queries: grad_Q = grad_scores @ K
            std::vector<std::vector<float>> grad_queries(seq_len, std::vector<float>(embedding_dim, 0.0f));
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < embedding_dim; j++) {
                    for (int k = 0; k <= i; k++) {
                        grad_queries[i][j] += grad_scores[i][k] * scale * cache.keys[k][j];
                    }
                }
            }
            
            // Gradient w.r.t. W_q: grad_Wq = input^T @ grad_queries
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < embedding_dim; j++) {
                    for (int k = 0; k < embedding_dim; k++) {
                        W_q_grad[j][k] += cache.input[i][j] * grad_queries[i][k];
                        W_k_grad[j][k] += cache.input[i][j] * grad_keys[i][k];
                        W_v_grad[j][k] += cache.input[i][j] * grad_values[i][k];
                    }
                }
            }

            // Gradient w.r.t. input: grad_input = grad_Q @ W_q^T + grad_K @ W_k^T + grad_V @ W_v^T
            std::vector<std::vector<float>> grad_input(seq_len, std::vector<float>(embedding_dim, 0.0f));
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < embedding_dim; j++) {
                    for (int k = 0; k < embedding_dim; k++) {
                        grad_input[i][j] += grad_queries[i][k] * W_q[j][k] +
                                       grad_keys[i][k] * W_k[j][k] +
                                       grad_values[i][k] * W_v[j][k];
                    }
                }
            }

            return grad_input;

        }
        void update_weights(float learning_rate) {
        for (int i = 0; i < embedding_dim; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                W_q[i][j] -= learning_rate * W_q_grad[i][j];
                W_k[i][j] -= learning_rate * W_k_grad[i][j];
                W_v[i][j] -= learning_rate * W_v_grad[i][j];
                
                W_q_grad[i][j] = 0.0f;
                W_k_grad[i][j] = 0.0f;
                W_v_grad[i][j] = 0.0f;
            }
        }
    }
};


// ============================================================================
// FEED-FORWARD NETWORK
// ============================================================================

struct FeedForward {
    int embedding_dim, hidden_dim;
    std::vector<std::vector<float>> W1, W2;
    std::vector<std::vector<float>> W1_grad, W2_grad;  // ← Ekle

    FeedForward(int embedding_dim, int hidden_dim = 128)
        : embedding_dim(embedding_dim), hidden_dim(hidden_dim) {
        W1.assign(embedding_dim, std::vector<float>(hidden_dim, 0.0f));
        W2.assign(hidden_dim, std::vector<float>(embedding_dim, 0.0f));
        W1_grad.assign(embedding_dim, std::vector<float>(hidden_dim, 0.0f));  // ← Ekle
        W2_grad.assign(hidden_dim, std::vector<float>(embedding_dim, 0.0f));  // ← Ekle
        
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.01f);

        for (int i = 0; i < embedding_dim; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                W1[i][j] = dist(gen);
            }
        }

        for (int i = 0; i < hidden_dim; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                W2[i][j] = dist(gen);
            }
        }
    }

    struct FFOutput {
        std::vector<std::vector<float>> hidden;
        std::vector<std::vector<float>> output;
        std::vector<std::vector<float>> input;
    };

    FFOutput forward_with_cache(
        const std::vector<std::vector<float>>& input) const {
        
        FFOutput cache;
        cache.input = input;
        
        int seq_len = input.size();

        // Layer 1: Linear + ReLU
        cache.hidden.assign(seq_len, std::vector<float>(hidden_dim, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                for (int k = 0; k < embedding_dim; k++) {
                    cache.hidden[i][j] += input[i][k] * W1[k][j];
                }
                cache.hidden[i][j] = relu(cache.hidden[i][j]);
            }
        }

        // Layer 2: Linear
        cache.output.assign(seq_len, std::vector<float>(embedding_dim, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                for (int k = 0; k < hidden_dim; k++) {
                    cache.output[i][j] += cache.hidden[i][k] * W2[k][j];
                }
            }
        }
        
        return cache;
    }

    // ===== BACKWARD =====
    std::vector<std::vector<float>> backward(
        const std::vector<std::vector<float>>& output_grad,
        const FFOutput& cache) {
        
        int seq_len = output_grad.size();
        
        // Gradient w.r.t. W2: grad_W2 = hidden^T @ grad_output
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                for (int k = 0; k < embedding_dim; k++) {
                    W2_grad[j][k] += cache.hidden[i][j] * output_grad[i][k];
                }
            }
        }
        
        // Gradient w.r.t. hidden: grad_hidden = grad_output @ W2^T
        std::vector<std::vector<float>> grad_hidden(seq_len, std::vector<float>(hidden_dim, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                for (int k = 0; k < embedding_dim; k++) {
                    grad_hidden[i][j] += output_grad[i][k] * W2[j][k];
                }
            }
        }
        
        // ReLU backward: grad_hidden *= (hidden > 0)
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                if (cache.hidden[i][j] <= 0.0f) {
                    grad_hidden[i][j] = 0.0f;
                }
            }
        }
        
        // Gradient w.r.t. W1: grad_W1 = input^T @ grad_hidden
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                for (int k = 0; k < hidden_dim; k++) {
                    W1_grad[j][k] += cache.input[i][j] * grad_hidden[i][k];
                }
            }
        }
        
        // Gradient w.r.t. input: grad_input = grad_hidden @ W1^T
        std::vector<std::vector<float>> grad_input(seq_len, std::vector<float>(embedding_dim, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                for (int k = 0; k < hidden_dim; k++) {
                    grad_input[i][j] += grad_hidden[i][k] * W1[j][k];
                }
            }
        }
        
        return grad_input;
    }

    void update_weights(float learning_rate) {
        for (int i = 0; i < embedding_dim; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                W1[i][j] -= learning_rate * W1_grad[i][j];
                W1_grad[i][j] = 0.0f;
            }
        }
        
        for (int i = 0; i < hidden_dim; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                W2[i][j] -= learning_rate * W2_grad[i][j];
                W2_grad[i][j] = 0.0f;
            }
        }
    }
};
    

// ============================================================================
// TRANSFORMER BLOK
// ============================================================================

struct TransformerBlock {
    SelfAttention attention;
    FeedForward feed_forward;
    int embedding_dim;

    TransformerBlock(int embedding_dim)
        : attention(embedding_dim), feed_forward(embedding_dim),
          embedding_dim(embedding_dim) {}

    struct BlockCache {
        std::vector<std::vector<float>> input;
        SelfAttention::AttentionOutput attn_cache;
        std::vector<std::vector<float>> attn_residual;
        FeedForward::FFOutput ff_cache;
        std::vector<std::vector<float>> output;
    };

    // Forward with caching
    BlockCache forward_with_cache(
        const std::vector<std::vector<float>>& input) const {
        
        BlockCache cache;
        cache.input = input;
        
        // Attention
        cache.attn_cache = attention.forward_with_cache(input);
        
        // Residual connection: input + attention_output
        cache.attn_residual.assign(input.size(), 
            std::vector<float>(embedding_dim, 0.0f));
        for (size_t i = 0; i < input.size(); i++) {
            for (int j = 0; j < embedding_dim; j++) {
                cache.attn_residual[i][j] = input[i][j] + cache.attn_cache.output[i][j];
            }
        }
        
        // FeedForward
        cache.ff_cache = feed_forward.forward_with_cache(cache.attn_residual);
        
        // Residual connection: attn_residual + ff_output
        cache.output.assign(input.size(), 
            std::vector<float>(embedding_dim, 0.0f));
        for (size_t i = 0; i < input.size(); i++) {
            for (int j = 0; j < embedding_dim; j++) {
                cache.output[i][j] = cache.attn_residual[i][j] + cache.ff_cache.output[i][j];
            }
        }
        
        return cache;
    }
    
    // ===== BACKWARD =====
    // output_grad: gradient w.r.t. output
    // cache: forward pass values
    // return: gradient w.r.t. input
    std::vector<std::vector<float>> backward(
        const std::vector<std::vector<float>>& output_grad,
        const BlockCache& cache) {
        
        // Backward through second residual connection
        // grad_attn_residual = output_grad + grad_ff_output
        auto grad_ff_out = output_grad;  // Gradient for FF output
        
        // FeedForward backward
        auto grad_attn_residual = feed_forward.backward(grad_ff_out, cache.ff_cache);
        
        // Add residual gradient
        for (size_t i = 0; i < output_grad.size(); i++) {
            for (int j = 0; j < embedding_dim; j++) {
                grad_attn_residual[i][j] += output_grad[i][j];
            }
        }
        
        // Backward through first residual connection
        // grad_input = grad_attn_residual + grad_attn_output
        auto grad_attn_out = grad_attn_residual;
        
        // Attention backward
        auto grad_input = attention.backward(grad_attn_out, cache.attn_cache);
        
        // Add residual gradient
        for (size_t i = 0; i < grad_attn_residual.size(); i++) {
            for (int j = 0; j < embedding_dim; j++) {
                grad_input[i][j] += grad_attn_residual[i][j];
            }
        }
        
        return grad_input;
    }
    
    void update_weights(float learning_rate) {
        attention.update_weights(learning_rate);
        feed_forward.update_weights(learning_rate);
    }
};


// ============================================================================
// GPT-2 MODEL
// ============================================================================

struct GPT2 {
    Embedding embedding;
    std::vector<TransformerBlock> blocks;
    std::vector<std::vector<float>> output_projection;
    std::vector<std::vector<float>> output_projection_grad;

    int vocab_size, embedding_dim, num_blocks, seq_len;

    GPT2(int vocab_size, int embedding_dim = 64, int num_blocks = 2, int seq_len = 32)
        : vocab_size(vocab_size), embedding_dim(embedding_dim), 
          num_blocks(num_blocks), seq_len(seq_len),
          embedding(vocab_size, embedding_dim){

        // Create Transformer blocks
        for (int i = 0; i < num_blocks; i++) {
            blocks.emplace_back(embedding_dim);
        }

        // Output projection initializing
        output_projection.assign(embedding_dim, std::vector<float>(vocab_size, 0.0f));
        output_projection_grad.assign(embedding_dim, std::vector<float>(vocab_size, 0.0f));

        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.01f);
        for (int i = 0; i < embedding_dim; i++) {
            for (int j = 0; j < vocab_size; j++) {
                output_projection[i][j] = dist(gen);
            }
        }
    }

    // Cache structure for full forward pass
    struct ModelCache {
        std::vector<int> token_ids;
        std::vector<std::vector<float>> embeddings;
        std::vector<TransformerBlock::BlockCache> block_caches;
        std::vector<std::vector<float>> final_hidden_states;
        std::vector<std::vector<float>> logits;
    };

    // Forward pass with caching
    ModelCache forward_with_cache(const std::vector<int>& token_ids) {
        ModelCache cache;
        cache.token_ids = token_ids;
        
        // Step 1: Embedding
        std::vector<std::vector<float>> x(token_ids.size(), 
            std::vector<float>(embedding_dim));
        for (size_t i = 0; i < token_ids.size(); i++) {
            x[i] = embedding.embed(token_ids[i]);
        }
        cache.embeddings = x;
        
        // Step 2: Transformer blocks with caching
        for (auto& block : blocks) {
            auto block_cache = block.forward_with_cache(x);
            cache.block_caches.push_back(block_cache);
            x = block_cache.output;
        }
        cache.final_hidden_states = x;
        
        // Step 3: Output projection -> vocab scores
        std::vector<std::vector<float>> logits(token_ids.size(), 
            std::vector<float>(vocab_size, 0.0f));
        for (size_t i = 0; i < token_ids.size(); i++) {
            for (int j = 0; j < vocab_size; j++) {
                for (int k = 0; k < embedding_dim; k++) {
                    logits[i][j] += x[i][k] * output_projection[k][j];
                }
            }
        }
        cache.logits = logits;
        
        return cache;
    }

    // ===== BACKWARD =====
    // logits_grad: gradient w.r.t. logits (from loss function)
    // cache: forward pass'tan kaydedilen values
    void backward(
        const std::vector<std::vector<float>>& logits_grad,
        const ModelCache& cache) {
        
        // Gradient through output projection
        // grad_x = logits_grad @ output_projection^T
        std::vector<std::vector<float>> grad_x(cache.logits.size(), 
            std::vector<float>(embedding_dim, 0.0f));
        
        for (size_t i = 0; i < cache.logits.size(); i++) {
            for (int j = 0; j < embedding_dim; j++) {
                for (int k = 0; k < vocab_size; k++) {
                    // Gradient w.r.t. input
                    grad_x[i][j] += logits_grad[i][k] * output_projection[k][j];
                    
                    // Gradient w.r.t. weights: grad_W = hidden^T @ grad_output
                    output_projection_grad[j][k] += 
                        cache.final_hidden_states[i][j] * logits_grad[i][k];
                }
            }
        }
        
        // Backward through transformer blocks (in reverse order)
        for (int block_idx = blocks.size() - 1; block_idx >= 0; block_idx--) {
            grad_x = blocks[block_idx].backward(grad_x, cache.block_caches[block_idx]);
        }
        
        // Backward through embedding
        embedding.backward(grad_x, cache.token_ids);
    }

    // Update all weights
    void update_weights(float learning_rate) {
        // Update output projection
        for (int i = 0; i < embedding_dim; i++) {
            for (int j = 0; j < vocab_size; j++) {
                output_projection[i][j] -= learning_rate * output_projection_grad[i][j];
                output_projection_grad[i][j] = 0.0f;
            }
        }
        
        // Update all transformer blocks
        for (auto& block : blocks) {
            block.update_weights(learning_rate);
        }
        
        // Update embedding
        embedding.update_weights(learning_rate);
    }

    // Compute cross-entropy loss
    float compute_loss(const std::vector<std::vector<float>>& logits,
                      const std::vector<int>& target_ids) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < logits.size(); i++) {
            auto probs = softmax(logits[i]);
            float prob = probs[target_ids[i]];
            if (prob > 0) {
                total_loss -= std::log(prob);
            }
        }
        
        return total_loss / logits.size();
    }

    // Compute gradient w.r.t. logits from cross-entropy loss
    // Loss = -log(softmax(logits)[target])
    // grad_logits = softmax(logits) - one_hot(target)
    std::vector<std::vector<float>> compute_logits_grad(
        const std::vector<std::vector<float>>& logits,
        const std::vector<int>& target_ids) {
        
        std::vector<std::vector<float>> grad(logits.size(), 
            std::vector<float>(vocab_size, 0.0f));
        
        for (size_t i = 0; i < logits.size(); i++) {
            auto probs = softmax(logits[i]);
            
            for (int j = 0; j < vocab_size; j++) {
                if (j == target_ids[i]) {
                    grad[i][j] = (probs[j] - 1.0f) / logits.size();
                } else {
                    grad[i][j] = probs[j] / logits.size();
                }
            }
        }
        
        return grad;
    }
};