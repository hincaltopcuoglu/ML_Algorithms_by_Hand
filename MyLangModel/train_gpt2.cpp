#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <random>
#include <cmath>
#include "gpt2_model.h"

// ============================================================================
// DATA PROCESSING FUNCTIONS
// ============================================================================

// upload text file
std::string load_text_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Can not open file: " + filepath);
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
}

// create vocabulary : give every character a unique ID
std::unordered_map<char, int> create_vocabulary(const std::string& text) {
    std::unordered_map<char, int> vocab;
    int char_count = 0;
    
    for (char c : text) {
        if (vocab.find(c) == vocab.end()) {
            vocab[c] = char_count++;
        }
    }
    
    return vocab;
}

// Reverse vocabulary: ID -> Character
std::unordered_map<int, char> create_reverse_vocab(const std::unordered_map<char, int>& vocab) {
    std::unordered_map<int, char> reverse_vocab;
    for (const auto& pair : vocab) {
        reverse_vocab[pair.second] = pair.first;
    }
    return reverse_vocab;
}

// convert text into token ids
std::vector<int> tokenize(const std::string& text, const std::unordered_map<char, int>& vocab) {
    std::vector<int> tokens;
    for (char c : text) {
        if (vocab.find(c) != vocab.end()) {
            tokens.push_back(vocab.at(c));
        }
    }
    return tokens;
}

// convert token id to text
std::string detokenize(const std::vector<int>& tokens, const std::unordered_map<int, char>& reverse_vocab) {
    std::string text;
    for (int token : tokens) {
        if (reverse_vocab.find(token) != reverse_vocab.end()) {
            text += reverse_vocab.at(token);
        }
    }
    return text;
}

// ============================================================================
// TRAINING LOOP
// ============================================================================

struct TrainingConfig {
    int num_epochs = 5;
    int batch_size = 4;
    int seq_len = 32;
    float learning_rate = 0.001f;
    int log_interval = 50;
};


// ============================================================================
// TEXT GENERATION
// ============================================================================

// create text from given prompt
std::string generate_text(GPT2& model,
                         const std::string& prompt,
                         const std::unordered_map<char, int>& vocab,
                         const std::unordered_map<int, char>& reverse_vocab,
                         int max_length = 100) {
                        
    std::string generated = prompt;
    std::vector<int> tokens = tokenize(prompt, vocab);

    // generate new tokens one by one
    for (int i = 0; i < max_length; i++) {
        // take last seq_len token
        std::vector<int> input_tokens;
        int start = std::max(0, (int)tokens.size() - model.seq_len + 1);
        for (size_t j = start; j < tokens.size(); j++) {
            input_tokens.push_back(tokens[j]);
        }

        // padding if needed
        while ((int)input_tokens.size() < model.seq_len) {
            input_tokens.insert(input_tokens.begin(), 0);
        }

        // Forward pass
        auto cache = model.forward_with_cache(input_tokens);
        auto logits = cache.logits;
        
        // get last position's logits
        auto last_logits = logits.back();
        
        // Softmax: convert to probabilities
        auto probs = softmax(last_logits);
        
        // Argmax: select most probable token 
        int next_token = 0;
        float max_prob = probs[0];
        for (size_t j = 1; j < probs.size(); j++) {
            if (probs[j] > max_prob) {
                max_prob = probs[j];
                next_token = j;
            }
        }
        // add token as text
        tokens.push_back(next_token);
        if (next_token < model.vocab_size && reverse_vocab.find(next_token) != reverse_vocab.end()) {
            generated += reverse_vocab.at(next_token);
        }
    }
    
    return generated;
}


// ============================================================================
// MAIN: TRAINING
// ============================================================================

int main() {
    try {
        std::cout << "=== GPT-2 Similar Model:Training with Backpropagation ===" << std::endl;
        std::cout << std::endl;
        
        // -------- ADIM 1: VERİ YÜKLEME --------
        std::cout << "[Step 1] Data is loading..." << std::endl;
        
        std::string text = load_text_file("tiny_shakespeare.txt");
        
        std::cout << "  - Text size: " << text.size() << " karakter" << std::endl;
        std::cout << "  - First 100 character: " << text.substr(0, 100) << std::endl;
        std::cout << std::endl;
        
        // -------- ADIM 2: VOCABULARY --------
        std::cout << "[STEP 2] Vocabulary is being created..." << std::endl;
        
        auto vocab = create_vocabulary(text);
        auto reverse_vocab = create_reverse_vocab(vocab);
        
        std::cout << "  - Vocab shape: " << vocab.size() << std::endl;
        std::cout << std::endl;
        
        // -------- ADIM 3: MODEL İNİTİALİZASYON --------
        std::cout << "[STEP 3] Model is being created..." << std::endl;
        
        TrainingConfig config;
        config.seq_len = 16;  // Küçültüldü
        config.num_epochs = 1;  // 1 epoch
        config.batch_size = 2;  // Batch size küçültüldü
        config.learning_rate = 0.001f;
        
        GPT2 model(vocab.size(), 32, 1, config.seq_len);  // embedding_dim=32, blocks=1
        
        std::cout << "  - Model config:" << std::endl;
        std::cout << "    - Vocab size: " << vocab.size() << std::endl;
        std::cout << "    - Embedding dim: 64" << std::endl;
        std::cout << "    - Blocks: 2" << std::endl;
        std::cout << "    - Seq len: " << config.seq_len << std::endl;
        std::cout << std::endl;
        
        // -------- ADIM 4: TOKENIZATION --------
        std::cout << "[STEP 4] data is being tokenized..." << std::endl;
        
        auto tokens = tokenize(text, vocab);
        std::cout << "  - Token count: " << tokens.size() << std::endl;
        std::cout << std::endl;
        
        // -------- ADIM 5: TRAINING LOOP --------
        std::cout << "[STEP 5] Model training..." << std::endl;
        std::cout << std::endl;
        
        // Training sequences oluştur
        std::vector<std::vector<int>> sequences;
        int num_samples = std::min(50, (int)tokens.size() - config.seq_len);
        
        for (int i = 0; i < num_samples; i++) {
            std::vector<int> seq(tokens.begin() + i, tokens.begin() + i + config.seq_len);
            sequences.push_back(seq);
        }
        
        std::cout << "  - Training sequences: " << sequences.size() << std::endl;
        std::cout << std::endl;
        
        // Eğitim döngüsü
        for (int epoch = 0; epoch < config.num_epochs; epoch++) {
            float total_loss = 0.0f;
            int batch_count = 0;
            
            // Shuffle sequences
            auto shuffled_sequences = sequences;
            std::mt19937 gen(epoch);
            std::shuffle(shuffled_sequences.begin(), shuffled_sequences.end(), gen);
            
            // Her batch'i işle
            for (size_t i = 0; i < shuffled_sequences.size(); i += config.batch_size) {
                float batch_loss = 0.0f;
                int batch_size = 0;
                
                try {
                    std::cout << "  DEBUG: Batch " << (i / config.batch_size) + 1 << std::endl;
                    
                    // Mini-batch işle
                    for (int b = 0; b < config.batch_size && i + b < shuffled_sequences.size(); b++) {
                        auto seq = shuffled_sequences[i + b];
                        
                        // Input: seq_len-1, Target: son token
                        std::vector<int> input_tokens(seq.begin(), seq.end() - 1);
                        std::vector<int> target_tokens(seq.begin() + 1, seq.end());
                        
                        std::cout << "    Sample " << b << ": input_size=" << input_tokens.size() 
                                 << ", target_size=" << target_tokens.size() 
                                 << ", model.seq_len=" << model.seq_len << std::endl;
                        
                        // FORWARD PASS
                        auto cache = model.forward_with_cache(input_tokens);
                        auto logits = cache.logits;
                        
                        std::cout << "      Forward done: logits.size()=" << logits.size() << std::endl;
                        
                        // Loss hesapla
                        float sample_loss = 0.0f;
                        try {
                            sample_loss = model.compute_loss(logits, target_tokens);
                            std::cout << "      Loss computed: " << sample_loss << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error in compute_loss: " << e.what() << std::endl;
                            throw;
                        }
                        batch_loss += sample_loss;
                        batch_size++;
                        
                        // BACKWARD PASS
                        // Logits'ten loss gradient'ı hesapla
                        auto logits_grad = model.compute_logits_grad(logits, target_tokens);
                        std::cout << "      Logits grad computed: size=" << logits_grad.size() << std::endl;
                        
                        // Backward pass
                        try {
                            model.backward(logits_grad, cache);
                            std::cout << "      Backward done" << std::endl;
                        } catch (const std::exception& e) {
                            std::cerr << "Error in backward: " << e.what() << std::endl;
                            throw;
                        }
                    }
                    
                    if (batch_size > 0) {
                        batch_loss /= batch_size;
                        total_loss += batch_loss;
                        batch_count++;
                        
                        // WEIGHT UPDATE
                        model.update_weights(config.learning_rate);
                    }
                    
                    // Log
                    if (batch_count % config.log_interval == 0 && batch_count > 0) {
                        std::cout << "  Epoch " << epoch + 1 << "/" << config.num_epochs
                                 << " - Batch " << batch_count 
                                 << " - Loss: " << batch_loss << std::endl;
                    }
                } catch (const std::exception& batch_e) {
                    std::cerr << "Error at Epoch " << epoch + 1 << ", Batch " << (i / config.batch_size) + 1 << std::endl;
                    std::cerr << "Error message: " << batch_e.what() << std::endl;
                    throw;
                }
            }
            
            float avg_loss = (batch_count > 0) ? total_loss / batch_count : 0.0f;
            std::cout << "  Epoch " << epoch + 1 << " completed - Avg Loss: " << avg_loss << std::endl;
        }
        
        std::cout << std::endl;
        
        // -------- ADIM 6: TEXT GENERATION --------
        std::cout << "[STEP 6] TEXT IS BEING CREATED..." << std::endl;
        std::cout << std::endl;
        
        std::string prompt = "The";
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << std::endl;
        
        std::string generated = generate_text(model, prompt, vocab, reverse_vocab, 200);
        
        std::cout << "Generated text:" << std::endl;
        std::cout << "---" << std::endl;
        std::cout << generated << std::endl;
        std::cout << "---" << std::endl;
        std::cout << std::endl;
        
        std::cout << "=== Trainig Completed! ===" << std::endl;

        
        
    } catch (const std::exception& e) {
        std::cerr << "Hata oluştu: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}