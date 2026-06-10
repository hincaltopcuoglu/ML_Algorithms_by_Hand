// Include necessary headers
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>

// Function prototypes
std::unordered_map<char, int> create_vocabulary(const std::string& text);
std::vector<int> tokenize_text(const std::string& text, const std::unordered_map<char, int>& vocabulary);
std::string prepare_text_from_file(const std::string& file_content);

// Function to create vocabulary from given text
std::unordered_map<char, int> create_vocabulary(const std::string& text) {
    std::unordered_map<char, int> vocabulary;
    int char_count = 0;

    for (char c : text) {
        if (vocabulary.count(c) == 0) {
            vocabulary[c] = char_count;
            char_count++;
        }
    }
    return vocabulary;
}

// Function to prepare text from a file
std::string prepare_text_from_file(const std::string& file_content) {
    std::istringstream ss(file_content);
    std::string line, full_text;
    while (std::getline(ss, line)) {
        full_text += line + "\n";
    }
    return full_text;
}

// Function to tokenize given text using a vocabulary
std::vector<int> tokenize_text(const std::string& text, const std::unordered_map<char, int>& vocabulary) {
    std::vector<int> tokenized_text;

    for (char c : text) {
        tokenized_text.push_back(vocabulary.at(c));
    }

    return tokenized_text;
}

// Function to print the results (tokenized text, vocabulary size, and individual characters)
void print_results(const std::string& file_name, const std::vector<int>& tokenized_text, const std::unordered_map<char, int>& vocabulary) {
    std::cout << "Tokenized text for " << file_name << ":\n";
    for (int i : tokenized_text) {
        std::cout << i << " ";
    }
    std::cout << "\nVocabulary size: " << vocabulary.size() << "\n";
    for (const auto& pair : vocabulary) {
        std::cout << pair.first << ": " << pair.second << "\n";
    }
}

int main() {
    try {
        // Read the Tiny Shakespeare text dataset
        std::ifstream file("tiny_shakespeare.txt");
        if (!file.is_open()) {
            std::cerr << "Failed to open file: 'tiny_shakespeare.txt'" << std::endl;
            return 1;
        }

        // Read the file content
        std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        // Prepare text from file content
        std::string raw_text = prepare_text_from_file(file_content);

        // Create vocabulary from raw text
        std::unordered_map<char, int> vocabulary = create_vocabulary(raw_text);

        // Tokenize the text using the created vocabulary
        std::vector<int> tokenized_text = tokenize_text(raw_text, vocabulary);

        // Print the results
        print_results("tiny_shakespeare.txt", tokenized_text, vocabulary);
    } catch (const std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << std::endl;
    }
    return 0;
}