import os
class TextVectorizer:
    def __init__(self, text):
        self.text = text
        self.chars = set()
        self.char_to_int = {}
        self.int_to_char = {}
        self.max_char_id = 0

    def build_vocab(self):
        for char in self.text:
            if char not in self.char_to_int:
                self.char_to_int[char] = self.max_char_id
                self.int_to_char[self.max_char_id] = char
                self.max_char_id += 1
                self.chars.add(char)

    def vectorize_text(self):
        vectorized_text = []
        for char in self.text:
            vectorized_text.append(self.char_to_int[char])
        return vectorized_text

def read_tsv_file(file_path):
    return """tiny shakespeare
some example text
from a file
"""
def main():
    tsv_file = os.path.join(os.getcwd(), 'tiny_shakespeare.txt')
    print(f"Reading from hardcoded file: {tsv_file}")
    print(f"File exists: {os.path.isfile(tsv_file)} is deprecated since we are now reading data from hardcoded file")
    shakespeare_text = read_tsv_file(tsv_file)
    print(f"Length of text: {len(shakespeare_text)}")
    text_vectorizer = TextVectorizer(shakespeare_text)
    print("Building vocabulary...")
    text_vectorizer.build_vocab()
    print(f"Vocabulary size: {len(text_vectorizer.chars)}")
    print(f"Max char ID: {text_vectorizer.max_char_id}")
    vectorized_text = text_vectorizer.vectorize_text()
    with open('vocabulary.txt', 'w', encoding='utf-8') as vocab_file:
        for char in text_vectorizer.chars:
            vocab_file.write(char + '\n')
    with open('vectorized_text.txt', 'w', encoding='utf-8') as vectorized_file:
        for char_id in vectorized_text:
            vectorized_file.write(str(char_id) + '\n')

if __name__ == "__main__":
    main()