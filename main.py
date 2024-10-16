import spacy
import random
import csv
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# Create a blank Urdu tokenizer
nlp = spacy.blank("ur")

def load_corpus(file_path: str) -> str:
    corpus = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        for row in csv_reader:
            if len(row) >= 2:
                corpus += row[1] + " "  # Add story text to corpus
    return corpus.strip()

def tokenize_corpus(text: str, chunk_size: int = 100000) -> List[str]:
    tokens = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        doc = nlp(chunk)
        tokens.extend([token.text for token in doc if not token.is_space])
    return tokens

def generate_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], Counter]:
    ngrams = defaultdict(Counter)
    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i:i+n-1])
        target = tokens[i+n-1]
        ngrams[context][target] += 1
    return ngrams

def get_starting_words(tokens: List[str]) -> List[str]:
    urdu_sentence_starters = ["ا", "ب", "پ", "ت", "ٹ", "ث", "ج", "چ", "ح", "خ", "د", "ڈ", "ذ", "ر", "ڑ", "ز", "ژ", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن", "و", "ہ", "ی"]
    return [token for token in tokens if token[0] in urdu_sentence_starters or token in ["،", "۔"]]

def weighted_random_choice(counter: Counter) -> str:
    if not counter:
        return ""  # Return an empty string if the counter is empty
    total = sum(counter.values())
    r = random.uniform(0, total)
    cumulative = 0
    for item, count in counter.items():
        cumulative += count
        if r <= cumulative:
            return item
    return list(counter.keys())[-1]  # Fallback

def generate_sentence(ngrams: Dict[int, Dict[Tuple[str, ...], Counter]], starting_words: List[str], max_n: int, prev_sentence: List[str] = None) -> List[str]:
    if prev_sentence and random.random() < 0.3:
        sentence = [random.choice(prev_sentence)]
    else:
        sentence = [random.choice(starting_words)]
    
    while len(sentence) < random.randint(5, 19):
        for n in range(max_n, 0, -1):
            context = tuple(sentence[-(n-1):])  # Adjust context length
            if context in ngrams[n]:
                next_word = weighted_random_choice(ngrams[n][context])
                sentence.append(next_word)
                if next_word in ["۔", "؟", "!"]:
                    return sentence
                break
        else:
            sentence.append(random.choice(starting_words))
    
    return sentence

def generate_paragraph(ngrams: Dict[int, Dict[Tuple[str, ...], Counter]], starting_words: List[str], max_n: int) -> str:
    sentences = []
    prev_sentence = None
    for _ in range(random.randint(5, 10)):
        sentence = generate_sentence(ngrams, starting_words, max_n, prev_sentence)
        sentences.append(" ".join(sentence))
        prev_sentence = sentence
    return " ".join(sentences)

def generate_ngram_story(ngrams: Dict[int, Dict[Tuple[str, ...], Counter]], starting_words: List[str], max_n: int) -> str:
    paragraphs = []
    for _ in range(3):
        paragraphs.append(generate_paragraph(ngrams, starting_words, max_n))
    return "\n\n".join(paragraphs)

class CustomMarkovChain:
    def __init__(self, corpus: str, state_size: int = 2):
        self.state_size = state_size
        self.model = defaultdict(Counter)
        self.starting_words = []
        self.build_model(corpus)

    def build_model(self, corpus: str):
        words = corpus.split()
        for i in range(len(words) - self.state_size):
            state = tuple(words[i:i+self.state_size])
            next_word = words[i+self.state_size]
            self.model[state][next_word] += 1

        self.starting_words = [word for word in words if word[0].isalpha()]

    def generate_sentence(self):
        current_state = tuple(random.choice(self.starting_words) for _ in range(self.state_size))
        sentence = list(current_state)

        while len(sentence) < 20:
            if current_state not in self.model or not self.model[current_state]:
                # If we reach a dead end, start a new sentence
                current_state = tuple(random.choice(self.starting_words) for _ in range(self.state_size))
                sentence.extend(current_state)
            else:
                next_word = weighted_random_choice(self.model[current_state])
                sentence.append(next_word)
                if next_word in ["۔", "؟", "!"]:
                    break
                current_state = tuple(sentence[-self.state_size:])

        return " ".join(sentence)

    def generate_paragraph(self, num_sentences: int = 5):
        return " ".join(self.generate_sentence() for _ in range(num_sentences))

def generate_markov_story(corpus: str, num_paragraphs: int = 3, sentences_per_paragraph: int = 5) -> str:
    markov_model = CustomMarkovChain(corpus, state_size=2)
    paragraphs = [markov_model.generate_paragraph(sentences_per_paragraph) for _ in range(num_paragraphs)]
    return "\n\n".join(paragraphs)

def main():
    corpus = load_corpus("urdu_stories.csv")
    tokens = tokenize_corpus(corpus)
    
    # Generate n-gram models
    ngrams = {
        1: generate_ngrams(tokens, 1),
        2: generate_ngrams(tokens, 2),
        3: generate_ngrams(tokens, 3),
        4: generate_ngrams(tokens, 4),  # Add 4-grams
        5: generate_ngrams(tokens, 5)   # Add 5-grams
    }
    
    starting_words = get_starting_words(tokens)
    
    print("5-gram Model Story with Backoff:")
    print(generate_ngram_story(ngrams, starting_words, 5))

    print("\nCustom Markov Chain Model Story:")
    print(generate_markov_story(corpus))

if __name__ == "__main__":
    main()
