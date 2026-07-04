import numpy as np
import mmap
import spacy

# Using SpaCy Model For Text Preprocessing
nlp = spacy.load("en_core_web_sm")

def load_vectors(file_path):
    print("Vectors file loading...")
    word_to_index = {}
    vectors = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        index = 0
        for line in iter(mmapped_file.readline, b""):
            line_str = line.decode('utf-8').strip()
            if not line_str:
                continue
                
            parts = line_str.split(' ')
            try:
                word = parts[0]
                vector = [float(x) for x in parts[1:]]
                word_to_index[word] = index
                vectors.append(vector)
                index += 1
            except ValueError:
                continue

    print("Converting to NumPy Array...")
    vectors = np.array(vectors, dtype='float32')
    print("Done!\n")
    return word_to_index, vectors


# 1. Text Preprocessing Function (POS Tag Filter)
def text_preprocess(sentence):
    doc = nlp(sentence)
    
    # Only Content-Bearing words (Nouns, Verbs, Adjectives, Adverbs, Proper Nouns)
    ALLOWED_TAGS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}
    
    cleaned_words = []
    for token in doc:
        if token.pos_ in ALLOWED_TAGS:
            cleaned_words.append(token.text.lower())
            
    return cleaned_words


# 2. Making One Vector Of Whole Sentence 
def get_sentence_vector(cleaned_words, word_to_index, vectors):
    valid_vectors = []
    for word in cleaned_words:
        if word in word_to_index:
            valid_vectors.append(vectors[word_to_index[word]])
            
    if len(valid_vectors) == 0:
        return np.zeros(50, dtype='float32')
        
    return np.mean(valid_vectors, axis=0)


# 3. Cosine Similarity Formula
def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_w1 = np.linalg.norm(vec1)
    magnitude_w2 = np.linalg.norm(vec2)
    
    if magnitude_w1 == 0 or magnitude_w2 == 0:
        return 0 
        
    return dot_product / (magnitude_w1 * magnitude_w2)


# ---- MAIN CODE ----
# ========================================================================================================================
# Enter Correct File Path Where You Installed Your Vecotrs Library
file_path = r"C:\University\Final Year Project\wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt" 
# ========================================================================================================================
word_to_index, vectors = load_vectors(file_path)

while True:
    print("\n================ Sentence Similarity Module ================")
    s1 = input("Enter First Sentence (or type 'exit' to exit): ").strip()
    if s1.lower() == 'exit':
        print("Existing...")
        break
        
    s2 = input("Enter Second Sentence: ").strip()
    
    # To Clean Text
    cleaned1 = text_preprocess(s1)
    cleaned2 = text_preprocess(s2)
    
    # To Show in Console That Sentences Are Preprocessed
    print(f"   [Cleaned 1]: {cleaned1}")
    print(f"   [Cleaned 2]: {cleaned2}")
    
    # Get Average Of Both Sentences
    vector1 = get_sentence_vector(cleaned1, word_to_index, vectors)
    vector2 = get_sentence_vector(cleaned2, word_to_index, vectors)
    
    if np.all(vector1 == 0) or np.all(vector2 == 0):
        print("\nError: There is word/words in sentences that is/are not in vectors library!")
        continue
        
    # Calculate Similarity score 
    score = calculate_cosine_similarity(vector1, vector2)
    
    # To Change Score Into Percentage 
    percentage_score = max(0, score) * 100 
    
    print(f"\nResult:")
    print(f"Sentence 1: \"{s1}\"")
    print(f"Sentence 2: \"{s2}\"")
    print(f"Calculated Similarity Score: {score:.4f}")
    print(f"Similarity Percentage: {percentage_score:.2f}%")
    print("============================================================")