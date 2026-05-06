import re
import numpy as np
from collections import defaultdict

# Pre-tokenization
import collections
def initialize_vocabulary(text):
    # 1. Split text by whitespace to get individual words
    words = text.split()
    
    # 2. Count occurrences of each word
    counts = collections.Counter(words)
    
    # 3. Format for BPE: "word" -> "w o r d </w>"
    # We split every character by a space so BPE can treat them as individual symbols
    bpe_vocab = {}
    for word, freq in counts.items():
        # Join characters with spaces and add end-of-word token
        char_sequence = ' '.join(list(word)) + ' </w>'
        bpe_vocab[char_sequence] = freq
        
    return bpe_vocab
def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6}
for _ in range(5):
    p = get_stats(vocab)
    if not p: break
    best = max(p, key=p.get)
    vocab = merge_vocab(best, vocab)
                           # ******continue********
