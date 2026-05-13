class LanguageModel:
    def __init__(self, vocab_size, d_model, num_heads):
        self.emb_matrix = np.random.randn(vocab_size, d_model) * 0.1
        self.block = TransformerBlock(d_model, num_heads)
        self.W_out = np.random.randn(d_model, vocab_size) * 0.1 # Projecting back to vocab

    def forward(self, token_ids):
        seq_len = len(token_ids)
        
        # 1. Embedding + Positional Encoding
        x = self.emb_matrix[token_ids]
        pe = get_positional_encoding(seq_len, D_MODEL)
        x = (x + pe)[np.newaxis, :, :] # Add batch dimension
        
        # 2. Causal Mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        
        # 3. Transformer Logic
        hidden_states = self.block.forward(x, mask=mask)
        
        # 4. Final Linear Head (Logits)
        logits = np.dot(hidden_states, self.W_out)
        
        # 5. Final Softmax for probability distribution
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs
