class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 1. Scaled Dot-Product Attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # 2. Causal Masking (Adds -inf to future tokens)
        if mask is not None:
            scores += (mask * -1e9) 

        # 3. Softmax (Row-wise probability distribution)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True)) 
        weights /= np.sum(weights, axis=-1, keepdims=True)

        # 4. Contextual Mixing
        context = np.matmul(weights, V)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.dot(context, self.W_o)
