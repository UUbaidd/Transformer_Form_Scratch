class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1) # ReLU activation
        return np.dot(hidden, self.W2) + self.b2
