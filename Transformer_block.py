class TransformerBlock:
    def __init__(self, d_model, num_heads):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff=d_model * 4)

    def forward(self, x, mask=None):
        # Attention Layer + Residual Connection + Norm
        attn_out = self.mha.forward(x, mask=mask)
        x = layer_norm(x + attn_out)
        
        # FeedForward Layer + Residual Connection + Norm
        ffn_out = self.ffn.forward(x)
        x = layer_norm(x + ffn_out)
        return x
