import torch.nn as nn

class MaskedTransformerAutoencoder(nn.Module):


    def __init__(self, embed_dim):
        super().__init__()

        self.n_heads = 8

        encode_head_1 = nn.MultiHeadAttention(1280, self.n_heads)
        encode_1 = nn.Sequential(
            nn.Linear((1280, ), 1024),
            nn.GELU()
        )
        
        encode_head_2 = nn.MultiHeadAttention(1024, self.n_heads)
        encode_2 = nn.Sequential(
            nn.Linear((1024, ), 896),
            nn.GELU()
        )
        
        encode_head_3 = nn.MultiHeadAttention(896, self.n_heads)
        encode_3 = nn.Linear((896, ), 768)

        attn_head_4 = nn.MultiHeadAttention(768, self.n_heads)

        decode_1 = nn.Sequential(
            nn.Linear((768, ), 896),
            nn.GELU()
        ) 

        decode_2 = nn.Sequential(
            nn.Linear((768, ), 896),
            nn.GELU()
        )
        decode_head_2 = nn.MultiHeadAttention(896, self.n_heads)
        
        decode_3 = nn.Sequential(
            nn.Linear((896, ), 1024),
            nn.GELU()
        )
        decode_head_3 = nn.MultiHeadAttention(1024, self.n_heads)

        decode_4 = nn.Linear((1024, ), 1280)
        decode_head_4 = nn.MultiHeadAttention(1280, self.n_heads)


        # Sketch ...
        def forward(self, x):
            attn_output, attn_output_weights = encode_head_1(x, x, x) # have to specify batch first
            x = encode_1(attn_output)

            attn_output, attn_output_weights = encode_head_1(x, x, x) # have to specify batch first
            x = encode_2(attn_output)
            
            attn_output, attn_output_weights = encode_head_3(x, x, x) # have to specify batch first
            x = encode_3(attn_output)

            x = decode_1(x)
            x, attn_output_weights = decode_head_2(x, x, x) # have to specify batch first

            # and so on...


