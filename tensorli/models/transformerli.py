import numpy as np

from tensorli.tensorli import Tensorli
from tensorli.nnli import Moduli, Linearli, MultiHeadAttentionli, Embeddingli


class Blockli(Moduli):
    def __init__(self, embd_dim, seq_len, n_heads) -> None:
        super().__init__()
        self.causal_attention = MultiHeadAttentionli(
            embd_dim=embd_dim, seq_len=seq_len, n_heads=n_heads
        )
        self.fc = Linearli(embd_dim, 4 * embd_dim, bias=True)
        self.proj = Linearli(4 * embd_dim, embd_dim, bias=True)

    def forward(self, x: Tensorli) -> Tensorli:
        x_2 = x + self.causal_attention(x.layer_norm())  # including residual connection
        x = x_2.layer_norm()  # layer norm 2
        x = self.fc(x)
        x = x.relu()
        x = self.proj(x)
        # dropout could be added here
        x = x + x_2  # residual connection
        return x

    def parameters(self) -> list[Tensorli]:
        return self.causal_attention.parameters() + self.fc.parameters() + self.proj.parameters()


class Transformerli(Moduli):
    def __init__(self, vocb_size, embd_dim, seq_len, n_layer, n_head) -> None:
        super().__init__()
        # seq_len also called block size or context window size
        self.vocb_size = vocb_size
        self.embd_dim = embd_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_size = embd_dim // n_head
        self.seq_len = seq_len

        self.word_embedding = Embeddingli(vocb_size, embd_dim)
        self.pos_embedding = Embeddingli(seq_len, embd_dim)

        self.blocks = [Blockli(embd_dim, seq_len, n_head) for _ in range(n_layer)]

        self.lm_head = Linearli(embd_dim, vocb_size, bias=False)

    def forward(self, idx: Tensorli) -> Tensorli:
        # x is a batch of sequences of word indices
        # (batch_size, seq_len)
        _, seq_len = idx.shape
        assert seq_len <= self.seq_len, "sequence length is too long for the model"
        pos = Tensorli(np.arange(seq_len, dtype=np.int64).reshape(1, -1))

        token_embedding = self.word_embedding(idx)
        pos_embedding = self.pos_embedding(pos)
        x = token_embedding + pos_embedding
        for block in self.blocks:
            print("block")
            x = block(x)
        print("after block")
        x = x.layer_norm()
        x = self.lm_head(x)
        return x

    def parameters(self) -> list[Tensorli]:
        return (
            self.word_embedding.parameters()
            + self.pos_embedding.parameters()
            + [p for block in self.blocks for p in block.parameters()]
            + self.lm_head.parameters()
        )
