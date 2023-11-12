import torch
import numpy as np
from tensorli.tensorli import Tensorli
from tensorli.models.transformerli import Transformerli


class TorchBlock(torch.nn.Module):
    def __init__(self, embd_dim, seq_len, n_heads) -> None:
        super().__init__()
        self.causal_attention = torch.nn.MultiheadAttention(
            embd_dim, n_heads, bias=False, dropout=0.0, batch_first=True
        )
        self.fc = torch.nn.Linear(embd_dim, 4 * embd_dim, bias=True)
        self.proj = torch.nn.Linear(4 * embd_dim, embd_dim, bias=True)

        self.layer_norm_1 = torch.nn.LayerNorm(embd_dim, elementwise_affine=False)
        self.layer_norm_2 = torch.nn.LayerNorm(embd_dim, elementwise_affine=False)

        self.attn_mask = torch.tril(torch.ones(seq_len, seq_len)) == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.layer_norm_1(x)
        x, _ = self.causal_attention(
            x, x, x, need_weights=True, attn_mask=self.attn_mask
        )  # including residual connection
        x_2 = input_x + x
        x = self.layer_norm_2(x_2)  # layer norm 2
        x = self.fc(x)
        x = x.relu()
        x = self.proj(x)
        # dropout could be added here
        x = x + x_2
        return x


class TorchTransformer(torch.nn.Module):
    def __init__(self, vocb_size, embd_dim, seq_len, n_layer, n_head) -> None:
        super().__init__()

        self.word_embedding = torch.nn.Embedding(vocb_size, embd_dim)
        self.pos_embedding = torch.nn.Embedding(seq_len, embd_dim)

        self.blocks = torch.nn.ModuleList(
            [TorchBlock(embd_dim, seq_len, n_head) for _ in range(n_layer)]
        )

        self.lm_head = torch.nn.Linear(embd_dim, vocb_size, bias=False)
        self.layer_norm = torch.nn.LayerNorm(embd_dim, elementwise_affine=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        pos = torch.tensor(np.arange(seq_len, dtype=np.int64).reshape(1, -1))
        token_embedding = self.word_embedding(idx)
        pos_embedding = self.pos_embedding(pos)
        x = token_embedding + pos_embedding
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = self.lm_head(x)
        return x


def build_torch_transformer(transformer: Transformerli) -> torch.nn.Module:
    torch_transformer = TorchTransformer(
        transformer.vocb_size,
        transformer.embd_dim,
        transformer.seq_len,
        transformer.n_layer,
        transformer.n_head,
    )

    # transfer the weights to the torch model
    torch_transformer.word_embedding.weight = torch.nn.Parameter(
        torch.tensor(transformer.word_embedding.weight.data, dtype=torch.float64)
    )
    torch_transformer.pos_embedding.weight = torch.nn.Parameter(
        torch.tensor(transformer.pos_embedding.weight.data, dtype=torch.float64)
    )
    torch_transformer.lm_head.weight = torch.nn.Parameter(
        torch.tensor(transformer.lm_head.weight.data, dtype=torch.float64)
    )
    for block, block_torch in zip(transformer.blocks, torch_transformer.blocks):
        block_torch.causal_attention.in_proj_weight = torch.nn.Parameter(
            torch.tensor(
                np.concatenate(
                    [
                        np.concatenate(
                            [head.query.weight.data for head in block.causal_attention.heads]
                        ),
                        np.concatenate(
                            [head.key.weight.data for head in block.causal_attention.heads]
                        ),
                        np.concatenate(
                            [head.value.weight.data for head in block.causal_attention.heads]
                        ),
                    ],
                    axis=0,
                ),
                dtype=torch.float64,
            )
        )
        block_torch.causal_attention.out_proj.weight = torch.nn.Parameter(
            torch.tensor(block.causal_attention.out_proj.weight.data, dtype=torch.float64)
        )
        block_torch.fc.weight = torch.nn.Parameter(
            torch.tensor(block.fc.weight.data, dtype=torch.float64)
        )
        block_torch.fc.bias = torch.nn.Parameter(
            torch.tensor(block.fc.bias.data, dtype=torch.float64)
        )
        block_torch.proj.weight = torch.nn.Parameter(
            torch.tensor(block.proj.weight.data, dtype=torch.float64)
        )
        block_torch.proj.bias = torch.nn.Parameter(
            torch.tensor(block.proj.bias.data, dtype=torch.float64)
        )

    return torch_transformer


def test_transformer():
    vocb_size = 4
    embd_dim = 4
    seq_len = 16
    n_layer = 1
    n_head = 4

    batch_size = 8

    transformer = Transformerli(vocb_size, embd_dim, seq_len, n_layer, n_head)

    x_numpy = np.random.randint(0, vocb_size - 1, size=(batch_size, seq_len))
    x = Tensorli(x_numpy)

    out = transformer(x)
    torch_transformer = build_torch_transformer(transformer)
    x_torch = torch.tensor(x_numpy, dtype=torch.int64)
    out_torch = torch_transformer(x_torch)

    # inference test
    assert np.allclose(out.data, out_torch.detach().numpy())

    params = transformer.parameters()
    param_count = sum(p.data.size for p in params)
    print(f"Number of parameters: {param_count}")

    params_torch = torch_transformer.parameters()
    param_count_torch = sum(p.numel() for p in params_torch)
    print(f"Number of parameters: {param_count_torch}")

    # compare the number of parameters
    assert np.allclose(param_count, param_count_torch)

    # grad test

    # compute the loss
    loss = (out - 0.005).mean().mean().mean()
    print(loss)
    loss.backward()

    loss_torch = (out_torch - 0.005).mean()
    print(loss_torch)
    loss_torch.backward()

    # compare the loss
    assert np.allclose(loss.data, loss_torch.detach().numpy())

    # compare the gradients
    assert np.allclose(
        transformer.word_embedding.weight.grad,
        torch_transformer.word_embedding.weight.grad.detach().numpy(),
    )
