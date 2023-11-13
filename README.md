
# Tensorli

Absolute minimalistic implementation of a GPT-like transformer using only `numpy`.

The implementation includes:
- Automatic differentiation
- `Tensorli` object (PyTorch like)
- Simple NN layers: `Linearli`, `Embeddingli`, `MultiheadAttentionli`, `LayerNorm`
- Optimizers: `Adamli`

All that is "needed" to train and execute a GPT-like transformer model.

```
Everything else is optimizations.
```

## Naming

In the region where I grew up, a word for "little" is used as a suffix \[[1](https://de.wikipedia.org/wiki/-li)\]. For example, "little dog" would be "dogli". I thought it would be a nice name for a minimalistic implementation of a neural network library.

### Caveats

This library works, but it is NOT optimized. It is not meant to be used in production or for anything at scale. It is meant to be used as a learning tool.

## Inspiration

This library is heavily inspired by the following projects:
- [minGPT](https://github.com/karpathy/minGPT)
- [tinygrad](https://github.com/tinygrad/tinygrad)

I highly recommend checking them out.

### Plans & Outlook

- [ ] Add Dropout
- [ ] Add more experimental architectures
