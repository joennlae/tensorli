
# Tensorli

Absolute minimalistic implementation of a GPT-like transformer using only numpy.

The implementation includes:
- Automatic differentiation
- `Tensorli` object (PyTorch like)
- Simple NN layers: `Linearli`, `Embeddingli`, `MultiheadAttentionli`, `LayerNorm`
- Optimizers: `Adamli`

everything that is needed to train and execute a GPT-like transformer model.

```
Everything else are optimizations.
```

## Naming

In the region where I grew up we have a word for "little" that is used as a suffix. For example, "little dog" would be "dogli". I thought it would be a nice name for a minimalistic implementation of a neural network library.

### Caveats

This library works but it is not optimized. It is not meant to be used in production. It is meant to be used as a learning tool.


### Plans & Outlook

- [ ] Add Dropout
- [ ] Add more experimental architectures

