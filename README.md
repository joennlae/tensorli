<div align="center">
<img src="logo.png" alt="logo" width="200"/>

<h3>
Minimalistic implementation of a GPT-like transformer using only numpy (<650 lines).
</h3>
</div>

---

The implementation includes:
- Automatic differentiation
- `Tensorli` object (PyTorch like)
- Simple NN layers: `Linearli`, `Embeddingli`, `MultiheadAttentionli`, `LayerNorm`
- Optimizers: `Adamli`

All that is "needed" to train and execute a GPT-like transformer model.

*`...and everything else is just efficiency`* ~ Andrej Karpathy<sup>[1](#myfootnote1)</sup>.

<a name="myfootnote1">1</a>: [Youtube - micrograd](https://youtu.be/VMj-3S1tku0?si=6qISQdXUKBSMOy3Z&t=474) 

## Example

```python
from tensorli.tensorli import Tensorli
from tensorli.models.transformerli import Transformerli

vocb_size, embd_dim, seq_len, n_layer, n_head, batch_size = 10, 64, 10, 3, 4, 16

transformer = Transformerli(vocb_size, embd_dim, seq_len, n_layer, n_head)

x_numpy = np.random.randint(0, vocb_size - 1, size=(batch_size, seq_len))
x = Tensorli(x_numpy)

out = transformer(x)
```

## Installation & Development

```bash
# create conda env (if you want to do it faster use mamba)
# conda env create -f environment.yml
# (if you want to do it faster use mamba)
mamba env create -f environment.yml
# activate env
conda activate tensorli
# python path need to be set to the root of the project
export PYTHONPATH=$PWD
# run all tests
pytest
# run specific test (verbose)
pytest -v -rP -k "transformer"
```

## Naming

In the region where I grew up, a word for "little" is used as a suffix \[[2](https://de.wikipedia.org/wiki/-li)\]. For example, "little dog" would be "dogli". I thought it would be a nice name for a minimalistic implementation of a neural network library.

### Caveats

This library works, but it is NOT optimized. It is not meant to be used in production or for anything at scale. It is meant to be used as a learning tool.

## Inspiration

This library is heavily inspired by the following projects:
- [minGPT](https://github.com/karpathy/minGPT)
- [tinygrad](https://github.com/tinygrad/tinygrad)

I highly recommend checking them out.

## Training loss plot

Running the test script `test/test_adder.py` will produce the following loss curve:

![loss](loss.png)

### Plans & Outlook

- [ ] Add Dropout
- [ ] Add more experimental architectures
