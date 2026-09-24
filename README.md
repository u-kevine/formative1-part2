# Formative I, Part 1: A Small Neural Network Library in NumPy

I built a small, PyTorch-style neural network library using only NumPy. It has
one linear layer, three activations (ReLU, Sigmoid, Softmax), two losses
(binary and categorical cross-entropy) and an SGD optimizer. `main.py` puts
them together to train a single-layer classifier on the AND gate. No deep
learning framework is used anywhere.

## Layout

```
nn/
  module.py                                base Module interface
  layers/linear.py                         Linear: z = xW + b
  activations/relu.py, sigmoid.py, softmax.py
  losses/cross_entropy_loss.py             binary cross-entropy
  losses/categorical_cross_entropy_loss.py multi-class cross-entropy
  optim/sgd.py                             SGD
main.py                                    training loop on the AND dataset
```

Every module follows the same contract: `forward`, `backward`, `parameters`
and `zero_grad`. The losses and the optimizer are plain classes, not
`Module`s, because their signatures differ (`backward()` takes no argument,
and they have no learnable parameters of their own).

## Design decisions

- **Linear.** `W` has shape `(in, out)` and uses Glorot/Xavier uniform
  initialization with bound `sqrt(6 / (in + out))`, so the scale adapts to both
  dimensions. `b` starts at zero. The gradients are `dW = Xᵀ G`,
  `db = G.sum(axis=0)` and `dX = G Wᵀ`. They are summed over the batch, not
  averaged, because the loss already divides by `m`.
- **In-place gradients.** `dW` and `db` are allocated once in `__init__` and
  overwritten with `[...] =` in `backward`. This keeps the `(param, grad)`
  pairs held by the optimizer valid across steps. The layer also stores a copy
  of `x` in `forward`, so editing the input afterwards cannot corrupt the
  gradients.
- **Softmax.** The forward pass subtracts the row-wise max
  (`axis=1, keepdims=True`) before exponentiating, so large logits don't
  overflow. The backward pass applies the Jacobian `diag(a) - a aᵀ` to the
  upstream gradient without building it. This reduces to
  `dz = a * (g - sum(g * a, axis=1, keepdims=True))`, which is fully
  vectorized with no per-example loop.
- **Sigmoid.** A naive `1 / (1 + exp(-x))` overflows for large negative `x`.
  I split by sign so that only non-positive numbers are ever exponentiated.
  The backward pass reuses the cached output: `g * a * (1 - a)`.
- **ReLU.** The forward pass saves the mask `x > 0`, and the backward pass
  multiplies the gradient by it. At `x == 0` the gradient is 0.
- **Losses.** Predictions are clipped to `[1e-12, 1 - 1e-12]` so `log` never
  sees 0. Both losses average over the batch and return a Python `float`.
  Chained with their matching activation they give the simple gradient
  `(a - y) / m`.
- **SGD.** `param -= lr * grad` is done in place, and `zero_grad` clears every
  gradient in place. The only loop over parameters is the loop over the
  optimizer's `(param, grad)` list.
- **main.py.** `toy_data()` returns the AND dataset. `fit()` trains and returns
  both the model and the loss history. `train()` is a thin wrapper that returns
  only the loss history (the interface the tests expect). `accuracy()` works
  with no arguments: if no model is given, it trains one first.

## Running it

From the repository root, with the provided environment active:

```bash
conda env create -f environment.yml
conda activate iml-formative1

pytest              # public tests
ruff check nn/      # must exit 0
ruff check main.py  # must exit 0
python main.py      # trains on AND and prints the loss and accuracy
```

Running `python main.py` prints the loss falling from about 0.72 at epoch 0
to about 0.004 at epoch 3999, and a final accuracy of 100%.

## Vectorization

There are no Python loops over examples, rows or classes anywhere in `nn/`,
including the Softmax backward pass. The only loops are the optimizer's loop
over parameters and the training-epoch loop in `main.py`. A short loop over
five epoch numbers in `main.py` only controls which loss values get printed.


## References

- `nn_guide.pdf` (course guide), used for the chapter structure and the
  derivations.
