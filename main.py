"""Assembles Linear + Sigmoid + CrossEntropyLoss into a trained model on toy data."""

import numpy as np

from nn.activations import Sigmoid
from nn.layers import Linear
from nn.losses import CrossEntropyLoss
from nn.optim import SGD

# Populated by train(); read by accuracy(). Holds the model's final
# predictions and the true targets from the most recent training run.
_last_predictions = None
_last_targets = None


def toy_data() -> tuple[np.ndarray, np.ndarray]:
    """Build the toy AND-gate dataset used to sanity-check the pipeline.

    AND is linearly separable, so a single Linear + Sigmoid layer can
    solve it exactly -- unlike XOR, which no single-layer network can
    learn (see Chapter 10, "What a healthy run looks like").

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y). X has shape (4, 2), one
            row per input combination. y has shape (4, 1), the AND of
            each row.
    """
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([[0.0], [0.0], [0.0], [1.0]])
    return X, y


def train(epochs: int = 4000, lr: float = 1.0, seed: int = 0) -> list[float]:
    """Train a Linear + Sigmoid model on the toy AND-gate data.

    Runs forward through Linear then Sigmoid, computes binary
    cross-entropy loss, runs backward through the loss then Sigmoid
    then Linear (reverse order), steps the SGD optimizer, and zeros
    gradients before the next iteration.

    Args:
        epochs (int): number of full-batch training iterations.
        lr (float): learning rate passed to the SGD optimizer.
        seed (int): seed for NumPy's random generator, so the
            Linear layer's Xavier initialization is reproducible.

    Returns:
        list[float]: the loss recorded at every epoch, in order.
    """
    global _last_predictions, _last_targets

    np.random.seed(seed)
    X, y = toy_data()

    linear = Linear(in_features=2, out_features=1)
    sigmoid = Sigmoid()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(linear.parameters(), lr=lr)

    loss_history = []
    predictions = None
    for _ in range(epochs):
        z = linear.forward(X)
        predictions = sigmoid.forward(z)
        loss = loss_fn.forward(predictions, y)
        loss_history.append(loss)

        grad = loss_fn.backward()
        grad = sigmoid.backward(grad)
        linear.backward(grad)

        optimizer.step()
        optimizer.zero_grad()

    _last_predictions = predictions
    _last_targets = y
    return loss_history


def accuracy(loss_history: list[float] = None) -> float:
    """Report final classification accuracy from the most recent training run.

    Args:
        loss_history (list[float]): unused; accepted for interface
            compatibility. Accuracy is computed from the predictions
            train() produced on its last epoch, stored as module
            state.

    Returns:
        float: fraction of the toy dataset's 4 examples classified
            correctly (prediction >= 0.5 counts as class 1).
    """
    if _last_predictions is None or _last_targets is None:
        train()
    predicted_labels = (_last_predictions >= 0.5).astype(float)
    correct = np.sum(predicted_labels == _last_targets)
    return float(correct / _last_targets.shape[0])


if __name__ == "__main__":
    history = train()
    for epoch in (0, 999, 1999, 2999, 3999):
        if epoch < len(history):
            print(f"epoch {epoch}: loss = {history[epoch]:.6f}")
    print(f"final accuracy: {accuracy(history):.2%}")
