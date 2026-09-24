"""Assembles Linear + Sigmoid + CrossEntropyLoss into a trained model on toy data."""

from typing import NamedTuple

import numpy as np

from nn.activations import Sigmoid
from nn.layers import Linear
from nn.losses import CrossEntropyLoss
from nn.optim import SGD


class Model(NamedTuple):
    """A trained single-layer classifier: Linear followed by Sigmoid.

    Attributes:
        linear (Linear): the fully connected layer producing logits.
        sigmoid (Sigmoid): the activation turning logits into
            probabilities.
    """

    linear: Linear
    sigmoid: Sigmoid

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute predicted probabilities for a batch of inputs.

        Args:
            X (np.ndarray): inputs, shape (m, in_features).

        Returns:
            np.ndarray: probabilities of class 1, shape (m, 1).
        """
        return self.sigmoid.forward(self.linear.forward(X))


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


def fit(
    epochs: int = 4000, lr: float = 1.0, seed: int = 0
) -> tuple[Model, list[float]]:
    """Train a Linear + Sigmoid model on the toy AND-gate data.

    Runs forward through Linear then Sigmoid, computes binary
    cross-entropy loss, runs backward through the loss then Sigmoid
    then Linear (reverse order), steps the SGD optimizer, and zeros
    gradients before the next iteration.

    Unlike train(), this also returns the trained model itself, so
    callers (such as accuracy()) can use it without any module-level
    state.

    Args:
        epochs (int): number of full-batch training iterations.
        lr (float): learning rate passed to the SGD optimizer.
        seed (int): seed for NumPy's random generator, so the
            Linear layer's Xavier initialization is reproducible.

    Returns:
        tuple[Model, list[float]]: the trained model, and the loss
            recorded at every epoch, in order.
    """
    np.random.seed(seed)
    X, y = toy_data()

    linear = Linear(in_features=2, out_features=1)
    sigmoid = Sigmoid()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(linear.parameters(), lr=lr)

    loss_history = []
    for _ in range(epochs):
        predictions = sigmoid.forward(linear.forward(X))
        loss_history.append(loss_fn.forward(predictions, y))

        grad = loss_fn.backward()
        grad = sigmoid.backward(grad)
        linear.backward(grad)

        optimizer.step()
        optimizer.zero_grad()

    return Model(linear, sigmoid), loss_history


def train(epochs: int = 4000, lr: float = 1.0, seed: int = 0) -> list[float]:
    """Train a Linear + Sigmoid model on the toy AND-gate data.

    Thin wrapper around fit() that keeps the interface required by
    the tests: it returns only the loss history. Use fit() when you
    also need the trained model.

    Args:
        epochs (int): number of full-batch training iterations.
        lr (float): learning rate passed to the SGD optimizer.
        seed (int): seed for NumPy's random generator, so the
            Linear layer's Xavier initialization is reproducible.

    Returns:
        list[float]: the loss recorded at every epoch, in order.
    """
    _, loss_history = fit(epochs=epochs, lr=lr, seed=seed)
    return loss_history


def accuracy(loss_history: list[float] = None, model: Model = None) -> float:
    """Report classification accuracy on the toy dataset.

    Works with no arguments and without any earlier train() call:
    when no model is given, it trains one with fit()'s default
    settings and scores that. To score a specific run instead, pass
    the model returned by fit().

    Args:
        loss_history (list[float]): optional loss history of the run
            being scored. Its only job is a sanity check: if the
            final loss is not finite the run diverged, and a
            reported accuracy would be meaningless, so a ValueError
            is raised instead. Accuracy itself is computed from the
            model's predictions, never from the losses.
        model (Model): optional trained model to score. If omitted,
            a fresh model is trained with fit()'s defaults.

    Returns:
        float: fraction of the toy dataset's examples classified
            correctly (probability >= 0.5 counts as class 1).

    Raises:
        ValueError: if loss_history ends in a non-finite loss.
    """
    if loss_history is not None and len(loss_history) > 0:
        if not np.isfinite(loss_history[-1]):
            raise ValueError("training diverged (non-finite loss)")
    if model is None:
        model, _ = fit()
    X, y = toy_data()
    predicted_labels = (model.predict(X) >= 0.5).astype(float)
    return float(np.mean(predicted_labels == y))


if __name__ == "__main__":
    trained_model, history = fit()
    for epoch in (0, 999, 1999, 2999, 3999):
        if epoch < len(history):
            print(f"epoch {epoch}: loss = {history[epoch]:.6f}")
    print(f"final accuracy: {accuracy(history, model=trained_model):.2%}")
