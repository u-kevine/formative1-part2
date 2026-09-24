"""Categorical cross-entropy loss, for one-hot multi-class targets."""

import numpy as np

_EPS = 1e-12


class CategoricalCrossEntropyLoss:
    """Categorical cross-entropy loss over C classes.

    Does not subclass Module -- see the note in Chapter 6.
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average categorical cross-entropy loss.

        Args:
            predictions (np.ndarray): softmax probabilities,
                shape (m, C). Clipped away from exactly 0 before
                use.
            targets (np.ndarray): one-hot true labels, shape
                (m, C).

        Returns:
            float: the scalar loss, averaged over the batch.

        Sets:
            self.predictions (np.ndarray): the clipped predictions,
                reused by backward.
            self.targets (np.ndarray): a private copy of the
                targets, reused by backward, so later in-place edits
                by the caller cannot change the gradient.
        """
        self.predictions = np.clip(predictions, _EPS, 1.0 - _EPS)
        self.targets = np.copy(targets)
        m = self.predictions.shape[0]
        loss = -np.sum(self.targets * np.log(self.predictions)) / m
        return float(loss)

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
            np.ndarray: dL/da, shape (m, C), same shape as the
                predictions passed to forward. Uses the same
                clipping as forward.
        """
        m = self.predictions.shape[0]
        return -(self.targets / self.predictions) / m
