"""Binary cross-entropy loss."""

import numpy as np

_EPS = 1e-12


class CrossEntropyLoss:
    """Binary cross-entropy loss for a single output probability.

    Does not subclass Module -- see the note in Chapter 6 (its
    forward/backward signatures don't match the Module contract,
    and it has no learnable parameters).
    """

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the average binary cross-entropy loss.

        Args:
            predictions (np.ndarray): predicted probabilities,
                shape (m,) or (m, 1). Clipped away from exactly
                0 or 1 before use.
            targets (np.ndarray): true labels, same shape as
                predictions, values 0 or 1.

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
        loss = -np.sum(
            self.targets * np.log(self.predictions)
            + (1 - self.targets) * np.log(1 - self.predictions)
        ) / m
        return float(loss)

    def backward(self) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. predictions.

        Returns:
            np.ndarray: dL/da, same shape as the predictions
                passed to forward. Uses the same clipped
                predictions as forward.
        """
        m = self.predictions.shape[0]
        return -(
            self.targets / self.predictions
            - (1 - self.targets) / (1 - self.predictions)
        ) / m
