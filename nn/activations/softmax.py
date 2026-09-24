"""Softmax activation: converts logits into a probability distribution."""

import numpy as np

from nn.module import Module


class Softmax(Module):
    """Softmax activation, applied row-wise to a batch of logits.

    Unlike ReLU or Sigmoid, each output depends on every logit in
    its own row, not just the matching input -- see "A shape
    subtlety" in Chapter 2.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for a batch of logits.

        Args:
            x (np.ndarray): logits, shape (batch_size, C).

        Returns:
            np.ndarray: probabilities, shape (batch_size, C).
                Each row sums to 1.

        Sets:
            self.a (np.ndarray): the output probabilities, saved
                so backward can reuse them when building the
                per-row Jacobian.
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.a = exp / np.sum(exp, axis=1, keepdims=True)
        return self.a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Uses the per-example Jacobian
        d a_c / d z_k = a_c * (delta_ck - a_k), applied one row
        at a time.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, shape
                (batch_size, C).

        Returns:
            np.ndarray: gradient of the loss with respect to
                this layer's input (the logits), shape
                (batch_size, C).
        """
        batch_size, num_classes = self.a.shape
        grad_input = np.empty_like(grad_output)
        for i in range(batch_size):
            a_i = self.a[i].reshape(-1, 1)  # (C, 1)
            jacobian = np.diagflat(a_i) - a_i @ a_i.T  # (C, C)
            grad_input[i] = jacobian @ grad_output[i]
        return grad_input
