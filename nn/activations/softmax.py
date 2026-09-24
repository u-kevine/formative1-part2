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
            self.a (np.ndarray): a private copy of the output
                probabilities, saved so backward can reuse them.
                It is a copy, not the returned array, so a caller
                who modifies the returned probabilities in place
                cannot corrupt the cache backward depends on.
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        a = exp / np.sum(exp, axis=1, keepdims=True)
        self.a = a.copy()
        return a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        For one example, the softmax Jacobian is
        d a_c / d z_k = a_c * (delta_ck - a_k), i.e.
        J = diag(a) - a a^T, which is symmetric. Applying it to the
        upstream gradient g gives

            dL/dz_k = a_k * (g_k - sum_c g_c * a_c),

        so the whole batch needs only one row-wise reduction (the
        sum over c) and elementwise arithmetic -- no explicit
        (C, C) Jacobian and no Python loop over examples.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, shape
                (batch_size, C). Any numeric dtype, including
                integer arrays.

        Returns:
            np.ndarray: gradient of the loss with respect to
                this layer's input (the logits), shape
                (batch_size, C). Always floating point: its dtype
                comes from the saved probabilities, never from
                grad_output.
        """
        weighted = np.sum(grad_output * self.a, axis=1, keepdims=True)
        return self.a * (grad_output - weighted)
