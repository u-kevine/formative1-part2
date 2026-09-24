"""Sigmoid activation: squashes real values into (0, 1)."""

import numpy as np

from nn.module import Module


class Sigmoid(Module):
    """Sigmoid activation, applied elementwise: 1 / (1 + e^{-x})."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid elementwise, and remember the output.

        Args:
            x (np.ndarray): input, any shape.

        Returns:
            np.ndarray: sigmoid(x), elementwise, same shape as x.

        Sets:
            self.a (np.ndarray): a private copy of the output,
                saved so backward can reuse it via a(1 - a) instead
                of recomputing the exponential. It is a copy, not
                the returned array, so a caller who modifies the
                returned values in place cannot corrupt the cache.
        """
        # np.exp(-x) overflows for very negative-of-negative x (i.e. large
        # positive x) and np.exp(x) overflows for very negative x. Route
        # each element through whichever expression only ever exponentiates
        # a non-positive number, so no branch can overflow.
        positive = x >= 0
        exp_term = np.exp(np.where(positive, -x, x))
        a = np.where(positive, 1.0 / (1.0 + exp_term), exp_term / (1.0 + exp_term))
        self.a = a.copy()
        return a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients given the upstream gradient.

        Args:
            grad_output (np.ndarray): gradient of the loss with
                respect to this layer's output, same shape as
                the original input to forward.

        Returns:
            np.ndarray: gradient of the loss with respect to
                this layer's input, same shape as grad_output.
        """
        return grad_output * self.a * (1.0 - self.a)
