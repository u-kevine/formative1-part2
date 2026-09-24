import numpy as np
from nn.activations.softmax import Softmax

softmax = Softmax()
z = np.array([[-1.3, 0.3, 1.75]])
output = softmax.forward(z)

print(output)
print(output.sum(axis=1))