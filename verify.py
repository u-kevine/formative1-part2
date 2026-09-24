"""Ad-hoc verification: reproduces the guide's worked examples plus numerical
gradient checks. The real assignment ships pytest test files under tests/;
this script exists only because that directory wasn't provided here."""

import numpy as np

from nn.layers import Linear
from nn.activations import ReLU, Sigmoid, Softmax
from nn.losses import CrossEntropyLoss, CategoricalCrossEntropyLoss
from nn.optim import SGD

np.set_printoptions(precision=4, suppress=True)
failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# --- Chapter 1: single neuron forward ---
lin = Linear(3, 1)
lin.W = np.array([[0.5], [1.0], [-0.5]])
lin.b = np.array([0.2])
x = np.array([[2.0, -1.0, 3.0]])
z = lin.forward(x)
check("Ch1 forward value == -1.3", np.allclose(z, [[-1.3]]))
check("Ch1 forward shape == (1,1)", z.shape == (1, 1))

# Xavier init sanity (separate layer, not overwritten)
lin2 = Linear(3, 5)
bound = np.sqrt(6.0 / (3 + 5))
check("Xavier: W shape correct", lin2.W.shape == (3, 5))
check("Xavier: W not all zero", not np.allclose(lin2.W, 0))
check("Xavier: within bound", np.all(np.abs(lin2.W) <= bound + 1e-9))
check("Xavier: b is zeros", np.allclose(lin2.b, 0) and lin2.b.shape == (5,))

# --- Chapter 2: multi-class logits + softmax forward ---
lin3 = Linear(3, 3)
lin3.W = np.array([[0.5, -0.3, 0.1], [1.0, 0.2, -0.6], [-0.5, 0.4, 0.3]])
lin3.b = np.array([0.2, -0.1, 0.05])
z3 = lin3.forward(x)
check("Ch2 logits", np.allclose(z3, [[-1.3, 0.3, 1.75]], atol=1e-6))

sm = Softmax()
a3 = sm.forward(z3)
expected_probs = np.array([[0.0369, 0.1830, 0.7801]])
check("Ch2 softmax probs", np.allclose(a3, expected_probs, atol=1e-3))
check("Ch2 softmax sums to 1", np.allclose(np.sum(a3, axis=1), 1.0))

big = np.array([[1000.0, 1.0, 0.0]])
sm_big = Softmax()
out_big = sm_big.forward(big)
check("Ch2 softmax stable on large logits", np.all(np.isfinite(out_big)))

# --- Chapter 3: Linear backward via gradient checking ---
def numerical_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = f()
        x[idx] = orig - eps
        minus = f()
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


np.random.seed(0)
lin4 = Linear(4, 3)
X4 = np.random.randn(5, 4)
upstream = np.random.randn(5, 3)


def scalar_loss():
    return np.sum(lin4.forward(X4) * upstream)


grad_out_dummy = upstream
lin4.forward(X4)
dX_analytic = lin4.backward(upstream)

num_dW = numerical_grad(lambda: np.sum(lin4.forward(X4) * upstream), lin4.W)
num_db = numerical_grad(lambda: np.sum(lin4.forward(X4) * upstream), lin4.b)
num_dX = numerical_grad(lambda: np.sum(lin4.forward(X4) * upstream), X4)
lin4.forward(X4)
lin4.backward(upstream)
check("Ch3 dW gradient check", np.allclose(lin4.dW, num_dW, atol=1e-4))
check("Ch3 db gradient check", np.allclose(lin4.db, num_db, atol=1e-4))
check("Ch3 dX gradient check", np.allclose(dX_analytic, num_dX, atol=1e-4))

# in-place buffer check (Chapter 3 "why backward must write in place")
dW_ref = lin4.dW
lin4.forward(X4)
lin4.backward(upstream)
check("Ch3 dW written in place (same object)", lin4.dW is dW_ref)

# --- Chapter 4: ReLU ---
relu = ReLU()
zr = np.array([[-2.0, 0.0, 3.0]])
ar = relu.forward(zr)
check("Ch4 forward", np.allclose(ar, [[0, 0, 3]]))
grad_up = np.array([[0.5, -0.2, 0.1]])
gr = relu.backward(grad_up)
check("Ch4 backward", np.allclose(gr, [[0, 0, 0.1]]))

# --- Chapter 5: Sigmoid ---
sig = Sigmoid()
a_sig = sig.forward(np.array([[-1.3]]))
check("Ch5 forward value", np.allclose(a_sig, [[0.2142]], atol=1e-3))
g_sig = sig.backward(np.array([[1.0]]))
check("Ch5 backward value", np.allclose(g_sig, [[0.1683]], atol=1e-3))

# --- Chapter 6: binary cross-entropy + a-y shortcut ---
bce = CrossEntropyLoss()
sig2 = Sigmoid()
a2 = sig2.forward(np.array([[-1.3]]))
y2 = np.array([[1.0]])
loss_val = bce.forward(a2, y2)
check("Ch6 loss value", np.isclose(loss_val, 1.5410, atol=1e-3))
d_loss = bce.backward()
d_z = sig2.backward(d_loss)
check("Ch6 a-y shortcut", np.allclose(d_z, a2 - y2, atol=1e-6))

# --- Chapter 7: softmax backward via gradient check ---
np.random.seed(1)
sm2 = Softmax()
logits = np.random.randn(4, 3)
upstream2 = np.random.randn(4, 3)
sm2.forward(logits)
grad_analytic = sm2.backward(upstream2)


def sm_loss():
    return np.sum(Softmax().forward(logits) * upstream2)


num_grad_softmax = numerical_grad(sm_loss, logits)
check("Ch7 softmax gradient check", np.allclose(grad_analytic, num_grad_softmax, atol=1e-4))

# --- Chapter 8: categorical cross-entropy + a-y shortcut ---
sm3 = Softmax()
logits3 = np.array([[-1.3, 0.3, 1.75]])
a_sm3 = sm3.forward(logits3)
y3 = np.array([[0.0, 0.0, 1.0]])
cce = CategoricalCrossEntropyLoss()
loss3 = cce.forward(a_sm3, y3)
check("Ch8 loss value", np.isclose(loss3, 0.2484, atol=1e-3))
d_a3 = cce.backward()
d_z3 = sm3.backward(d_a3)
expected_dz3 = np.array([[0.0369, 0.1830, -0.2199]])
check("Ch8 a-y shortcut", np.allclose(d_z3, expected_dz3, atol=1e-3))

# --- Chapter 9: SGD ---
p = np.array([1.0, 2.0])
g = np.array([0.1, 0.2])
opt = SGD([(p, g)], lr=1.0)
opt.step()
check("Ch9 SGD step direction/amount", np.allclose(p, [0.9, 1.8]))
opt.zero_grad()
check("Ch9 SGD zero_grad", np.allclose(g, [0.0, 0.0]))

# --- Chapter 10: full training loop on AND-gate toy data ---
import main as m

history = m.train(epochs=4000, lr=1.0, seed=0)
check("Ch10 loss decreases", history[-1] < history[0])
check("Ch10 loss near zero", history[-1] < 0.05)
acc = m.accuracy(history)
check("Ch10 accuracy == 100%", acc == 1.0)
print(f"\nfinal training loss: {history[-1]:.6f}, accuracy: {acc:.2%}")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}"))
