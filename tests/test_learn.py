import numpy as np
import pytest

torch = pytest.importorskip("torch")

from delivery_lab.learn.fit import Data, fit, gaussian_nll, mse  # noqa: E402
from delivery_lab.learn.nn import Net, NumpyNet  # noqa: E402
from delivery_lab.learn.optim import Adam  # noqa: E402


def small_net(dtype=torch.float64, ids=True):
    return Net(n_features=4, hidden=(5, 3), n_out=2, n_ids=7 if ids else 0, id_dim=3 if ids else 0, seed=1, dtype=dtype)


def batch(rng, n=12, ids=True):
    x = torch.tensor(rng.normal(size=(n, 4)))
    y = torch.tensor(rng.normal(size=(n, 1)))
    return x, y, (torch.tensor(rng.integers(0, 7, n)) if ids else None)


@pytest.mark.parametrize("with_ids", [True, False])
def test_autograd_gradient_matches_finite_differences(with_ids):
    rng = np.random.default_rng(0)
    net = small_net(ids=with_ids)
    x, y, ids = batch(rng, ids=with_ids)

    loss = gaussian_nll(net.forward(x, ids), y)
    for p in net.parameters():
        p.grad = None
    loss.backward()

    eps = 1e-6
    checked = 0
    for p in net.parameters():
        flat = p.detach().view(-1)
        grad = p.grad.view(-1)
        for i in rng.choice(len(flat), size=min(6, len(flat)), replace=False):
            old = float(flat[i])
            flat[i] = old + eps
            up = float(gaussian_nll(net.forward(x, ids), y).detach())
            flat[i] = old - eps
            down = float(gaussian_nll(net.forward(x, ids), y).detach())
            flat[i] = old
            numeric = (up - down) / (2 * eps)
            assert float(grad[i]) == pytest.approx(numeric, rel=1e-5, abs=1e-8)
            checked += 1
    assert checked >= 24


def test_numpy_inference_matches_the_torch_forward():
    rng = np.random.default_rng(1)
    net = small_net(dtype=torch.float32)
    x, _, ids = batch(rng, n=50)
    want = net.forward(x.float(), ids).detach().numpy()
    got = NumpyNet(net.state())(x.numpy(), ids.numpy())
    assert np.allclose(got, want, atol=1e-5)


def test_adam_matches_the_textbook_update_on_one_parameter():
    p = torch.tensor([1.0], requires_grad=True)
    opt = Adam([p], lr=0.1)
    (p**2).sum().backward()  # gradiente 2
    opt.step()
    # primeiro passo do Adam: m_hat = g, v_hat = g^2, passo = lr * g / (|g| + eps) ~= lr
    assert float(p) == pytest.approx(1.0 - 0.1, abs=1e-6)
    opt.zero_grad()
    assert p.grad is None


def test_fit_learns_a_known_function_and_keeps_the_best_epoch():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, (4000, 3)).astype(np.float32)
    y = (np.sin(2 * x[:, :1]) + 0.5 * x[:, 1:2] * x[:, 2:3]).astype(np.float32)
    train, val = Data(x[:3000], y[:3000]), Data(x[3000:], y[3000:])
    net = Net(3, (24, 24), 1, seed=0)
    history = fit(net, mse, train, val, epochs=25, batch=256, lr=1e-2, lr_end=1e-3, seed=0)
    assert history.best_val < 0.1 * float(val.y.var())  # explica mais de 90% da variância do alvo
    pred = NumpyNet(net.state())(val.x)
    assert float(np.mean((pred - val.y) ** 2)) == pytest.approx(history.best_val, rel=0.05)


def test_gaussian_nll_is_minimized_by_the_true_mean_and_spread():
    rng = np.random.default_rng(3)
    y = torch.tensor(rng.normal(2.0, 0.5, (20000, 1)), dtype=torch.float64)

    def nll(mu, sigma):
        out = torch.stack([torch.full((len(y),), mu), torch.full((len(y),), float(np.log(sigma)))], dim=1)
        return float(gaussian_nll(out, y))

    assert nll(2.0, 0.5) < nll(2.3, 0.5) and nll(2.0, 0.5) < nll(2.0, 1.0) and nll(2.0, 0.5) < nll(2.0, 0.25)
