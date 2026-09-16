"""
Correctness tests for the mutation-rate 3DNT.

Run: python tests/test_mutation_rate.py
"""
import os
import sys
import numpy as np
from scipy.stats import poisson, binom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mutation_rate_test import (
    _pvals_for_radius, poisson_pval_lookup, binomial_pval_lookup,
    simulate_poisson_null, simulate_multinomial_null,
)

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def banded_adjacency(n_res, half_width):
    """A simple 1-D 'structure': residue i is adjacent to those within half_width."""
    idx = np.arange(n_res)
    return (np.abs(idx[:, None] - idx[None, :]) <= half_width).astype(int)


# ---------------------------------------------------------------------------
def test_uniform_rate_degeneracy():
    """With a flat mutation rate, Test B must reduce to Binomial(N_g, |N(i)| / n_res)."""
    print('\n1. Uniform-rate degeneracy (Test B -> plain binomial)')
    n_res, n_g = 200, 30
    adj = banded_adjacency(n_res, 7)
    m = np.ones((n_res, 1))
    rng = np.random.default_rng(0)

    x = rng.multinomial(n_g, np.full(n_res, 1.0 / n_res), size=1).T
    _, p_b, x_nbhd, _, exp_b = _pvals_for_radius(adj, x, x, m, 1.0, n_g, float(m.sum()))

    nbhd_size = adj.sum(axis=1)
    expected_p = binom.sf(x_nbhd - 1, n_g, nbhd_size / n_res)
    check('p_B equals binom.sf(X-1, N_g, |N(i)|/n_res)',
          np.allclose(p_b[:, 0], expected_p, rtol=1e-12, atol=1e-15),
          f'max abs diff {np.abs(p_b[:, 0] - expected_p).max():.2e}')
    check('expected count equals N_g * |N(i)| / n_res',
          np.allclose(exp_b, n_g * nbhd_size / n_res))


# ---------------------------------------------------------------------------
def test_lookup_equivalence():
    """Table lookup must match direct elementwise evaluation."""
    print('\n2. Lookup-table equivalence')
    rng = np.random.default_rng(1)
    n_res = 120
    expected = rng.uniform(0.01, 8.0, size=n_res)
    x = rng.integers(0, 12, size=(n_res, 40))
    rows = np.arange(n_res)[:, None]

    lut = poisson_pval_lookup(expected, int(x.max()))
    check('poisson lookup == poisson.sf',
          np.allclose(lut[rows, x], poisson.sf(x - 1, expected[:, None]), rtol=1e-12))

    p_frac = rng.uniform(0.001, 0.4, size=n_res)
    lutb = binomial_pval_lookup(p_frac, 25, int(x.max()))
    check('binomial lookup == binom.sf',
          np.allclose(lutb[rows, x], binom.sf(x - 1, 25, p_frac[:, None]), rtol=1e-12))


# ---------------------------------------------------------------------------
def test_null_calibration():
    """
    Under each test's own null the p-values must be valid: P(p <= alpha) <= alpha.
    Exact equality is not expected because the counts are discrete, so these are
    one-sided validity checks, not uniformity checks.
    """
    print('\n3. Null calibration')
    n_res, n_sims = 150, 4000
    adj = banded_adjacency(n_res, 6)
    rng = np.random.default_rng(2)
    m = rng.uniform(0.2, 3.0, size=(n_res, 1))
    lambda_hat = 0.4

    x_a = simulate_poisson_null(m, lambda_hat, n_sims, rng)
    n_g = int(round(lambda_hat * m.sum()))
    x_b = simulate_multinomial_null(m, n_g, n_sims, rng)

    check('multinomial null preserves N_g', bool(np.all(x_b.sum(axis=0) == n_g)))

    p_a, p_b, _, _, _ = _pvals_for_radius(adj, x_a, x_b, m, lambda_hat, n_g, float(m.sum()))

    for alpha in (0.05, 0.01):
        rate_a = float((p_a <= alpha).mean())
        rate_b = float((p_b <= alpha).mean())
        check(f'Test A valid at alpha={alpha}', rate_a <= alpha * 1.05, f'rate {rate_a:.4f}')
        check(f'Test B valid at alpha={alpha}', rate_b <= alpha * 1.05, f'rate {rate_b:.4f}')


# ---------------------------------------------------------------------------
def test_regional_constraint():
    """
    The key experiment. A gene where a contiguous region is lethally constrained, with
    no true clustering anywhere: Test B should produce false positives in the
    unconstrained region, and Test A should not.
    """
    print('\n4. Regional constraint (the reason Test A exists)')
    n_res, n_sims = 400, 3000
    adj = banded_adjacency(n_res, 10)
    rng = np.random.default_rng(3)

    m = np.full((n_res, 1), 1.0)
    lambda_hat = 0.25

    # 40% of the protein is lethally constrained: de novos arise there at 0.2x rate.
    depletion = np.ones((n_res, 1))
    depletion[: int(0.4 * n_res)] = 0.2

    x = rng.poisson(lambda_hat * m * depletion, size=(n_res, n_sims))

    # Both tests see the same observed data; only their nulls differ.
    n_g_per_sim = x.sum(axis=0)
    p_a_all, p_b_all = [], []
    for s in range(n_sims):
        col = x[:, s:s + 1]
        n_g = int(col.sum())
        if n_g == 0:
            continue
        p_a, p_b, _, _, _ = _pvals_for_radius(adj, col, col, m, lambda_hat, n_g, float(m.sum()))
        unconstrained = slice(int(0.4 * n_res) + 12, n_res - 12)
        p_a_all.append(p_a[unconstrained, 0])
        p_b_all.append(p_b[unconstrained, 0])

    p_a_all = np.concatenate(p_a_all)
    p_b_all = np.concatenate(p_b_all)
    rate_a = float((p_a_all <= 0.05).mean())
    rate_b = float((p_b_all <= 0.05).mean())

    print(f'      unconstrained-region rejection rate at alpha=0.05:'
          f'  Test A {rate_a:.4f}   Test B {rate_b:.4f}')
    check('Test B is inflated by regional constraint', rate_b > 0.10,
          f'rate {rate_b:.4f} (this inflation is the failure mode Test A guards against)')
    check('Test A stays valid under regional constraint', rate_a <= 0.05 * 1.10,
          f'rate {rate_a:.4f}')
    rate_max = float((np.maximum(p_a_all, p_b_all) <= 0.05).mean())
    check('max(p_A, p_B) stays valid', rate_max <= 0.05 * 1.10, f'rate {rate_max:.4f}')


if __name__ == '__main__':
    test_uniform_rate_degeneracy()
    test_lookup_equivalence()
    test_null_calibration()
    test_regional_constraint()
    print('\n' + ('ALL TESTS PASSED' if not FAILURES else f'FAILURES: {FAILURES}'))
    sys.exit(1 if FAILURES else 0)
