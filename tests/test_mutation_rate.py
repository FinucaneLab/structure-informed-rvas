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


def _rejection_rates(x, m, lambda_hat, adj, region, n_g_from_sim=True):
    """Run both tests on each simulated replicate and pool p-values over `region`."""
    p_a_all, p_b_all = [], []
    for s in range(x.shape[1]):
        col = x[:, s:s + 1]
        n_g = int(col.sum())
        if n_g == 0:
            continue
        p_a, p_b, _, _, _ = _pvals_for_radius(adj, col, col, m, lambda_hat, n_g, float(m.sum()))
        p_a_all.append(p_a[region, 0])
        p_b_all.append(p_b[region, 0])
    return np.concatenate(p_a_all), np.concatenate(p_b_all)


# ---------------------------------------------------------------------------
def test_gene_level_inflation():
    """
    The mirror of the regional-constraint case. A gene uniformly enriched 3x with no
    spatial clustering: Test A fires everywhere (it is measuring the gene-level excess),
    Test B stays calibrated, and so does max(p_A, p_B).
    """
    print('\n5. Gene-level inflation (mirror case)')
    n_res, n_sims = 300, 2000
    adj = banded_adjacency(n_res, 8)
    rng = np.random.default_rng(11)
    m = np.full((n_res, 1), 1.0)
    lambda_hat = 0.2

    x = rng.poisson(3.0 * lambda_hat * m, size=(n_res, n_sims))
    region = slice(12, n_res - 12)
    p_a, p_b = _rejection_rates(x, m, lambda_hat, adj, region)

    rate_a = float((p_a <= 0.05).mean())
    rate_b = float((p_b <= 0.05).mean())
    rate_max = float((np.maximum(p_a, p_b) <= 0.05).mean())
    print(f'      rejection at alpha=0.05:  Test A {rate_a:.4f}   Test B {rate_b:.4f}'
          f'   max {rate_max:.4f}')
    check('Test A is inflated by gene-level enrichment', rate_a > 0.20, f'rate {rate_a:.4f}')
    check('Test B stays valid under gene-level enrichment', rate_b <= 0.05 * 1.10,
          f'rate {rate_b:.4f}')
    check('max(p_A, p_B) stays valid', rate_max <= 0.05 * 1.10, f'rate {rate_max:.4f}')


# ---------------------------------------------------------------------------
def test_injected_cluster_recovery():
    """
    Power check: a genuine cluster on top of each of the two problem backgrounds must be
    detected by both tests, so that requiring both does not cost us real signal.
    """
    print('\n6. Injected cluster recovery (power)')
    n_res, n_sims = 400, 400
    half = 10
    adj = banded_adjacency(n_res, half)
    rng = np.random.default_rng(12)
    m = np.full((n_res, 1), 1.0)
    lambda_hat = 0.25
    centre = 300
    window = np.zeros((n_res, 1))
    window[centre - half: centre + half + 1] = 1.0

    backgrounds = {
        'flat': np.ones((n_res, 1)),
        'regional constraint elsewhere': np.where(np.arange(n_res)[:, None] < 0.4 * n_res, 0.2, 1.0),
        'gene-level 3x inflation': np.full((n_res, 1), 3.0),
    }

    for label, bg in backgrounds.items():
        rate = lambda_hat * m * bg + 4.0 * lambda_hat * m * window
        x = rng.poisson(rate, size=(n_res, n_sims))
        p_a, p_b = _rejection_rates(x, m, lambda_hat, adj, slice(centre, centre + 1))
        power_max = float((np.maximum(p_a, p_b) <= 0.05).mean())
        print(f'      {label:32s} power(max) = {power_max:.3f}  '
              f'A {float((p_a <= 0.05).mean()):.3f}  B {float((p_b <= 0.05).mean()):.3f}')
        check(f'cluster recovered on background: {label}', power_max > 0.80,
              f'power {power_max:.3f}')


# ---------------------------------------------------------------------------
def test_depletion_direction():
    """
    The depletion tail must be valid under the null, and must fire on depleted data
    while the enrichment tail does not.
    """
    print('\n7. Depletion direction')
    n_res, n_sims = 150, 3000
    adj = banded_adjacency(n_res, 6)
    rng = np.random.default_rng(21)
    m = rng.uniform(0.2, 3.0, size=(n_res, 1))
    lambda_hat = 0.8

    x_null = simulate_poisson_null(m, lambda_hat, n_sims, rng)
    n_g = int(round(lambda_hat * m.sum()))
    x_b = simulate_multinomial_null(m, n_g, n_sims, rng)
    p_a, p_b, _, _, _ = _pvals_for_radius(adj, x_null, x_b, m, lambda_hat, n_g,
                                          float(m.sum()), 'depletion')
    for alpha in (0.05, 0.01):
        ra, rb = float((p_a <= alpha).mean()), float((p_b <= alpha).mean())
        check(f'depletion Test A valid at alpha={alpha}', ra <= alpha * 1.05, f'rate {ra:.4f}')
        check(f'depletion Test B valid at alpha={alpha}', rb <= alpha * 1.05, f'rate {rb:.4f}')

    # a genuinely depleted region: depletion fires, enrichment does not
    depleted = np.ones((n_res, 1))
    depleted[40:90] = 0.15
    x = rng.poisson(lambda_hat * m * depleted, size=(n_res, 200))
    region = slice(55, 75)
    pd_a, _, _, _, _ = _pvals_for_radius(adj, x, x, m, lambda_hat,
                                         int(x[:, 0].sum()), float(m.sum()), 'depletion')
    pe_a, _, _, _, _ = _pvals_for_radius(adj, x, x, m, lambda_hat,
                                         int(x[:, 0].sum()), float(m.sum()), 'enrichment')
    dep = float((pd_a[region] <= 0.05).mean())
    enr = float((pe_a[region] <= 0.05).mean())
    print(f'      in the depleted region:  depletion {dep:.3f}   enrichment {enr:.3f}')
    check('depletion tail detects a depleted region', dep > 0.80, f'rate {dep:.3f}')
    check('enrichment tail does not', enr < 0.05, f'rate {enr:.3f}')


# ---------------------------------------------------------------------------
def test_min_denovo_selection_bias():
    """
    --min-denovo is selection on the outcome: it keeps genes whose count came out high.
    Test A's null must be conditioned on the same filter or the observed data sits above
    its own null. Test B is immune because it conditions on N_g.

    Regression guard for the real finding: on the simulated-null ASD run, Test A's
    per-gene FWER was 14.3% instead of 5%, with selected genes showing a median
    observed/expected gene total of 1.50.
    """
    print('\n8. --min-denovo selection bias in the Test A null')
    n_res, n_sims, min_denovo = 250, 600, 5
    adj = banded_adjacency(n_res, 8)
    rng = np.random.default_rng(11)

    hits = {False: 0, True: 0}
    n_kept = 0
    ratio = 0.0
    tried = 0
    while n_kept < 120 and tried < 100000:
        tried += 1
        m = rng.uniform(0.2, 3.0, size=(n_res, 1))
        lam = 4.0 / m.sum()            # expected total ~4, so ">5" really is selective
        obs = rng.poisson(lam * m, size=(n_res, 1))
        if obs.sum() <= min_denovo:
            continue
        n_kept += 1
        n_g = int(obs.sum())
        ratio += n_g / (lam * m.sum())
        for cond in (False, True):
            nul = simulate_poisson_null(m, lam, n_sims, rng, min_denovo if cond else None)
            x = np.hstack([obs, nul])
            p_a, _, _, _, _ = _pvals_for_radius(adj, x, x, m, lam, n_g, float(m.sum()))
            o = p_a[:, 0].min()
            hits[cond] += (p_a[:, 1:].min(axis=0) <= o).mean() < 0.05

    unc = 100 * hits[False] / n_kept
    con = 100 * hits[True] / n_kept
    print(f'      {n_kept} genes kept, mean observed/expected {ratio / n_kept:.2f}; '
          f'per-gene FWER unconditioned {unc:.1f}%, conditioned {con:.1f}%')
    check('unconditioned Test A null is inflated by the gene filter', unc > 9.0,
          f'{unc:.1f}%')
    check('conditioning on the filter restores calibration', con <= 9.0, f'{con:.1f}%')


if __name__ == '__main__':
    test_uniform_rate_degeneracy()
    test_lookup_equivalence()
    test_null_calibration()
    test_regional_constraint()
    test_gene_level_inflation()
    test_injected_cluster_recovery()
    test_depletion_direction()
    test_min_denovo_selection_bias()
    print('\n' + ('ALL TESTS PASSED' if not FAILURES else f'FAILURES: {FAILURES}'))
    sys.exit(1 if FAILURES else 0)
