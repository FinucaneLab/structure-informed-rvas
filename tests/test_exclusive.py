import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmh import cmh_expected_variance, cmh_pvals_by_column
from utils import fisher_greater_pvals_by_column, exclusive_case_control


def make_df(rows):
    return pd.DataFrame(rows, columns=["aa_pos", "aa_alt", "ac_case", "ac_control", "ancestry"])


def test_all_singletons_totals():
    df = make_df([(i + 1, "A", int(i % 2 == 0), int(i % 2 == 1), "x") for i in range(6)])
    case, ctrl = exclusive_case_control(df, 6, 10, seed=1)
    assert (case[0].sum(0) == 3).all() and (ctrl[0].sum(0) == 3).all()
    assert (case[0, :, 0] == [1, 0, 1, 0, 1, 0]).all()


def test_shared_variant_dropped_observed_and_null_conditionally():
    df = make_df([(1, "A", 2, 1, "x"), (2, "A", 3, 0, "x"), (3, "A", 0, 4, "x")])
    case, ctrl = exclusive_case_control(df, 3, 300, seed=2)
    assert case[0, 0, 0] == 0 and ctrl[0, 0, 0] == 0
    assert case[0, 1, 0] == 3 and ctrl[0, 2, 0] == 4
    kept = (case[0, 0] + ctrl[0, 0]) > 0
    assert kept.any() and not kept.all()
    assert ((case[0, 0, kept] == 0) | (ctrl[0, 0, kept] == 0)).all()


def test_keep_masks_every_column():
    df = make_df([(1, "A", 1, 0, "x"), (2, "A", 0, 1, "x"), (3, "A", 1, 0, "x")])
    keep = np.array([1, 0, 1])
    case, ctrl = exclusive_case_control(df, 3, 10, seed=1, keep=keep)
    assert (case[0, 1] == 0).all() and (ctrl[0, 1] == 0).all()


def test_pooled_over_strata():
    df = make_df([(1, "A", 2, 0, "a"), (1, "A", 0, 2, "b"), (2, "A", 1, 0, "a"), (3, "A", 0, 1, "b")])
    case, ctrl = exclusive_case_control(df, 3, 5, seed=3, stratum_col="ancestry")
    assert case.shape == (2, 3, 6)
    assert case[:, 0, 0].sum() == 0 and ctrl[:, 0, 0].sum() == 0


def test_same_seed_identical():
    df = make_df([(1, "A", 2, 1, "x"), (2, "A", 3, 2, "x"), (3, "B", 1, 4, "x")])
    a = exclusive_case_control(df, 3, 50, seed=7)
    b = exclusive_case_control(df, 3, 50, seed=7)
    assert all((x == y).all() for x, y in zip(a, b))


def test_fisher_by_column_matches_hypergeom():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 6, (20, 4)), rng.integers(0, 6, (20, 4))
    nc, nn = np.array([30, 30, 40, 25]), np.array([50, 50, 45, 60])
    got = fisher_greater_pvals_by_column(a, b, nc, nn)
    want = hypergeom.sf(a - 1, nc + nn, nc, a + b)
    enriched = (a > 0) & (a * nn > b * nc)
    assert np.allclose(got, np.where(enriched, want, 1.0))


def test_fisher_by_column_bitwise_with_repeated_float_margins():
    # The packed-key dedupe must hand hypergeom.sf exactly the per-cell arguments.
    rng = np.random.default_rng(1)
    nc = rng.choice([1, 7, 40, 300], 50).astype(float)
    nn = rng.choice([1, 9, 60, 900], 50).astype(float)
    a = np.minimum(rng.integers(0, 40, (30, 50)), nc.astype(int))
    b = np.minimum(rng.integers(0, 40, (30, 50)), nn.astype(int))
    a[0], b[0], b[1] = nc, 0, nn  # a = whole margin, b = 0, b = whole margin
    got = fisher_greater_pvals_by_column(a, b, nc, nn)
    enriched = (a > 0) & (a * nn > b * nc)
    assert np.array_equal(got, np.where(enriched, hypergeom.sf(a - 1, nc + nn, nc, a + b), 1.0))
    assert (got[0] < 1).any() and (got == 1).any()


def test_cmh_by_column_matches_expected_variance():
    rng = np.random.default_rng(0)
    n_res = 8
    adj = (rng.random((n_res, n_res)) < 0.4).astype(int)
    case = rng.integers(0, 3, (2, n_res, 3)).astype(np.int32)
    ctrl = rng.integers(0, 3, (2, n_res, 3)).astype(np.int32)
    ctrl[:, :, 1] = ctrl[:, :, 0]
    case[:, :, 1] = case[:, :, 0]  # constant margins in cols 0 and 1
    pvals, stat, e0, v0 = cmh_pvals_by_column(adj, case, ctrl)
    inside = np.stack([adj @ (case[s, :, 0] + ctrl[s, :, 0]) for s in range(2)], axis=1)
    e, v = cmh_expected_variance(inside, case[:, :, 0].sum(1), (case + ctrl)[:, :, 0].sum(1))
    assert np.allclose(e0, e) and np.allclose(v0, v)
    assert np.allclose(pvals[:, 0], pvals[:, 1])


def toy_adjacency(n):
    idx = np.arange(n)
    return (np.abs(idx[:, None] - idx[None, :]) <= 1).astype(int)


def toy_df():
    rows = [(1, "A", 3, 0, "a"), (2, "A", 2, 1, "b"), (3, "A", 0, 2, "a"), (4, "A", 1, 0, "b"),
            (5, "A", 0, 3, "b"), (6, "A", 1, 1, "a"), (6, "C", 0, 1, "a"), (7, "A", 0, 1, "a")]
    df = make_df(rows)
    df["uniprot_id"] = "P1"
    return df


def test_scan_observed_pvalue_matches_hand_computation(monkeypatch):
    import scan_test
    adj = toy_adjacency(8)
    monkeypatch.setattr(scan_test, "get_adjacency_matrix", lambda *a, **k: adj)
    df = toy_df()
    out, _ = scan_test.compute_all_pvals(df, None, None, None, "P1", 20, radius=15, seed=3, exclusive=True)
    case, ctrl = exclusive_case_control(df, 8, 20, seed=3)
    a, b = adj @ case[0, :, 0], adj @ ctrl[0, :, 0]
    nc, nn = case[0, :, 0].sum(), ctrl[0, :, 0].sum()
    want = np.where((a > 0) & (a * nn > b * nc), hypergeom.sf(a - 1, nc + nn, nc, a + b), 1.0)
    assert np.allclose(out["p_value"], want)
    assert (out["original_case"] == case[0, :, 0]).all()
    assert case[0, 1, 0] == 0  # shared variant at residue 2 is gone


def test_scan_cmh_and_annotation_run(monkeypatch):
    import scan_test_cmh
    from annotation_test_permutation import test_protein_annotations, test_protein_annotations_cmh
    adj = toy_adjacency(8)
    monkeypatch.setattr(scan_test_cmh, "get_adjacency_matrix", lambda *a, **k: adj)
    df = toy_df()
    out, _ = scan_test_cmh.compute_all_cmh_pvals(df, None, None, None, "P1", 20, "ancestry", seed=1, exclusive=True)
    assert out.shape[0] == 8 and out["p_value"].between(0, 1).all()
    labels = np.eye(8, dtype=int)[:3]
    meta = {i: ("a", str(i)) for i in range(3)}
    res = test_protein_annotations("P1", df, labels, meta, 8, 20, exclusive=True)
    assert res["p_value"].between(0, 1).all() and res["ac_in_case"].iloc[1] == 0
    res = test_protein_annotations_cmh("P1", df, labels, meta, 20, "ancestry", exclusive=True)
    assert res["p_value"].between(0, 1).all()


def test_cmh_pvals_exact_on_screened_cells():
    from cmh import exact_cmh_upper_lookup
    # 2 strata, 2 residues, neighbourhood = residue 0; one column
    case = np.array([[[4], [0]], [[3], [1]]], dtype=np.int32)
    ctrl = np.array([[[0], [4]], [[0], [4]]], dtype=np.int32)
    pvals, *_ = cmh_pvals_by_column(np.array([[1, 0], [0, 1]]), case, ctrl)
    support, upper = exact_cmh_upper_lookup([4, 3], [4, 4], [8, 8])
    assert np.isclose(pvals[0, 0], upper[7 - support[0]])


def random_margins(rng, rows, k):
    n = rng.integers(0, 40, (rows, k))
    n[:, 1] = 0  # empty stratum
    n[:, 2] = rng.integers(0, 3, rows)  # tiny strata, mostly point masses
    a = rng.integers(0, n + 1)
    m = rng.integers(0, n + 1)
    m[:10, 3], a[10:20, 3] = n[:10, 3], 0  # m == n and a == 0 edges
    return m, a, n


def test_batched_upper_pvals_match_per_row_lookup():
    from cmh import exact_cmh_upper_lookup, exact_cmh_upper_pvals
    rng = np.random.default_rng(0)
    for k in (5, 4):
        m, a, n = random_margins(rng, 300, k)
        if k == 4:
            n[:, 0] = m[:, 0] = a[:, 0] = 0
            m[:50, 3] = 0  # the widest stratum (handled through its tail) is a point mass here
        cases = [(i, s) for i in range(len(n)) for s in exact_cmh_upper_lookup(m[i], a[i], n[i])[0]]
        rows, x = np.array(cases).T
        got = exact_cmh_upper_pvals(m[rows], a[rows], n[rows], x, batch=97)
        for (i, s), p in zip(cases, got):
            support, want = exact_cmh_upper_lookup(m[i], a[i], n[i])
            assert np.isclose(p, want[s - support[0]], rtol=1e-10, atol=0)


def test_batched_upper_pvals_single_stratum_and_point_masses():
    from cmh import exact_cmh_upper_lookup, exact_cmh_upper_pvals
    m, a, n = np.array([[3], [0], [5]]), np.array([[4], [2], [5]]), np.array([[10], [6], [9]])
    got = exact_cmh_upper_pvals(m, a, n, [2, 0, 5])
    want = [exact_cmh_upper_lookup(m[i], a[i], n[i]) for i in range(3)]
    assert np.allclose(got, [u[x - s[0]] for (s, u), x in zip(want, [2, 0, 5])], rtol=1e-12)
    assert got[1] == 1.0  # point mass at the observed value


def test_batched_upper_pvals_independent_of_batch_and_order():
    # observed and null cells with the same margins must tie bitwise within one call,
    # whatever batch they land in or what they share it with
    from cmh import exact_cmh_upper_pvals
    rng = np.random.default_rng(1)
    m, a, n = random_margins(rng, 400, 5)
    lo = np.maximum(0, m - (n - a)).sum(1)
    x = lo + rng.integers(0, 4, len(n))
    ref = exact_cmh_upper_pvals(m, a, n, x)
    perm = rng.permutation(len(n))
    for batch in (1, 7, 1000):
        got = exact_cmh_upper_pvals(m[perm], a[perm], n[perm], x[perm], batch=batch)
        assert (got == ref[perm]).all()


def test_single_informative_stratum_ties_whatever_its_position():
    # the same law in stratum 0 or in the tail-handled widest stratum must give the same bits
    from cmh import exact_cmh_upper_pvals
    m = np.array([[5, 0], [0, 5], [3, 20]])
    a = np.array([[4, 0], [0, 4], [3, 20]])
    n = np.array([[10, 3], [3, 10], [10, 40]])
    for x in range(1, 5):
        p = exact_cmh_upper_pvals(m, a, n, [x, x, 5])
        assert p[0] == p[1]


def test_cmh_by_column_sparse_path_matches_dense():
    # a mostly empty aggregator stays csr; padding adds zero-variance rows that are never screened
    rng = np.random.default_rng(2)
    adj = toy_adjacency(8)
    case = rng.integers(0, 4, (2, 8, 30)).astype(np.int32)
    ctrl = rng.integers(0, 4, (2, 8, 30)).astype(np.int32)
    big = np.zeros((500, 500), dtype=int)
    big[:8, :8] = adj
    pad = np.zeros((2, 492, 30), dtype=np.int32)
    dense, *_ = cmh_pvals_by_column(adj, case, ctrl)
    sparse, *_ = cmh_pvals_by_column(big, np.concatenate([case, pad], 1), np.concatenate([ctrl, pad], 1))
    assert (sparse[:8] == dense).all() and (sparse[8:] == 1.0).all()
