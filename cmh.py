from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom


REL_ERR = 1.0 + 1e-7


def cmh_expected_variance(inside_totals, case_totals, stratum_totals):
    """
    Expected value and variance, under the CMH null, of the sum over strata of
    the "inside & case" counts. case_totals/stratum_totals are the per-stratum
    margins (shape (k,)); inside_totals is reduced over its last axis against
    them, so it may carry a leading axis (e.g. one row per candidate
    neighborhood) that is broadcast across the k strata.
    """
    m = np.asarray(inside_totals, dtype=np.float64)
    a = np.asarray(case_totals, dtype=np.float64)
    n = np.asarray(stratum_totals, dtype=np.float64)

    if a.shape != n.shape or m.shape[-1:] != a.shape[-1:]:
        raise ValueError(
            "shape mismatch"
        )
    if np.any(m < 0) or np.any(a < 0) or np.any(n < 0):
        raise ValueError("negative margins")
    if np.any(m > n) or np.any(a > n):
        raise ValueError("total count mismatch")

    expected_k = np.zeros_like(m, dtype=np.float64)
    variance_k = np.zeros_like(m, dtype=np.float64)

    nonempty = n > 0
    expected_k[..., nonempty] = m[..., nonempty] * a[nonempty] / n[nonempty]

    informative = n > 1
    ni = n[informative]
    mi = m[..., informative]
    ai = a[informative]
    variance_k[..., informative] = (
        mi
        * (ni - mi)
        * ai
        * (ni - ai)
        / (ni * ni * (ni - 1.0))
    )

    expected = expected_k.sum(axis=-1)
    variance = variance_k.sum(axis=-1)
    if expected.ndim == 0:
        return float(expected), float(variance)
    return expected, variance


def cmh_chisq_stat(case_inside_sum, expected, variance):
    """
    (x - expected)^2 / variance, zeroed out wherever variance is non-positive
    (the degenerate case where the null outcome is deterministic). expected
    and variance may carry fewer trailing axes than case_inside_sum (e.g. one
    value per residue against null draws per residue per simulation); they are
    broadcast against its leading axes.
    """
    x = np.asarray(case_inside_sum, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    v = np.asarray(variance, dtype=np.float64)

    extra_dims = x.ndim - e.ndim
    if extra_dims < 0 or e.shape != v.shape:
        raise ValueError("shape mismatch")
    e = e.reshape(e.shape + (1,) * extra_dims)
    v = v.reshape(v.shape + (1,) * extra_dims)

    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(v > 0, (x - e) ** 2 / np.where(v > 0, v, 1.0), 0.0)
    return stat


def exact_cmh_distribution(inside_totals, case_totals, stratum_totals):
    m = np.asarray(inside_totals, dtype=np.int64)
    a = np.asarray(case_totals, dtype=np.int64)
    n = np.asarray(stratum_totals, dtype=np.int64)

    if m.ndim != 1 or not (m.shape == a.shape == n.shape):
        raise ValueError(
            "shape mismatch"
        )
    if np.any(m < 0) or np.any(a < 0) or np.any(n < 0):
        raise ValueError("negative margins")
    if np.any(m > n) or np.any(a > n):
        raise ValueError("total count mismatch")

    pmf = np.array([1.0], dtype=np.float64)
    support_lo = 0

    for mk, ak, nk in zip(m, a, n):
        if nk == 0:
            continue

        lo = int(max(0, mk - (nk - ak)))
        hi = int(min(mk, ak))

        if lo == hi:
            support_lo += lo
            continue

        values = np.arange(lo, hi + 1, dtype=np.int64)
        pk = hypergeom.pmf(values, nk, ak, mk).astype(np.float64)
        total = pk.sum()
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError(
                f"Could not construct hypergeometric law for N={nk}, case={ak}, inside={mk}"
            )
        pk /= total

        pmf = np.convolve(pmf, pk)
        pmf[pmf < 0] = 0.0
        pmf /= pmf.sum()
        support_lo += lo

    support = np.arange(support_lo, support_lo + len(pmf), dtype=np.int64)
    return support, pmf


def exact_cmh_minlike_lookup(inside_totals, case_totals, stratum_totals):
    support, pmf = exact_cmh_distribution(
        inside_totals=inside_totals,
        case_totals=case_totals,
        stratum_totals=stratum_totals,
    )

    sorted_p = np.sort(pmf)
    cumulative = np.cumsum(sorted_p)
    idx = np.searchsorted(sorted_p, pmf * REL_ERR, side="right") - 1
    pvals = np.where(idx >= 0, cumulative[np.maximum(idx, 0)], 0.0)
    pvals = np.clip(pvals, 0.0, 1.0)
    return support, pvals


def exact_cmh_pvalue(
    observed_case_inside,
    inside_totals,
    case_totals,
    stratum_totals,
):
    support, pvals = exact_cmh_minlike_lookup(
        inside_totals,
        case_totals,
        stratum_totals,
    )
    observed = int(observed_case_inside)
    if observed < support[0] or observed > support[-1]:
        raise ValueError(
            f"Observed S={observed} is outside the conditional support "
            f"[{support[0]}, {support[-1]}]"
        )
    return float(pvals[observed - support[0]])
