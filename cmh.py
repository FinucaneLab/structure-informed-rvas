"""Exact Cochran-Mantel-Haenszel utilities for stratified 3DNT.

For each stratum k, conditioning on the 2x2 table margins gives

    A_k | M_k, C_k, N_k ~ Hypergeom(N_k, C_k, M_k),

where A_k is the case count inside the neighborhood, M_k is the total count
inside the neighborhood, C_k is the protein-wide case count, and N_k is the
protein-wide total count.  The CMH sufficient statistic is S = sum_k A_k, so
its exact conditional null law is the convolution of the stratum-specific
hypergeometric laws.

The two-sided p-value uses probability ordering ("minimum likelihood"), which
matches R's exact mantelhaen.test convention and reduces to the usual two-sided
Fisher exact test when there is only one stratum.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom


REL_ERR = 1.0 + 1e-7


def cmh_expected_variance(inside_totals, case_totals, stratum_totals):
    """Return E[S] and Var[S] for S=sum_k A_k under the CMH null."""
    m = np.asarray(inside_totals, dtype=np.float64)
    a = np.asarray(case_totals, dtype=np.float64)
    n = np.asarray(stratum_totals, dtype=np.float64)

    if not (m.shape == a.shape == n.shape):
        raise ValueError(
            "inside_totals, case_totals, and stratum_totals must have the same shape"
        )
    if np.any(m < 0) or np.any(a < 0) or np.any(n < 0):
        raise ValueError("CMH margins must be nonnegative")
    if np.any(m > n) or np.any(a > n):
        raise ValueError("Inside totals and case totals cannot exceed stratum totals")

    expected_k = np.zeros_like(n, dtype=np.float64)
    variance_k = np.zeros_like(n, dtype=np.float64)

    nonempty = n > 0
    expected_k[nonempty] = m[nonempty] * a[nonempty] / n[nonempty]

    informative = n > 1
    ni = n[informative]
    mi = m[informative]
    ai = a[informative]
    variance_k[informative] = (
        mi
        * (ni - mi)
        * ai
        * (ni - ai)
        / (ni * ni * (ni - 1.0))
    )

    return float(expected_k.sum()), float(variance_k.sum())


def cmh_chisq_stat(case_inside_sum, expected, variance):
    """CMH chi-square statistic without continuity correction."""
    x = np.asarray(case_inside_sum, dtype=np.float64)
    if variance <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return (x - expected) ** 2 / variance


def exact_cmh_distribution(inside_totals, case_totals, stratum_totals):
    """Return the exact null distribution of S=sum_k A_k.

    Returns
    -------
    support : ndarray[int]
        Integer support of S.
    pmf : ndarray[float]
        Exact conditional null probability at each support point.
    """
    m = np.asarray(inside_totals, dtype=np.int64)
    a = np.asarray(case_totals, dtype=np.int64)
    n = np.asarray(stratum_totals, dtype=np.int64)

    if m.ndim != 1 or not (m.shape == a.shape == n.shape):
        raise ValueError(
            "inside_totals, case_totals, and stratum_totals must be 1-D arrays with equal shape"
        )
    if np.any(m < 0) or np.any(a < 0) or np.any(n < 0):
        raise ValueError("CMH margins must be nonnegative")
    if np.any(m > n) or np.any(a > n):
        raise ValueError("Inside totals and case totals cannot exceed stratum totals")

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
    """Return support and two-sided minimum-likelihood exact p-values."""
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
    """Convenience wrapper for one two-sided exact CMH p-value."""
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
