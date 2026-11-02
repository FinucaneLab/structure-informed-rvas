from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom

from logger_config import get_logger
from utils import DEFAULT_SCREEN_CHI2, SCREEN_CHI2_GREATER

logger = get_logger(__name__)


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


def exact_cmh_upper_lookup(inside_totals, case_totals, stratum_totals):
    """Support and one-sided P(S >= s) (case enrichment inside)."""
    support, pmf = exact_cmh_distribution(inside_totals, case_totals, stratum_totals)
    return support, np.clip(np.cumsum(pmf[::-1])[::-1], 0.0, 1.0)


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


def screen_and_exact_pvalues(cmh, screen_chi2=DEFAULT_SCREEN_CHI2, alternative="greater"):
    greater = alternative == "greater"
    if greater and screen_chi2 == DEFAULT_SCREEN_CHI2:
        screen_chi2 = SCREEN_CHI2_GREATER
    s = cmh["s_case_inside"]
    e = cmh["expected_sum"]
    v = cmh["variance_sum"]

    stat = cmh_chisq_stat(s, e, v)
    informative = v > 0

    candidate = (stat > float(screen_chi2)) & informative[:, None]
    if greater:
        candidate &= s > e[:, None]
    lookup_fn = exact_cmh_upper_lookup if greater else exact_cmh_minlike_lookup
    pvals = np.ones_like(stat, dtype=np.float64)

    inside_by_k = cmh["inside_totals_by_stratum"]
    case_totals = cmh["case_totals"]
    totals = cmh["stratum_totals"]

    cache = {}
    rows = np.where(candidate.any(axis=1))[0]

    for r in rows:
        cols = np.where(candidate[r])[0]
        inside = inside_by_k[r, :]
        key = inside.tobytes()

        if key not in cache:
            support, p_lookup = lookup_fn(
                inside,
                case_totals,
                totals,
            )
            cache[key] = (int(support[0]), int(support[-1]), p_lookup)

        lo, hi, p_lookup = cache[key]
        sval = s[r, cols].astype(np.int64)
        if np.any(sval < lo) or np.any(sval > hi):
            raise RuntimeError(
                f"CMH sufficient statistic outside exact support at residue {r + 1}"
            )
        pvals[r, cols] = p_lookup[sval - lo]

    return pvals, stat


def mantel_haenszel_or(inside_case_by_stratum, inside_totals_by_stratum, case_totals, stratum_totals):
    """
    Mantel-Haenszel common odds ratio of "case vs control" for inside vs outside, per row.
    Per stratum k: a=case inside, b=ctrl inside, c=case outside, d=ctrl outside;
    OR = sum(a*d/n) / sum(b*c/n). inf when only the denominator is 0, nan when both are.
    """
    a = np.asarray(inside_case_by_stratum, dtype=np.float64)
    b = np.asarray(inside_totals_by_stratum, dtype=np.float64) - a
    case = np.asarray(case_totals, dtype=np.float64)
    n = np.asarray(stratum_totals, dtype=np.float64)
    keep = n > 0
    a, b, case, n = a[:, keep], b[:, keep], case[keep], n[keep]
    c = case - a
    d = (n - case) - b
    num = (a * d / n).sum(axis=1)
    den = (b * c / n).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def summarize_strata(rvas_df, stratum_col):
    """Per-stratum allele counts; warns once about strata with no cases or no controls."""
    if stratum_col not in rvas_df.columns:
        raise KeyError(f"Missing stratum column '{stratum_col}'")
    if rvas_df[stratum_col].isna().any():
        raise ValueError(f"Missing values found in stratum column '{stratum_col}'")
    summary = rvas_df.groupby(stratum_col).agg(
        n_variants=("ac_case", "size"),
        ac_case=("ac_case", "sum"),
        ac_control=("ac_control", "sum"),
    )
    summary["case_frac"] = summary["ac_case"] / (summary["ac_case"] + summary["ac_control"])
    empty = summary.index[(summary["ac_case"] == 0) | (summary["ac_control"] == 0)]
    if len(empty):
        logger.warning(
            f"{len(empty)}/{len(summary)} strata have no cases or no controls and carry "
            f"no CMH information: {list(empty)}"
        )
    return summary
