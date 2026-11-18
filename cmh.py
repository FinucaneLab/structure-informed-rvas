from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import gammaln
from scipy.stats import hypergeom, norm

from logger_config import get_logger
from utils import DEFAULT_SCREEN_CHI2, SCREEN_CHI2_GREATER

logger = get_logger(__name__)


REL_ERR = 1.0 + 1e-7
EXACT_BATCH = 5000  # cells per batch; bounds the (cells, support) float64 tables


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


def cmh_pvals_by_column(aggregator, case, ctrl, screen_chi2=SCREEN_CHI2_GREATER):
    """
    One-sided CMH p-values when the margins differ per column (the exclusive-variants
    test). aggregator is the (n_res, n_res) adjacency or label matrix; case, ctrl are
    (k, n_res, P) from exclusive_case_control.

    Same screen-then-exact scheme as screen_and_exact_pvalues: cells with a normal-approx
    z^2 > screen_chi2 and z > 0 get the exact conditional p-value computed from that
    column's own margins (deduped over identical margins); the other cells keep the
    normal-approx p-value, 1.0 where the variance is 0. Observed and null columns go
    through the same function, so empirical FDR/FWER stay calibrated.

    Returns (pvals, stat = z^2, expected_obs, variance_obs); the last two are column 0.
    """
    agg = csr_matrix(aggregator)
    if agg.nnz > 0.02 * agg.shape[0] * agg.shape[1]:
        # dense BLAS was ~4x faster than csr @ dense on ~8%-filled label matrices; integer
        # sums are exact either way. shortcut: 2% cut-off untuned, re-time if inputs change
        agg = agg.toarray().astype(np.float64)
    n_rows, n_cols = agg.shape[0], case.shape[2]
    x = np.zeros((n_rows, n_cols))
    expected = np.zeros((n_rows, n_cols))
    variance = np.zeros((n_rows, n_cols))
    for case_s, ctrl_s in zip(case, ctrl):
        a = case_s.sum(0).astype(np.float64)
        n = a + ctrl_s.sum(0)
        m = agg @ (case_s + ctrl_s).astype(np.float64)
        x += agg @ case_s.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected += np.where(n > 0, m * a / n, 0.0)
            variance += np.where(n > 1, m * (n - m) * a * (n - a) / (n * n * (n - 1.0)), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(variance > 0, (x - expected) / np.sqrt(variance), 0.0)
    pvals = np.where(variance > 0, norm.sf(z), 1.0)

    rows, cols = np.nonzero((variance > 0) & (z > 0) & (z ** 2 > screen_chi2))
    if len(rows):
        screened_rows, rows_in = np.unique(rows, return_inverse=True)
        agg_screened = agg[screened_rows]
        inside, case_totals, totals = [], [], []
        for case_s, ctrl_s in zip(case, ctrl):
            a = case_s.sum(0)[cols]
            inside.append((agg_screened @ (case_s + ctrl_s).astype(np.float64))[rows_in, cols])
            case_totals.append(a)
            totals.append(a + ctrl_s.sum(0)[cols])
        m, a, n = (np.rint(np.stack(v, axis=1)).astype(np.int64) for v in (inside, case_totals, totals))
        pvals[rows, cols] = exact_cmh_upper_pvals(m, a, n, np.rint(x[rows, cols]))
    return pvals, z ** 2, expected[:, 0], variance[:, 0]


def exact_cmh_upper_pvals(inside_totals, case_totals, stratum_totals, observed, batch=EXACT_BATCH):
    """
    One-sided exact CMH p-values P(S >= observed) for many margin rows at once; margins are
    (R, k), observed (R,). Matches exact_cmh_upper_lookup to ~1e-12 relative: the pmf comes
    from a log-factorial table, and the widest stratum enters through its upper tail
    instead of a convolution. That stratum is chosen per call, so cells whose p-values are
    compared (observed and null columns) must go through the same call.
    """
    # shortcut: log-factorial cancellation error grows ~ n ln n eps (1e-12 at n ~ 1e3, ~1e-9 near
    # n ~ 1e5); switch to a mode-anchored ratio recurrence if strata get that large
    m, a, n = (np.asarray(v, dtype=np.int64) for v in (inside_totals, case_totals, stratum_totals))
    observed = np.asarray(observed, dtype=np.int64)
    if np.any(m < 0) or np.any(a < 0) or np.any(n < 0):
        raise ValueError("negative margins")
    if np.any(m > n) or np.any(a > n):
        raise ValueError("total count mismatch")
    lo = np.maximum(0, m - (n - a))
    width = np.minimum(m, a) - lo + 1
    log_fact = gammaln(np.arange(n.max() + 1) + 1.0)
    laws = []
    for mk, ak, nk in zip(m.T, a.T, n.T):
        # the law is symmetric in (m, a); keying on (min, max) keeps mathematically tied
        # p-values tied bitwise, as the old symmetric scipy pmf did
        base = int(nk.max()) + 1
        lo_ma, hi_ma = np.minimum(mk, ak), np.maximum(mk, ak)
        codes, inv = np.unique((lo_ma * base + hi_ma) * base + nk, return_inverse=True)
        um, ua, un = codes // (base * base), codes // base % base, codes % base
        ulo = np.maximum(0, um - (un - ua))
        uw = np.minimum(um, ua) - ulo + 1
        table = np.zeros((len(codes), int(uw.max())))
        table[:, 0] = 1.0  # point mass; also covers n == 0
        live = np.flatnonzero(uw > 1)
        row = np.repeat(live, uw[live])
        col = np.arange(len(row)) - np.repeat(np.cumsum(uw[live]) - uw[live], uw[live])
        k, rm, ra, rn = ulo[row] + col, um[row], ua[row], un[row]
        table[row, col] = np.exp(
            log_fact[ra] - log_fact[k] - log_fact[ra - k]
            + log_fact[rn - ra] - log_fact[rm - k] - log_fact[rn - ra - rm + k]
            - log_fact[rn] + log_fact[rm] + log_fact[rn - rm]
        )
        total = np.cumsum(table, axis=1)[:, -1:]
        if not np.all(np.isfinite(total) & (total > 0)):
            raise FloatingPointError("Could not construct hypergeometric law")
        laws.append((inv.ravel(), table / total, uw == 1))

    # widest stratum last: its convolution is replaced by a dot with its upper tail
    last = int(np.argmax(width.sum(axis=0)))
    rest = [s for s in range(m.shape[1]) if s != last]
    inv_last, pk_last, _ = laws[last]
    # renormalised and summed from the top as a convolution with pmf = [1] would be, so a
    # cell whose only informative stratum is `last` ties with one where it is another stratum
    pk_last = pk_last / np.cumsum(pk_last, axis=1)[:, -1:]
    tail_last = np.zeros((len(pk_last), pk_last.shape[1] + 1))
    tail_last[:, :-1] = np.cumsum(pk_last[:, ::-1], axis=1)[:, ::-1]

    # sequential sums (cumsum) ignore trailing zero padding, so a cell's p-value never
    # depends on its batch; identical margins in observed and null columns tie exactly
    pvals = np.empty(len(m))
    order = np.argsort(width[:, rest].sum(axis=1), kind="stable")
    for start in range(0, len(m), batch):
        cells = order[start:start + batch]
        pmf = np.ones((len(cells), 1))
        for s in rest:
            w = int(width[cells, s].max())
            if w == 1:
                continue  # all point masses
            inv, pk, point = laws[s]
            pk, point = pk[inv[cells], :w], point[inv[cells], None]
            out = np.zeros((len(cells), pmf.shape[1] + w - 1))
            for j in range(w):
                out[:, j:j + pmf.shape[1]] += pmf * pk[:, j:j + 1]
            # point masses stay a bitwise no-op, as when a whole batch skips the stratum
            pmf = out / np.where(point, 1.0, np.cumsum(out, axis=1)[:, -1:])
        # P(S >= x) = sum_s P(rest = s) P(last >= x - s); index past the last support -> 0
        need = observed[cells, None] - lo[cells].sum(axis=1)[:, None] - np.arange(pmf.shape[1])
        tail = tail_last[inv_last[cells, None], np.clip(need, 0, tail_last.shape[1] - 1)]
        pvals[cells] = np.cumsum((pmf * np.where(need < 0, 1.0, tail))[:, ::-1], axis=1)[:, -1]
    return np.clip(pvals, 0.0, 1.0)


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
