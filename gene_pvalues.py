"""
Gene-level p-values for the case/control scan: the within-gene FWER.

WHAT THIS IS, AND HOW IT DIFFERS FROM THE `fwer` COLUMN
  `empirical_fdr._compute_fwer` corrects each neighborhood against the minimum null
  p-value over the WHOLE set of proteins in the run. That is the right correction for
  "is this neighborhood significant", but it is not a p-value for the GENE, and it is
  not uniform under any null -- it is a conservative tail probability that depends on
  how many other genes happened to be scanned alongside.

  This module corrects a gene's best neighborhood against the minimum null p-value
  WITHIN THAT GENE ONLY:

      gene_p = (#{s : min_j null_p[j, s] <= min_j obs_p[j]} + 1) / (n_sims + 1)

  where j runs over the gene's tested centres and s over the simulations. That is
  uniform under the gene-level null -- it corrects for the within-gene search and
  nothing more -- so it can be combined across runs (see cauchy.py) and corrected
  across genes once, at the end. Combining `fwer` instead pays the across-gene
  correction once per input and saturates.

  It also does not depend on which OTHER genes were in the run, only on this gene's
  own positions. So one gene-level file can be reused under several gene sets; only
  the final across-gene correction changes.

THREE DETAILS THAT ARE NOT OPTIONAL
  The observed minimum and the null minima must be read from the SAME array. They are
  here, because both come from one `_load_all_pvalues` call. Recomputing the observed
  minimum from the written fdr.tsv instead costs the last binary digit
  (0.0350877192982456 vs 0.03508771929824561), which matters because of the next point.

  Ties must be counted (`<=`, not `<`). The exact CMH is discrete, badly so when the
  variant set is sparse: a gene can have three distinct observed p-values across
  hundreds of neighborhoods, and its null minimum four distinct values, so the null
  minimum frequently EQUALS the observed. Dropping those ties puts the statistic on
  its floor.

  The +1 on both sides is the standard Monte Carlo correction. Without it a gene can
  score exactly 0, which sends any downstream Cauchy combination to 0.
"""

import numpy as np
import pandas as pd

from logger_config import get_logger

logger = get_logger(__name__)

GENE_P_COLUMN = 'gene_p'

# Written in this order; gene_name is merged in by the caller, which is the only one
# that knows the reference directory.
GENE_COLUMNS = ['uniprot_id', GENE_P_COLUMN, 'n_pos', 'obs_min_p', 'best_aa_pos',
                'best_radius', 'n_null_le_obs', 'n_sims', 'gene_p_raw']


def gene_level_pvalues(df_pvals, null_pvals_dict, n_sims=None):
    """
    One row per protein: the within-gene FWER of its best neighborhood.

    Args:
        df_pvals: observed p-values, one row per tested neighborhood, with columns
            uniprot_id, aa_pos, p_value and (optionally) radius. Row order is
            irrelevant -- only the per-gene minimum is taken.
        null_pvals_dict: uniprot_id -> (n_positions, n_sims) null p-value array,
            restricted to the same positions as df_pvals.
        n_sims: expected simulation count, for a consistency check only. Each gene's
            own array width is what is actually used.

    Returns:
        DataFrame with GENE_COLUMNS, sorted by gene_p.
    """
    rows = []
    has_radius = 'radius' in df_pvals.columns
    for uniprot_id, g in df_pvals.groupby('uniprot_id', sort=True):
        null = null_pvals_dict.get(uniprot_id)
        if null is None or null.size == 0:
            continue

        obs = g['p_value'].to_numpy(dtype=float)
        finite = np.isfinite(obs)
        if not finite.any():
            logger.warning(f'{uniprot_id}: no finite observed p-value; no gene_p')
            continue
        i = int(np.nanargmin(obs))
        obs_min = float(obs[i])

        # Both sides must search the SAME centres. The observed minimum is taken over
        # the finite positions only, so the null minimum has to drop those rows too --
        # otherwise the null searched a larger neighborhood set than the observed did
        # and gene_p is biased upward, which breaks the uniformity the whole
        # combination rests on. Neither scan backend emits a NaN p-value today, so
        # this is a guard rather than a live correction.
        #
        # null rows are in ascending aa_pos order (read_p_values numbers them
        # positionally and the --df-filter mask is applied to both sides together),
        # while df_pvals has since been sorted by p_value -- so g has to be put back
        # in aa_pos order before its mask can index null.
        keep = finite
        if not finite.all():
            if null.shape[0] != len(g):
                logger.warning(
                    f'{uniprot_id}: {len(g)} observed rows but {null.shape[0]} null '
                    'rows; cannot align the two, so the null keeps every centre')
                keep = None
            else:
                by_pos = np.argsort(g['aa_pos'].to_numpy(), kind='stable')
                keep = finite[by_pos]
        nmin = np.nanmin(null if keep is None else null[keep, :], axis=0)
        n = int(nmin.size)
        if n_sims is not None and n != n_sims:
            logger.warning(f'{uniprot_id}: {n} simulations, expected {n_sims}')
        # A non-finite null minimum means that simulation produced no usable p-value
        # anywhere in the gene. Left uncounted, which is the conservative direction.
        n_bad = int((~np.isfinite(nmin)).sum())
        if n_bad:
            logger.warning(f'{uniprot_id}: {n_bad} of {n} simulations have no finite '
                           'null p-value; counted as not beating the observed')
        k = int(np.count_nonzero(nmin <= obs_min))

        rows.append({
            'uniprot_id': uniprot_id,
            GENE_P_COLUMN: (k + 1) / (n + 1),
            'n_pos': int(finite.sum()),
            'obs_min_p': obs_min,
            'best_aa_pos': int(g['aa_pos'].to_numpy()[i]),
            'best_radius': float(g['radius'].to_numpy()[i]) if has_radius else np.nan,
            'n_null_le_obs': k,
            'n_sims': n,
            'gene_p_raw': k / n,
        })

    if not rows:
        return pd.DataFrame(columns=GENE_COLUMNS)
    return pd.DataFrame(rows)[GENE_COLUMNS].sort_values(
        GENE_P_COLUMN).reset_index(drop=True)
