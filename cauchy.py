"""
Cauchy combination test (CCT) over gene-level p-values from several runs.

WHAT GOES IN
  The `gene_p` column written by --gene-pvals: the within-gene FWER, which is uniform
  under the gene-level null. CCT's calibration assumes uniform inputs, so this is the
  only column that may be combined. The neighborhood-level `fwer` column must NOT be:
  it is a conservative tail probability already corrected against every gene in its
  run, so combining k of them pays that correction k times and saturates. This module
  refuses a file that offers `fwer` without `gene_p`.

WHY THE CAUCHY COMBINATION AND NOT FISHER
  The inputs are severely dependent. Variant filters are nested (blosum_le1 contains
  le0 contains le-1 ...), the AlphaMissense thresholds likewise, and every run scores
  the same samples with the same structures. Fisher or Stouffer would be badly
  anticonservative under that dependence; CCT's tail approximation holds under
  arbitrary dependence, which is the whole reason to use it here.

THE CLAMP
  tan((0.5 - p) * pi) diverges at p = 1, and gene_p attains exactly 1 whenever every
  simulation beat the observed. One such input drives the mean to -inf and collapses
  the gene to 1 regardless of what the other inputs say. Those values are not really
  1 -- they mean "no simulation was beaten at a resolution of 1/(n_sims+1)" -- so each
  is clamped to 1 - 1/(n_sims + 1), the next value its own run could have produced.
  n_sims is read per row, so inputs run at different simulation counts each get their
  own clamp. --cct-clamp overrides it with a fixed value.

THE ACROSS-GENE CORRECTION IS APPLIED ONCE
  After combining, never inside the inputs -- that is the point of combining gene_p
  rather than fwer. Because gene_p does not depend on which other genes were scanned,
  one set of input files can be corrected under several gene sets by re-running with a
  different --cct-gene-list; only `cct_bonf` and `cct_bh` change.
"""

import os

import numpy as np
import pandas as pd

from gene_pvalues import GENE_P_COLUMN
from logger_config import get_logger

logger = get_logger(__name__)

DEFAULT_SIG = 0.05
# Below this the tan form loses precision and the 1/(p*pi) tail is used instead.
_TAIL = 1e-15


def cct(p, weights=None):
    """
    Equal-weight CCT by default. p must lie in (0, 1]; gene_p always does, since it
    is (k + 1) / (n_sims + 1).
    """
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return np.nan
    if not np.all(np.isfinite(p)):
        raise ValueError('CCT input contains a non-finite p-value')
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError('CCT input must lie in (0, 1]; got '
                         f'[{p.min():.6g}, {p.max():.6g}]. A p-value of exactly 0 '
                         'sends the combination to 0 -- gene_p carries the +1 Monte '
                         'Carlo correction precisely to avoid that, so a 0 means an '
                         'uncorrected column (gene_p_raw?) was passed by mistake.')

    if weights is None:
        w = np.full(p.size, 1.0 / p.size)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != p.shape:
            raise ValueError('weights and p must have the same shape')
        if np.any(w < 0) or not w.sum() > 0:
            raise ValueError('weights must be non-negative and not all zero')
        w = w / w.sum()

    t = np.where(p < _TAIL, 1.0 / (p * np.pi), np.tan((0.5 - p) * np.pi))
    return float(0.5 - np.arctan(float(np.sum(w * t))) / np.pi)


def _strip_suffix(name):
    for suffix in ('.gene.tsv', '.tsv'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _split_label(spec):
    """`label=path` if given, else (None, spec)."""
    if '=' in spec:
        label, path = spec.split('=', 1)
        return label.strip(), path.strip()
    return None, spec.strip()


def _assign_labels(specs, results_dir=''):
    """
    (label, path) per spec. An unlabelled input is named by its basename, which is
    what anyone wants when the files are <filter>.gene.tsv -- but scans run with the
    default --fdr-file all share the basename `all_proteins`, and the directory is
    the only thing telling them apart. So the basenames are checked for collisions
    first, and the whole set falls back to the path when any two collide. Labelling
    must be decided across all the specs at once, not one at a time, or a glob --
    which cannot carry an explicit label=path -- has no way out of the collision.
    """
    split = [_split_label(spec) for spec in specs]
    basenames = [_strip_suffix(os.path.basename(path)) for _, path in split]
    collide = len(set(basenames)) < len(basenames)
    labels = []
    for (explicit, path), base in zip(split, basenames):
        labels.append(explicit if explicit is not None
                      else (_strip_suffix(path) if collide else base))
    paths = [os.path.join(results_dir, path) if results_dir else path
             for _, path in split]
    return list(zip(labels, paths))


def _read_gene_file(label, path):
    df = pd.read_csv(path, sep='\t')
    if GENE_P_COLUMN not in df.columns:
        if 'fwer' in df.columns:
            raise ValueError(
                f'{path} has no {GENE_P_COLUMN} column but does have `fwer`. The '
                'neighborhood-level FWER cannot be Cauchy-combined: it is already '
                'corrected against every gene in its own run, so combining several '
                'pays that correction once per input. Re-run the scan with '
                '--gene-pvals to get a combinable gene-level p-value.')
        raise ValueError(f'{path} has no {GENE_P_COLUMN} column; it is not a '
                         'gene-level p-value file (--gene-pvals)')
    if 'uniprot_id' not in df.columns:
        raise ValueError(f'{path} has no uniprot_id column')
    df = df.copy()
    df['input'] = label
    for optional, default in (('gene_name', None), ('best_aa_pos', np.nan),
                              ('best_radius', np.nan), ('n_sims', np.nan)):
        if optional not in df.columns:
            df[optional] = default
    return df


def combine_gene_pvalues(specs, results_dir='', gene_list=None, clamp=None,
                         sig=DEFAULT_SIG):
    """
    Args:
        specs: gene-level p-value files, optionally `label=path`.
        results_dir: prepended to each path when given.
        gene_list: uniprot_ids to keep. Applied BEFORE the across-gene correction,
            which is the only thing it changes.
        clamp: fixed upper clamp. None uses each row's own 1 - 1/(n_sims + 1).
        sig: threshold for the n_sig diagnostic.

    Returns:
        One row per gene, sorted by `cct`.
    """
    if len(specs) < 2:
        logger.warning(f'Combining {len(specs)} input file; CCT of a single input '
                       'returns that input unchanged')

    frames = []
    labels = []
    for label, path in _assign_labels(specs, results_dir):
        if label in labels:
            raise ValueError(f'Duplicate input label {label!r} for {path}; pass '
                             'label=path to name the inputs explicitly')
        df = _read_gene_file(label, path)
        labels.append(label)
        frames.append(df)
        logger.info(f'{label}: {len(df)} genes')
    long = pd.concat(frames, ignore_index=True)

    if gene_list is not None:
        gene_list = set(gene_list)
        before = long.uniprot_id.nunique()
        long = long[long.uniprot_id.isin(gene_list)]
        logger.info(f'Gene list keeps {long.uniprot_id.nunique()} of {before} genes')
        if long.empty:
            raise ValueError('No genes left after the gene list was applied')

    counts = long.groupby('uniprot_id').size()
    # min(), not nunique(): if a glob matched files with entirely disjoint gene sets
    # every count is 1, nunique() is 1, and the warning that matters most would be
    # the one that never fires.
    if counts.min() < len(specs):
        logger.warning(
            f'{int((counts < len(specs)).sum())} of {len(counts)} genes are missing '
            'from at least one input. Each is combined over the inputs it does appear '
            'in, recorded in n_inputs -- their CCTs are still valid p-values but rest '
            'on different numbers of tests.')

    rows = []
    for uniprot_id, g in long.groupby('uniprot_id', sort=False):
        p = g[GENE_P_COLUMN].to_numpy(dtype=float)
        if clamp is None:
            n_sims = g['n_sims'].to_numpy(dtype=float)
            # A file without n_sims cannot say what one simulation unit is; leave
            # those uncapped rather than invent a resolution.
            hi = np.where(np.isfinite(n_sims), 1.0 - 1.0 / (n_sims + 1.0), 1.0)
        else:
            hi = np.full(p.shape, float(clamp))
        best = int(np.argmin(p))
        # Not g['gene_name'].iloc[best]: an input built by hand or by an older
        # version may have no gene_name at all, and the gene should not go unnamed
        # just because that file happened to hold its minimum.
        named = g['gene_name'].dropna()
        name = named.iloc[0] if len(named) else None
        rows.append({
            'uniprot_id': uniprot_id,
            'gene_name': name,
            'cct': cct(np.minimum(p, hi)),
            'n_inputs': int(len(g)),
            'n_sig': int((p < sig).sum()),
            'min_p': float(p[best]),
            'best_input': g['input'].iloc[best],
            'best_nbhd': g['best_aa_pos'].iloc[best],
            'best_radius': g['best_radius'].iloc[best],
            'n_clamped': int((p > hi).sum()),
        })

    if not rows:
        raise ValueError(
            f'No genes to combine: the {len(specs)} input file(s) hold no gene rows. '
            'A scan writes a header-only .gene.tsv when no gene had a finite '
            'observed p-value.')

    out = pd.DataFrame(rows).sort_values('cct').reset_index(drop=True)

    # The across-gene correction, applied once, here and nowhere else.
    m = len(out)
    out['cct_bonf'] = np.minimum(out.cct * m, 1.0)
    order = np.arange(1, m + 1)
    out['cct_bh'] = np.minimum.accumulate(
        (out.cct.to_numpy() * m / order)[::-1])[::-1].clip(max=1.0)
    out['n_genes_tested'] = m

    logger.info(f'Combined {len(specs)} inputs over {m} genes; '
                f'{int((out.cct_bonf < sig).sum())} significant at '
                f'Bonferroni < {sig}, {int((out.cct_bh < sig).sum())} at BH < {sig}')
    return out
