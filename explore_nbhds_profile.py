"""
Per-residue betahat/case-control profile across an entire protein, for the
single best (lowest-FDR) trait per protein, compared against a ClinVar +
common-variants case-control profile.

Workflow:
1. Read an FDR results file (e.g. produced by --run-hmp), with columns
   uniprot_id, cluster, aa_pos, fdr, ... Pick, per protein, the row with the
   lowest fdr; map its cluster back to a trait id via --trait-cluster-file.
2. For that trait, map its raw variant data (--rvas-data-to-map, as a
   template with a {trait} placeholder) to the protein, and compute
   mean_betahat / se_betahat / n_betahat for every residue's neighborhood, at
   every radius in --radii.
3. Do the analogous computation for a ClinVar + common-variants file (chr,
   pos, ref, alt, ac_case, ac_control), using a case/control ratio in place
   of mean_betahat (no standard error equivalent; an approximate delta-method
   SE of the ratio is included as an extra column, but the primary
   se_betahat column is NA for this file, as requested). The trait profile
   also includes mean_betahat_shrunk, a shrinkage-toward-0 estimate
   analogous to the ClinVar ratio's pseudocount regularization.
4. Both tsvs also carry significance marker columns (is_significant_discovery,
   is_significant_other_trait on the trait tsv; is_significant_clinvar on the
   ClinVar tsv), computed from --fdr-results-file (all traits, not just the
   discovery one) and --clinvar-fdr-file at --fdr-cutoff, so the plotting
   step can mark significant neighborhoods without needing the fdr files
   again.
5. Write two tsvs (trait profile and ClinVar profile). Plotting is a
   separate, manual step: run this file directly (see _main / plot_profiles_from_files)
   to generate the comparison PDF from the two tsvs, optionally restricting
   to a subset of neighborhoods and choosing which column to plot. Each plot
   marks significant neighborhoods: a dot on the blue (trait) line for the
   discovery neighborhood or any neighborhood significant for the discovery
   trait; a diamond on the blue line for neighborhoods significant in some
   other trait; a dot on the red (ClinVar) line for ClinVar-significant
   neighborhoods.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd

from utils import get_distance_matrix_structure, get_pae_matrix_structure
from q_scan_test import get_betahats_per_residue, get_nbhd_stats, MULTI_RADII_SMALL, MULTI_RADII_BIG
from read_data import map_to_protein_q, map_to_protein
from logger_config import get_logger

logger = get_logger(__name__)


def parse_radii(radii_arg):
    """Parse --radii: 'multiple-small', 'multiple-big', or a comma-separated list of numbers."""
    if radii_arg == 'multiple-small':
        return list(MULTI_RADII_SMALL)
    if radii_arg == 'multiple-big':
        return list(MULTI_RADII_BIG)
    try:
        return [float(x) for x in radii_arg.split(',')]
    except ValueError:
        raise ValueError(
            f"--radii must be 'multiple-small', 'multiple-big', or a comma-separated "
            f"list of numbers; got '{radii_arg}'"
        )


def select_best_trait_per_protein(fdr_results_file, trait_cluster_file, fdr_cutoff=None):
    """
    Group the FDR results by uniprot_id, take the row with the lowest fdr per
    protein, and map its cluster to a trait id via trait_cluster_file.

    If fdr_cutoff is not None, proteins whose best fdr exceeds it are dropped.
    Default (None) is no cutoff: every protein's best row is kept regardless
    of significance.

    Returns a DataFrame with columns: uniprot_id, trait, cluster, fdr,
    aa_pos (the discovery position, kept for reference only), and gene_name
    if present in the input file.
    """
    df_fdr = pd.read_csv(fdr_results_file, sep='\t', dtype={'cluster': str})
    if 'cluster' not in df_fdr.columns:
        raise ValueError(
            f"{fdr_results_file} has no 'cluster' column; expected --run-hmp output "
            f"(uniprot_id, cluster, aa_pos, fdr, ...)."
        )

    # A protein can be attributed to multiple gene names, which duplicates
    # otherwise-identical (uniprot_id, cluster, aa_pos, fdr) rows; collapse
    # those before picking the best row per protein.
    n_raw = len(df_fdr)
    df_fdr = df_fdr.drop_duplicates(subset=['uniprot_id', 'cluster', 'aa_pos', 'fdr'])
    if len(df_fdr) < n_raw:
        logger.info(f"Dropped {n_raw - len(df_fdr)} duplicate row(s) "
                    f"(same uniprot_id/cluster/aa_pos/fdr, different gene_name)")

    best_idx = df_fdr.groupby('uniprot_id')['fdr'].idxmin()
    df_best = df_fdr.loc[best_idx].copy()

    if fdr_cutoff is not None:
        n_before = len(df_best)
        df_best = df_best[df_best['fdr'] <= fdr_cutoff]
        logger.info(
            f"Applied fdr_cutoff={fdr_cutoff}: kept {len(df_best)} of {n_before} proteins"
        )

    df_map = pd.read_csv(trait_cluster_file, sep='\t', dtype={'trait': str, 'cluster': str})
    dupes = df_map['cluster'].value_counts()
    dupes = dupes[dupes > 1]
    if len(dupes) > 0:
        logger.warning(
            f"{len(dupes)} clusters in {trait_cluster_file} map to multiple traits; "
            f"using the first trait listed for each."
        )
    df_map = df_map.drop_duplicates(subset='cluster', keep='first')

    df_best = df_best.merge(df_map, on='cluster', how='left')
    missing = df_best['trait'].isna()
    if missing.any():
        logger.warning(
            f"{missing.sum()} protein(s) have a best cluster not found in "
            f"{trait_cluster_file}; they will be skipped: "
            f"{df_best.loc[missing, 'uniprot_id'].tolist()}"
        )
        df_best = df_best[~missing]

    keep_cols = ['uniprot_id', 'trait', 'cluster', 'fdr', 'aa_pos']
    if 'gene_name' in df_best.columns:
        keep_cols.append('gene_name')
    return df_best[keep_cols].rename(columns={'aa_pos': 'discovery_aa_pos', 'fdr': 'discovery_fdr'})


def load_significant_neighborhoods(fdr_results_file, fdr_cutoff, df_best):
    """
    For plot-marker purposes: find every (protein, aa_pos) significant
    (fdr <= fdr_cutoff) in fdr_results_file, for ANY cluster/trait -- not
    just the single best trait per protein kept by
    select_best_trait_per_protein. Split into two sets per protein:
      - discovery-significant: cluster matches the protein's discovery
        cluster (df_best, from select_best_trait_per_protein)
      - other-significant: cluster differs from the discovery cluster
        (and not already counted as discovery-significant)

    Returns {uniprot_id: (set_of_discovery_sig_aa_pos, set_of_other_sig_aa_pos)}.
    """
    df_fdr = pd.read_csv(fdr_results_file, sep='\t', dtype={'cluster': str})
    df_fdr = df_fdr.drop_duplicates(subset=['uniprot_id', 'cluster', 'aa_pos', 'fdr'])
    df_sig = df_fdr[df_fdr['fdr'] <= fdr_cutoff]

    disc_cluster_map = df_best.set_index('uniprot_id')['cluster'].to_dict()

    out = {}
    for uid, grp in df_sig.groupby('uniprot_id'):
        disc_cluster = disc_cluster_map.get(uid)
        disc_aa = set(grp.loc[grp['cluster'] == disc_cluster, 'aa_pos'].astype(int)) if disc_cluster is not None else set()
        other_aa = set(grp.loc[grp['cluster'] != disc_cluster, 'aa_pos'].astype(int)) - disc_aa
        out[uid] = (disc_aa, other_aa)
    return out


def load_significant_clinvar_neighborhoods(clinvar_fdr_file, fdr_cutoff):
    """
    For plot-marker purposes: find every (protein, aa_pos) significant
    (fdr <= fdr_cutoff) in a ClinVar-significance fdr file (columns
    uniprot_id, aa_pos, fdr, ..., e.g. produced by a q3dnt/scan-test FDR run
    on the ClinVar + common-variants data). Returns {uniprot_id: set(aa_pos)}.
    """
    df = pd.read_csv(clinvar_fdr_file, sep='\t')
    df_sig = df[df['fdr'] <= fdr_cutoff]
    return {uid: set(grp['aa_pos'].astype(int)) for uid, grp in df_sig.groupby('uniprot_id')}


def write_required_phenocodes(fdr_results_file, trait_cluster_file, output_file, fdr_cutoff=None):
    """
    Dry-run helper for --explore-nhbds-traits-dry: resolve the best trait per
    protein (same selection logic as the real run) and write the distinct
    list of required trait/phenocode ids to output_file, one per line, so the
    corresponding --rvas-data-to-map files can be prepared on demand before
    running --explore-nhbds-traits for real.
    """
    df_best = select_best_trait_per_protein(fdr_results_file, trait_cluster_file, fdr_cutoff)
    traits = sorted(df_best['trait'].unique(), key=str)
    with open(output_file, 'w') as fh:
        fh.write('\n'.join(traits) + '\n')
    logger.info(f"Wrote {len(traits)} required phenocode(s) to {output_file}")
    return traits


def _compute_adjacency_by_radius(pdb_file_pos_guide, pdb_dir, pae_dir, protein, radii, pae_cutoff):
    """Compute distance/PAE matrices once per protein and threshold at each radius."""
    distance_matrix = get_distance_matrix_structure(pdb_file_pos_guide, pdb_dir, protein)
    if distance_matrix is None:
        return None

    pae_thresh = None
    if pae_cutoff != 0:
        pae_matrix = get_pae_matrix_structure(pdb_file_pos_guide, pae_dir, protein)
        if pae_matrix is not None:
            pae_thresh = (pae_matrix < pae_cutoff) * 1

    adj_by_radius = {}
    for radius in radii:
        dist_thresh = (distance_matrix < radius) * 1
        adj_by_radius[radius] = dist_thresh if pae_thresh is None else (dist_thresh & pae_thresh)
    return adj_by_radius


def compute_betahat_profile(df_mapped_trait, protein, adj_by_radius, shrink_k=5.0):
    """
    Per-residue n_betahat/mean_betahat/se_betahat for every radius, for one
    protein. Also includes mean_betahat_shrunk, a shrinkage estimate that
    pulls mean_betahat toward 0 in proportion to how few variants (n_betahat)
    support it: mean_betahat_shrunk = n_betahat * mean_betahat / (n_betahat + shrink_k).
    This is the betahat analogue of the pseudocount regularization already
    applied to the ClinVar case/control ratio (see compute_clinvar_profile):
    shrink_k plays the same role as that pseudocount, and its effect vanishes
    as n_betahat grows.
    """
    df_p = df_mapped_trait[df_mapped_trait.uniprot_id == protein]
    if df_p.empty or len(df_p) == 0:
        return []

    n_res = next(iter(adj_by_radius.values())).shape[0]
    base, base_sq, nulls, nulls_sq, n_per_residue = get_betahats_per_residue(df_p, n_res, n_sims=0)
    sum_mat = np.column_stack([base, nulls])
    sumsq_mat = np.column_stack([base_sq, nulls_sq])

    rows = []
    for radius, adj in adj_by_radius.items():
        n_nbhd, sum_nbhd, sumsq_nbhd = get_nbhd_stats(adj, n_per_residue, sum_mat, sumsq_mat)
        n_in = np.asarray(n_nbhd).flatten()
        s = sum_nbhd[:, 0]
        ssq = sumsq_nbhd[:, 0]

        with np.errstate(divide='ignore', invalid='ignore'):
            mean = np.where(n_in > 0, s / np.maximum(n_in, 1), np.nan)
            var = np.where(n_in > 1, (ssq - n_in * mean ** 2) / np.maximum(n_in - 1, 1), np.nan)
            se = np.where(n_in > 1, np.sqrt(np.maximum(var, 0) / n_in), np.nan)
            mean_shrunk = n_in * np.nan_to_num(mean, nan=0.0) / (n_in + shrink_k)

        for i in range(n_res):
            rows.append({
                'aa_pos': i + 1,
                'n_betahat': float(n_in[i]),
                'mean_betahat': float(mean[i]),
                'mean_betahat_shrunk': float(mean_shrunk[i]),
                'se_betahat': float(se[i]),
                'radius': radius,
            })
    return rows


def compute_clinvar_profile(df_mapped_clinvar, protein, adj_by_radius):
    """Per-residue n_variants/case-control ratio for every radius, for one protein."""
    df_p = df_mapped_clinvar[df_mapped_clinvar.uniprot_id == protein]
    if df_p.empty:
        return []

    n_res = next(iter(adj_by_radius.values())).shape[0]
    _, _, _, _, n_per_residue = get_betahats_per_residue(df_p, n_res, n_sims=0, colname='ac_case')
    case_per_residue, _, _, _, _ = get_betahats_per_residue(df_p, n_res, n_sims=0, colname='ac_case')
    control_per_residue, _, _, _, _ = get_betahats_per_residue(df_p, n_res, n_sims=0, colname='ac_control')

    n_case_total = df_p['ac_case'].sum()
    n_control_total = df_p['ac_control'].sum()
    if n_case_total == 0 or n_control_total == 0:
        logger.warning(f"{protein}: zero total case or control alleles; skipping ClinVar profile.")
        return []

    rows = []
    for radius, adj in adj_by_radius.items():
        n_nbhd = np.asarray(adj @ n_per_residue).flatten()
        case_nbhd = np.asarray(adj @ case_per_residue.flatten()).flatten()
        control_nbhd = np.asarray(adj @ control_per_residue.flatten()).flatten()

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (case_nbhd + 2) / (control_nbhd + 2 * n_control_total / n_case_total)
            se_ratio_approx = ratio * np.sqrt(
                1 / np.maximum(case_nbhd, 1e-9) + 1 / np.maximum(control_nbhd, 1e-9)
            )

        for i in range(n_res):
            rows.append({
                'aa_pos': i + 1,
                'n_betahat': float(n_nbhd[i]),
                'mean_betahat': float(ratio[i]),
                'se_betahat': np.nan,
                'se_ratio_approx': float(se_ratio_approx[i]),
                'nbhd_case': float(case_nbhd[i]),
                'nbhd_control': float(control_nbhd[i]),
                'radius': radius,
            })
    return rows


def _zscore(x):
    x = np.asarray(x, dtype=float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def _minmax(x):
    """Scale to [-1, 1] via (x - min) / (max - min), rescaled."""
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(hi - lo) or hi == lo:
        return np.zeros_like(x)
    return 2 * (x - lo) / (hi - lo) - 1


def _robust_scale(x):
    """Median/MAD scaling: (x - median) / (1.4826 * MAD), robust to outliers."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


NORM_METHODS = {'zscore': _zscore, 'minmax': _minmax, 'robust': _robust_scale}


def _normalize(x, method='zscore'):
    if method not in NORM_METHODS:
        raise ValueError(f"Unknown --norm method '{method}'; choices: {list(NORM_METHODS)}")
    return NORM_METHODS[method](x)


def _import_matplotlib():
    """Import matplotlib lazily, installing it on the fly if missing."""
    try:
        import matplotlib
    except ImportError:
        logger.warning("matplotlib not found; installing it now with pip...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'matplotlib'])
        import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    return plt, PdfPages


def _import_requests():
    """Import requests lazily, installing it on the fly if missing."""
    try:
        import requests
    except ImportError:
        logger.warning("requests not found; installing it now with pip...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'requests'])
        import requests
    return requests


# UniProt feature types that are structural/boilerplate rather than
# functionally informative, skipped from the annotation track (see
# plot_add_annotations.py, which this is adapted from).
ANNOTATION_SKIP_TYPES = {"Chain", "Helix", "Beta strand", "Turn", "Initiator methionine"}


def fetch_uniprot_features(acc, skip_types=ANNOTATION_SKIP_TYPES):
    """
    Fetch UniProt feature annotations (domains, sites, motifs, etc.) for one
    accession via the UniProt REST API. Returns (feats, seq_len), where feats
    is a list of (feature_type, description, start, end), 1-based inclusive.
    Returns (None, None) on any network/parsing failure, so callers can skip
    annotation for that protein without failing the whole plot.
    """
    requests = _import_requests()
    try:
        resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json", timeout=30)
        resp.raise_for_status()
        j = resp.json()
        feats = [
            (f["type"], f.get("description", ""), f["location"]["start"]["value"], f["location"]["end"]["value"])
            for f in j["features"]
            if f["type"] not in skip_types
            and None not in (f["location"]["start"]["value"], f["location"]["end"]["value"])
        ]
        return feats, j["sequence"]["length"]
    except Exception as e:
        logger.warning(f"{acc}: failed to fetch UniProt annotations ({e}); skipping annotation track.")
        return None, None


def _draw_annotation_track(axt, feats, seq_len):
    """Draw one horizontal bar row per feature type, as in plot_add_annotations.py."""
    import matplotlib.pyplot as plt
    types = list(dict.fromkeys(f[0] for f in feats))
    cmap = plt.get_cmap("tab20")
    for t, d, s, e in feats:
        r = types.index(t)
        axt.barh(r, e - s + 1, left=s - 0.5, height=0.7, color=cmap(r % 20), alpha=0.85)
        if seq_len and (e - s + 1) / seq_len > 0.03 and d:
            axt.text((s + e) / 2, r, d[:20], ha="center", va="center", fontsize=6)
    axt.set_yticks(range(len(types)))
    axt.set_yticklabels(types, fontsize=8)
    axt.invert_yaxis()
    if seq_len:
        axt.set_xlim(0.5, seq_len + 0.5)
    axt.set_xlabel('aa_pos')
    axt.spines[['top', 'right']].set_visible(False)


def _apply_nbhd_subset(df, nbhd_subset, value_col):
    """
    If nbhd_subset is given (a set of (uniprot_id, aa_pos, radius) tuples),
    blank out (set to NaN) value_col/se_betahat for rows not in the
    subset, while keeping every row (and the full aa_pos range) in the
    frame, so the plotted x-axis still spans the whole protein with gaps
    where data was excluded.
    """
    if not nbhd_subset:
        return df
    df = df.copy()
    keys = list(zip(df['uniprot_id'], df['aa_pos'].astype(int), df['radius'].astype(float)))
    mask = np.array([k not in nbhd_subset for k in keys])
    for col in (value_col, 'se_betahat'):
        if col in df.columns:
            df.loc[mask, col] = np.nan
    return df


def _spearman_with_circular_perm(x, y, n_perm=1000, min_n=5, seed=0):
    """
    Spearman rank correlation between x and y (same length, aligned by
    position), using pairwise-complete observations (positions where either
    is NaN -- e.g. outside a --nbhd-subset, or naturally low-coverage -- are
    dropped from both before computing rho).

    Because adjacent neighborhoods overlap heavily, positions aren't
    independent, so the standard parametric p-value is too optimistic. If
    n_perm > 0, also returns a permutation p-value from circularly shifting
    the full y series (preserving its own autocorrelation) against x many
    times and comparing the real |rho| to the null distribution.

    Returns (rho, n_pairs, perm_pvalue). rho and perm_pvalue are NaN if
    fewer than min_n complete pairs are available.
    """
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    n_pairs = int(valid.sum())
    if n_pairs < min_n:
        return np.nan, n_pairs, np.nan

    rho, _ = spearmanr(x[valid], y[valid])

    perm_pvalue = np.nan
    if n_perm > 0 and len(x) > 1:
        rng = np.random.default_rng(seed)
        n = len(x)
        count = 0
        for shift in rng.integers(1, n, size=n_perm):
            y_shift = np.roll(y, shift)
            valid_shift = ~(np.isnan(x) | np.isnan(y_shift))
            if valid_shift.sum() < min_n:
                continue
            rho_shift, _ = spearmanr(x[valid_shift], y_shift[valid_shift])
            if np.isfinite(rho_shift) and abs(rho_shift) >= abs(rho):
                count += 1
        perm_pvalue = (count + 1) / (n_perm + 1)

    return rho, n_pairs, perm_pvalue


def _plot_profiles(df_trait, df_clinvar, pdf_path, nbhd_subset=None, value_col='mean_betahat',
                    n_perm=1000, annotate=True, norm='zscore', abs_corr=False):
    """
    One page per (protein, radius): normalized trait profile vs. normalized
    ClinVar ratio (normalization method set by norm: 'zscore', 'minmax', or
    'robust'). If annotate is True, a UniProt feature annotation track
    (domains, sites, motifs, etc.) is drawn below each line plot, sharing
    the aa_pos x-axis, so functionally known regions of the protein are
    visible alongside the signal. Annotations are fetched once per protein
    (not per radius) and skipped silently (with a logged warning) if the
    UniProt API call fails, e.g. no network access.

    If abs_corr is True, the trait series is sign-flipped (multiplied by -1)
    whenever its raw Spearman correlation with the ClinVar series is
    negative, since the sign of that correlation is just a convention of how
    the trait/ratio happen to be coded; the title then reports the
    (now-positive) correlation and notes that the trait was flipped.
    """
    plt, PdfPages = _import_matplotlib()
    df_trait = _apply_nbhd_subset(df_trait, nbhd_subset, value_col)
    clinvar_value_col = value_col if value_col in df_clinvar.columns else 'mean_betahat'
    df_clinvar = _apply_nbhd_subset(df_clinvar, nbhd_subset, clinvar_value_col)
    annotation_cache = {}
    with PdfPages(pdf_path) as pdf:
        combos = df_trait[['uniprot_id', 'radius', 'trait']].drop_duplicates()
        for _, combo in combos.iterrows():
            protein, radius, trait = combo['uniprot_id'], combo['radius'], combo['trait']

            t = df_trait[(df_trait.uniprot_id == protein) & (df_trait.radius == radius)].sort_values('aa_pos')
            c = df_clinvar[(df_clinvar.uniprot_id == protein) & (df_clinvar.radius == radius)].sort_values('aa_pos')
            if len(t) == 0:
                continue

            feats, seq_len = (None, None)
            if annotate:
                if protein not in annotation_cache:
                    annotation_cache[protein] = fetch_uniprot_features(protein)
                feats, seq_len = annotation_cache[protein]

            if feats:
                types = list(dict.fromkeys(f[0] for f in feats))
                fig, (ax, axt) = plt.subplots(
                    2, 1, sharex=True, figsize=(10, 4 + 0.35 * len(types)),
                    height_ratios=[4, max(1, 0.35 * len(types))],
                )
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                axt = None

            title = f'{protein} | radius={radius}'
            rho = n_pairs = perm_p = np.nan
            flip_sign = 1.0
            if len(c) > 0:
                # align on aa_pos so both series share the same positions in order
                t_vals = t[['aa_pos', value_col]].rename(columns={value_col: '_trait_val'})
                c_vals = c[['aa_pos', clinvar_value_col]].rename(columns={clinvar_value_col: '_clinvar_val'})
                merged = t_vals.merge(c_vals, on='aa_pos', how='outer').sort_values('aa_pos')
                rho, n_pairs, perm_p = _spearman_with_circular_perm(
                    merged['_trait_val'], merged['_clinvar_val'], n_perm=n_perm
                )
                if abs_corr and np.isfinite(rho) and rho < 0:
                    flip_sign = -1.0

            z_t = _normalize(flip_sign * t[value_col].to_numpy(dtype=float), norm)
            trait_label = f'trait {trait} ({value_col})'
            if flip_sign < 0:
                trait_label += ', flipped'
            ax.plot(t['aa_pos'], z_t, label=trait_label, color='tab:blue')

            if 'is_significant_discovery' in t.columns:
                disc_mask = t['is_significant_discovery'].fillna(False).to_numpy()
                other_mask = (t['is_significant_other_trait'].fillna(False).to_numpy()
                              if 'is_significant_other_trait' in t.columns
                              else np.zeros_like(disc_mask)) & ~disc_mask
                x_t = t['aa_pos'].to_numpy()
                if disc_mask.any():
                    ax.scatter(x_t[disc_mask], z_t[disc_mask], color='tab:blue', marker='o',
                               s=50, zorder=10, edgecolor='black', linewidth=0.6,
                               label=f'significant (discovery trait, n={int(disc_mask.sum())})')
                if other_mask.any():
                    ax.scatter(x_t[other_mask], z_t[other_mask], color='tab:blue', marker='D',
                               s=45, zorder=10, edgecolor='black', linewidth=0.6,
                               label=f'significant (other trait, n={int(other_mask.sum())})')

            if len(c) > 0:
                z_c = _normalize(c[clinvar_value_col].to_numpy(dtype=float), norm)
                ax.plot(c['aa_pos'], z_c, label=f'ClinVar ({clinvar_value_col})', color='tab:red')
                if 'is_significant_clinvar' in c.columns:
                    cv_mask = c['is_significant_clinvar'].fillna(False).to_numpy()
                    if cv_mask.any():
                        x_c = c['aa_pos'].to_numpy()
                        ax.scatter(x_c[cv_mask], z_c[cv_mask], color='tab:red', marker='o',
                                   s=50, zorder=5, edgecolor='black', linewidth=0.6,
                                   label=f'significant (ClinVar, n={int(cv_mask.sum())})')
                display_rho = abs(rho) if (abs_corr and np.isfinite(rho)) else rho
                if np.isnan(rho):
                    title += ' | Spearman rho: NA (too few shared positions)'
                elif n_perm > 0:
                    title += f' | Spearman rho={display_rho:.3f} (n={n_pairs}, perm p={perm_p:.3g})'
                else:
                    title += f' | Spearman rho={display_rho:.3f} (n={n_pairs})'
                if flip_sign < 0:
                    title += ' | trait sign flipped (--abs-corr)'
            ax.set_ylabel(f'{norm}-normalized effect')
            if nbhd_subset:
                title += ' | neighborhood subset applied'
            ax.set_title(title, fontsize=10)
            ax.legend()

            if axt is not None:
                _draw_annotation_track(axt, feats, seq_len)
            else:
                ax.set_xlabel('aa_pos')

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def parse_nbhd_subset(subset_arg):
    """
    Parse --nbhd-subset, or None. subset_arg must be a path to a tab-delimited
    file with columns uniprot_id, aa_pos, radius (one neighborhood/radius per
    line, since the plot shows one page per radius). Returns a set of
    (uniprot_id, aa_pos, radius) tuples.
    """
    if not subset_arg:
        return None
    df = pd.read_csv(subset_arg, sep='\t', dtype={'uniprot_id': str})
    for col in ('uniprot_id', 'aa_pos', 'radius'):
        if col not in df.columns:
            raise ValueError(f"--nbhd-subset file {subset_arg} must have columns "
                              f"'uniprot_id', 'aa_pos', and 'radius'; missing '{col}'.")
    return set(zip(df['uniprot_id'], df['aa_pos'].astype(int), df['radius'].astype(float)))


def plot_profiles_from_files(trait_tsv, clinvar_tsv, plot_file, nbhd_subset=None,
                              value_col='mean_betahat', n_perm=1000, annotate=True,
                              norm='zscore', abs_corr=False):
    """
    Standalone replotting entry point: read the two tsvs already produced by
    --explore-nhbds-traits and generate the comparison PDF, optionally
    restricting to a subset of neighborhoods (uniprot_id, aa_pos, radius) and
    choosing which column to plot (e.g. 'mean_betahat' or
    'mean_betahat_shrunk'). Positions outside the subset are treated as NA,
    but the full protein sequence is still plotted. If value_col isn't
    present in the ClinVar tsv (e.g. a shrunk column that only exists for
    the trait), the ClinVar series falls back to 'mean_betahat'. Each plot
    title reports the Spearman rank correlation between the two series
    (pairwise-complete, NAs dropped from both), plus a circular-shift
    permutation p-value (set n_perm=0 to skip it). If annotate is True
    (default), a UniProt feature annotation track is drawn below each plot
    (requires network access; set annotate=False to skip). norm selects the
    normalization used for plotting ('zscore', 'minmax', or 'robust'). If
    abs_corr is True, the trait series is sign-flipped when its raw
    correlation with ClinVar is negative, since that sign is just a
    convention (see _plot_profiles docstring).
    """
    df_trait = pd.read_csv(trait_tsv, sep='\t')
    df_clinvar = pd.read_csv(clinvar_tsv, sep='\t')
    if value_col not in df_trait.columns:
        raise ValueError(f"--value-col '{value_col}' not found in {trait_tsv}; "
                          f"available columns: {list(df_trait.columns)}")
    _plot_profiles(df_trait, df_clinvar, plot_file, nbhd_subset=nbhd_subset,
                    value_col=value_col, n_perm=n_perm, annotate=annotate,
                    norm=norm, abs_corr=abs_corr)
    logger.info(f"Wrote profile comparison plot to {plot_file}")


def run_explore_nbhds_profile(
        fdr_results_file,
        trait_cluster_file,
        radii_arg,
        clinvar_file,
        rvas_data_to_map_template,
        beta_col,
        reference_dir,
        genome_build,
        pae_cutoff,
        output_file,
        clinvar_output_file,
        fdr_cutoff=None,
        shrink_k=5.0,
        clinvar_fdr_file=None,
):
    radii = parse_radii(radii_arg)
    df_best = select_best_trait_per_protein(fdr_results_file, trait_cluster_file, fdr_cutoff)
    logger.info(f"Selected best trait for {len(df_best)} proteins; radii={radii}")

    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'

    logger.info(f"Mapping ClinVar/common variants from {clinvar_file}")
    df_clinvar_mapped = map_to_protein(
        clinvar_file, 'Variant ID', 'ac_case', 'ac_control', reference_dir,
        which_proteins='all', genome_build=genome_build,
    )
    df_clinvar_mapped = df_clinvar_mapped[df_clinvar_mapped.uniprot_id.isin(df_best['uniprot_id'])]

    trait_rows = []
    clinvar_rows = []

    for trait, group in df_best.groupby('trait'):
        raw_path = rvas_data_to_map_template.format(trait=trait)
        if not os.path.exists(raw_path):
            logger.error(f"{trait}: raw rvas-to-map file not found at {raw_path}; skipping "
                         f"{group['uniprot_id'].nunique()} protein(s).")
            continue

        logger.info(f"{trait}: mapping raw variants from {raw_path} for "
                    f"{group['uniprot_id'].nunique()} protein(s)")
        df_mapped_trait = map_to_protein_q(raw_path, beta_col, reference_dir, genome_build=genome_build)
        df_mapped_trait = df_mapped_trait[df_mapped_trait.uniprot_id.isin(group['uniprot_id'])]

        for _, r in group.iterrows():
            protein = r['uniprot_id']
            adj_by_radius = _compute_adjacency_by_radius(
                pdb_file_pos_guide, pdb_dir, pae_dir, protein, radii, pae_cutoff
            )
            if adj_by_radius is None:
                logger.warning(f"{protein}: no structure found; skipping.")
                continue

            for row in compute_betahat_profile(df_mapped_trait, protein, adj_by_radius, shrink_k=shrink_k):
                row.update({'trait': trait, 'uniprot_id': protein,
                            'cluster': r['cluster'], 'discovery_fdr': r['discovery_fdr'],
                            'discovery_aa_pos': r['discovery_aa_pos']})
                if 'gene_name' in r:
                    row['gene_name'] = r['gene_name']
                trait_rows.append(row)

            for row in compute_clinvar_profile(df_clinvar_mapped, protein, adj_by_radius):
                row.update({'trait': trait, 'uniprot_id': protein,
                            'cluster': r['cluster'], 'discovery_fdr': r['discovery_fdr'],
                            'discovery_aa_pos': r['discovery_aa_pos']})
                if 'gene_name' in r:
                    row['gene_name'] = r['gene_name']
                clinvar_rows.append(row)

    df_out = pd.DataFrame(trait_rows)
    df_clinvar_out = pd.DataFrame(clinvar_rows)

    # Significance markers for plotting: is_significant_discovery/
    # is_significant_other_trait on the trait tsv, is_significant_clinvar on
    # the ClinVar tsv. Computed here (not at plot time) so the tsvs are
    # self-contained and the plotting script needs no fdr-file access.
    if fdr_cutoff is not None:
        sig_by_protein = load_significant_neighborhoods(fdr_results_file, fdr_cutoff, df_best)
        disc_rows = [(uid, pos) for uid, (dset, _) in sig_by_protein.items() for pos in dset]
        other_rows = [(uid, pos) for uid, (_, oset) in sig_by_protein.items() for pos in oset]
        df_disc_sig = pd.DataFrame(disc_rows, columns=['uniprot_id', 'aa_pos'])
        df_disc_sig['is_significant_discovery'] = True
        df_other_sig = pd.DataFrame(other_rows, columns=['uniprot_id', 'aa_pos'])
        df_other_sig['is_significant_other_trait'] = True
        df_out = df_out.merge(df_disc_sig, on=['uniprot_id', 'aa_pos'], how='left')
        df_out = df_out.merge(df_other_sig, on=['uniprot_id', 'aa_pos'], how='left')
        df_out['is_significant_discovery'] = df_out['is_significant_discovery'].fillna(False)
        df_out['is_significant_other_trait'] = df_out['is_significant_other_trait'].fillna(False)
    else:
        logger.warning("fdr_cutoff is None; 'significant' is undefined without a cutoff, "
                        "skipping is_significant_discovery/is_significant_other_trait markers.")
        df_out['is_significant_discovery'] = False
        df_out['is_significant_other_trait'] = False

    if clinvar_fdr_file is not None and fdr_cutoff is not None:
        sig_clinvar_by_protein = load_significant_clinvar_neighborhoods(clinvar_fdr_file, fdr_cutoff)
        cv_rows = [(uid, pos) for uid, posset in sig_clinvar_by_protein.items() for pos in posset]
        df_cv_sig = pd.DataFrame(cv_rows, columns=['uniprot_id', 'aa_pos'])
        df_cv_sig['is_significant_clinvar'] = True
        df_clinvar_out = df_clinvar_out.merge(df_cv_sig, on=['uniprot_id', 'aa_pos'], how='left')
        df_clinvar_out['is_significant_clinvar'] = df_clinvar_out['is_significant_clinvar'].fillna(False)
    else:
        if clinvar_fdr_file is None:
            logger.info("--clinvar-fdr-file not given; skipping is_significant_clinvar marker.")
        df_clinvar_out['is_significant_clinvar'] = False

    col_order = ['trait', 'uniprot_id', 'aa_pos', 'n_betahat', 'mean_betahat', 'mean_betahat_shrunk',
                 'se_betahat', 'radius', 'cluster', 'discovery_fdr', 'discovery_aa_pos', 'gene_name']

    trait_col_order = col_order + ['is_significant_discovery', 'is_significant_other_trait']
    df_out = df_out[[c for c in trait_col_order if c in df_out.columns]]
    df_out.to_csv(output_file, sep='\t', index=False)
    logger.info(f"Wrote {len(df_out)} rows to {output_file}")

    clinvar_col_order = col_order + ['se_ratio_approx', 'nbhd_case', 'nbhd_control', 'is_significant_clinvar']
    df_clinvar_out = df_clinvar_out[[c for c in clinvar_col_order if c in df_clinvar_out.columns]]
    df_clinvar_out.to_csv(clinvar_output_file, sep='\t', index=False)
    logger.info(f"Wrote {len(df_clinvar_out)} rows to {clinvar_output_file}")

    return df_out, df_clinvar_out


def _main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Standalone replot of --explore-nhbds-traits output, optionally '
                     'restricted to a subset of neighborhoods (aa_pos).'
    )
    parser.add_argument('--trait-tsv', required=True, help='trait profile tsv (--profile-output-file)')
    parser.add_argument('--clinvar-tsv', required=True, help='clinvar profile tsv (the _clinvar.tsv output)')
    parser.add_argument('--plot-file', required=True, help='output PDF path')
    parser.add_argument('--nbhd-subset', type=str, default=None,
                         help='path to a tab-delimited file with columns uniprot_id, '
                              'aa_pos, radius listing the neighborhoods to keep (one '
                              'row per radius page); all other neighborhoods are set '
                              'to NA, but the full sequence is still plotted. '
                              'Default: no filter.')
    parser.add_argument('--value-col', type=str, default='mean_betahat',
                         choices=['mean_betahat', 'mean_betahat_shrunk'],
                         help='which trait column to plot (and ClinVar, if present); '
                              'mean_betahat_shrunk is a shrinkage-toward-0 estimate '
                              'that down-weights low-n_betahat neighborhoods. '
                              'Default: mean_betahat.')
    parser.add_argument('--n-perm', type=int, default=1000,
                         help='number of circular-shift permutations for the Spearman '
                              'correlation p-value shown in each plot title (accounts '
                              'for spatial autocorrelation from overlapping '
                              'neighborhoods); set to 0 to skip and show only rho. '
                              'Default: 1000.')
    parser.add_argument('--no-annotate', action='store_true', default=False,
                         help='skip the UniProt feature annotation track (domains, '
                              'sites, motifs, etc.) drawn below each plot by default; '
                              'use this if you have no network access or want faster '
                              'plotting.')
    parser.add_argument('--norm', type=str, default='zscore',
                         choices=['zscore', 'minmax', 'robust'],
                         help='''
                         Normalization used to make the trait and ClinVar series
                         visually comparable. 'zscore' (default): (x-mean)/sd.
                         'minmax': rescale to [-1, 1]. 'robust': (x-median)/(1.4826*MAD),
                         less sensitive to outlier residues than zscore/minmax.
                         ''')
    parser.add_argument('--abs-corr', action='store_true', default=False,
                         help='''
                         If the raw Spearman correlation between the trait and
                         ClinVar series is negative, sign-flip the trait series
                         before plotting/normalizing (the sign is only a
                         convention of how the trait/ratio are coded). The title
                         then reports the resulting (positive) correlation and
                         notes the flip.
                         ''')
    args = parser.parse_args()
    nbhd_subset = parse_nbhd_subset(args.nbhd_subset)
    plot_profiles_from_files(args.trait_tsv, args.clinvar_tsv, args.plot_file,
                              nbhd_subset=nbhd_subset, value_col=args.value_col, n_perm=args.n_perm,
                              annotate=not args.no_annotate, norm=args.norm, abs_corr=args.abs_corr)


if __name__ == '__main__':
    _main()
