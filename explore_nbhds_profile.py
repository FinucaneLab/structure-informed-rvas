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
4. Write two tsvs (trait profile and ClinVar profile). Plotting is a
   separate, manual step: run this file directly (see _main / plot_profiles_from_files)
   to generate the comparison PDF from the two tsvs, optionally restricting
   to a subset of neighborhoods and choosing which column to plot.
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


def _plot_profiles(df_trait, df_clinvar, pdf_path, nbhd_subset=None, value_col='mean_betahat'):
    """One page per (protein, radius): z-scored trait profile vs. z-scored ClinVar ratio."""
    plt, PdfPages = _import_matplotlib()
    df_trait = _apply_nbhd_subset(df_trait, nbhd_subset, value_col)
    clinvar_value_col = value_col if value_col in df_clinvar.columns else 'mean_betahat'
    df_clinvar = _apply_nbhd_subset(df_clinvar, nbhd_subset, clinvar_value_col)
    with PdfPages(pdf_path) as pdf:
        combos = df_trait[['uniprot_id', 'radius', 'trait']].drop_duplicates()
        for _, combo in combos.iterrows():
            protein, radius, trait = combo['uniprot_id'], combo['radius'], combo['trait']

            t = df_trait[(df_trait.uniprot_id == protein) & (df_trait.radius == radius)].sort_values('aa_pos')
            c = df_clinvar[(df_clinvar.uniprot_id == protein) & (df_clinvar.radius == radius)].sort_values('aa_pos')
            if len(t) == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t['aa_pos'], _zscore(t[value_col]), label=f'trait {trait} ({value_col})', color='tab:blue')
            if len(c) > 0:
                ax.plot(c['aa_pos'], _zscore(c[clinvar_value_col]),
                        label=f'ClinVar ({clinvar_value_col})', color='tab:red')
            ax.set_xlabel('aa_pos')
            ax.set_ylabel('z-scored effect')
            title = f'{protein} | radius={radius}'
            if nbhd_subset:
                title += ' | neighborhood subset applied'
            ax.set_title(title)
            ax.legend()
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
                              value_col='mean_betahat'):
    """
    Standalone replotting entry point: read the two tsvs already produced by
    --explore-nhbds-traits and generate the comparison PDF, optionally
    restricting to a subset of neighborhoods (uniprot_id, aa_pos, radius) and
    choosing which column to plot (e.g. 'mean_betahat' or
    'mean_betahat_shrunk'). Positions outside the subset are treated as NA,
    but the full protein sequence is still plotted. If value_col isn't
    present in the ClinVar tsv (e.g. a shrunk column that only exists for
    the trait), the ClinVar series falls back to 'mean_betahat'.
    """
    df_trait = pd.read_csv(trait_tsv, sep='\t')
    df_clinvar = pd.read_csv(clinvar_tsv, sep='\t')
    if value_col not in df_trait.columns:
        raise ValueError(f"--value-col '{value_col}' not found in {trait_tsv}; "
                          f"available columns: {list(df_trait.columns)}")
    _plot_profiles(df_trait, df_clinvar, plot_file, nbhd_subset=nbhd_subset, value_col=value_col)
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

    col_order = ['trait', 'uniprot_id', 'aa_pos', 'n_betahat', 'mean_betahat', 'mean_betahat_shrunk',
                 'se_betahat', 'radius', 'cluster', 'discovery_fdr', 'discovery_aa_pos', 'gene_name']

    df_out = pd.DataFrame(trait_rows)
    df_out = df_out[[c for c in col_order if c in df_out.columns]]
    df_out.to_csv(output_file, sep='\t', index=False)
    logger.info(f"Wrote {len(df_out)} rows to {output_file}")

    clinvar_col_order = col_order + ['se_ratio_approx', 'nbhd_case', 'nbhd_control']
    df_clinvar_out = pd.DataFrame(clinvar_rows)
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
    args = parser.parse_args()
    nbhd_subset = parse_nbhd_subset(args.nbhd_subset)
    plot_profiles_from_files(args.trait_tsv, args.clinvar_tsv, args.plot_file,
                              nbhd_subset=nbhd_subset, value_col=args.value_col)


if __name__ == '__main__':
    _main()
