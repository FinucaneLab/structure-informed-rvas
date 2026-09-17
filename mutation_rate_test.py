"""
3D neighborhood test with a mutation-rate null.

For a neighborhood N(i) with observed de novo count X_i and summed mutation rate M_i,
in a gene with totals N_g and M_g over structure-covered residues:

  Test A (poisson)   X_i ~ Poisson(lambda_hat * M_i)
  Test B (binomial)  X_i | N_g ~ Binomial(N_g, M_i / M_g)

Test B is the test of interest: a within-gene comparison of a neighborhood against the
rest of the gene, with rate-weighted rest-of-gene replacing the control variants used by
the standard 3DNT.

Test A guards against the specific way Test B fails under regional constraint. When part
of a gene is depleted for variation because mutations there are incompatible with life,
the de novos that do exist are pushed into the unconstrained regions, and a neighborhood
sitting in one of those looks enriched relative to M_i / M_g without being special. Such a
neighborhood sits at X_i ~= lambda_hat * M_i, so Test A does not call it.

Both tests are one-sided (upper tail). Each is corrected separately by the existing
empirical FDR/FWER machinery; a neighborhood passes when it passes both, which is
recorded as fdr_max = max(fdr_poisson, fdr_binomial) and likewise for FWER.

M_i must be the summed rate of ALL possible missense SNVs in the neighborhood, so it
comes from the precomputed per-residue table (precompute_mu_per_residue.py), not from
the variant file. A cohort de novo file lists only variants someone observed: taking
the universe from one makes M_g an observation-selected subset of the gene's real
opportunity. For the ASD counts file that subset was ~1% of possible missense variants
and skewed observed/expected by up to 19.7x across mutation-rate deciles, while the
full universe is flat to within 1.37x.
"""

import os
import h5py
import numpy as np
import pandas as pd
from scipy.stats import poisson, binom

from utils import get_adjacency_matrix, write_dataset
from logger_config import get_logger
from empirical_fdr import compute_fdr
from scan_test import MULTI_RADII_SMALL, MULTI_RADII_BIG, P_FLOOR

logger = get_logger(__name__)

POISSON_GROUP = 'poisson'
BINOMIAL_GROUP = 'binomial'


# ---------------------------------------------------------------------------
# per-residue mutation rate table
# ---------------------------------------------------------------------------

def load_mu_per_residue(path):
    """Load the precomputed per-(gene, residue) mutation rate table."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Per-residue mutation rate table not found at {path}. Build it with:\n'
            f'  python precompute_mu_per_residue.py --reference-dir <REFERENCE_DIR>'
        )
    df_mu = pd.read_parquet(path)
    logger.info(f'Loaded per-residue mutation rates: {len(df_mu)} residues across '
                f'{df_mu.uniprot_id.nunique()} genes from {os.path.basename(path)}')
    return df_mu


def build_mu_lookup(df_mu):
    """Group the table once into {uniprot_id: (aa_pos, mu, n_with_mu, n_possible)}."""
    lookup = {}
    for uniprot_id, g in df_mu.groupby('uniprot_id', sort=False):
        lookup[uniprot_id] = (g.aa_pos.values.astype(int),
                              g.mu.values.astype(float),
                              g.n_with_mu.values.astype(float),
                              g.n_possible.values.astype(float))
    return lookup


def mu_vector_for_protein(mu_lookup, uniprot_id, n_res):
    """
    Per-residue mutation rate as an (n_res, 1) column, plus the rate-table coverage
    over the residues the structure covers.
    """
    entry = mu_lookup.get(uniprot_id)
    if entry is None:
        raise ValueError(f'no mutation rate data for {uniprot_id}')
    aa_pos, mu, n_with_mu, n_possible = entry
    in_range = (aa_pos >= 1) & (aa_pos <= n_res)
    m = np.zeros((n_res, 1), dtype=float)
    m[aa_pos[in_range] - 1, 0] = mu[in_range]
    denom = n_possible[in_range].sum()
    coverage = float(n_with_mu[in_range].sum() / denom) if denom > 0 else np.nan
    return m, coverage


# ---------------------------------------------------------------------------
# lambda_hat
# ---------------------------------------------------------------------------

def estimate_lambda(df_rvas, df_mu, rate_calibration, calibration_genes=None, n_trios=None):
    """
    Estimate the factor converting relative mutation rates into expected de novo counts.

    The denominator is the total rate over the FULL set of possible missense variants in
    the relevant genes (from df_mu), not over the variants present in the de novo file.
    Using the file's own variants understates the opportunity by ~100x and biases the
    result, because such a file contains only variants that were observed.

    MUST be called after all variant-level filters (missense mapping, rate join, common
    variant and LCR removal, max-AC) and BEFORE any gene-level selection. Gene-level
    selection is selection on the outcome: restricting to an associated gene set lets
    lambda_hat absorb the very enrichment Test A exists to verify, and applying
    --min-denovo first keeps only genes whose de novo count came out high. Measured on
    the ASD data, the variant-level filters move lambda_hat by under 0.2% while applying
    the gene filter first inflates it by 1.43x.
    """
    if rate_calibration == 'none':
        logger.info('Rate calibration: none (lambda_hat = 1). '
                    'Mutation rates are assumed to be absolute expected counts.')
        return 1.0

    if rate_calibration.startswith('fixed:'):
        lambda_hat = float(rate_calibration.split(':', 1)[1])
        logger.info(f'Rate calibration: fixed, lambda_hat = {lambda_hat:.6g}')
        return lambda_hat

    # Genes where a de novo could have been seen at all, i.e. those the input maps to.
    genes = set(df_rvas.uniprot_id.unique())
    if calibration_genes is not None:
        genes &= set(calibration_genes)
        logger.info(f'Estimating lambda_hat within {len(genes)} genes from --rate-calibration-genes')

    df_cal = df_rvas[df_rvas.uniprot_id.isin(genes)]
    mu_cal = df_mu[df_mu.uniprot_id.isin(genes)]

    # Not de-duplicated by variant: df_mu carries one row per (gene, residue), so a
    # variant shared by two genes contributes opportunity to both, and the matching
    # numerator must count its de novos once per gene too.
    sum_x = float(df_cal.ac_case.sum())
    sum_mu = float(mu_cal.mu.sum())
    if sum_mu <= 0:
        raise ValueError('Total mutation rate over the calibration set is zero; cannot estimate lambda_hat.')

    lambda_hat = sum_x / sum_mu
    logger.info(
        f'lambda_hat = {lambda_hat:.6g}  '
        f'(sum_x = {sum_x:.0f} de novos, sum_mu = {sum_mu:.6g} over all possible '
        f'missense variants in {len(genes)} genes)'
    )
    if n_trios is not None and n_trios > 0:
        # lambda_hat itself is not interpretable when the rate model is relative (as
        # Roulette's is), so check the quantity that is: de novo missense per trio.
        per_trio = sum_x / n_trios
        logger.info(f'Sanity check: {per_trio:.3f} de novo missense per trio '
                    f'({sum_x:.0f} over {n_trios} trios); expect roughly 0.6-1.1')
        if not 0.3 < per_trio < 2.0:
            logger.warning(
                f'{per_trio:.3f} de novo missense per trio is outside the plausible '
                'range. Check --n-trios, and whether the input is exome-wide.'
            )
    return lambda_hat


# ---------------------------------------------------------------------------
# per-residue aggregation
# ---------------------------------------------------------------------------

def restrict_to_structure(df, n_res, uniprot_id):
    """Drop variants whose residue position falls outside the solved structure."""
    in_range = (df.aa_pos >= 1) & (df.aa_pos <= n_res)
    n_out = int((~in_range).sum())
    if n_out:
        logger.warning(
            f'{uniprot_id}: dropping {n_out} variants with aa_pos outside the structure '
            f'(n_res = {n_res})'
        )
        df = df[in_range]
    return df


def sum_per_residue(df, colname, n_res, dtype=float):
    """Sum a per-variant column onto residues. Returns an (n_res, 1) column vector."""
    out = np.zeros((n_res, 1), dtype=dtype)
    by_pos = df.groupby('aa_pos')[colname].sum()
    out[by_pos.index.values.astype(int) - 1, 0] = by_pos.values
    return out


# ---------------------------------------------------------------------------
# p-value lookup tables
#
# M_i is continuous so it cannot index a table, but it is fixed per residue across all
# simulations and X_i is a small integer. Building L[i, k] once costs n_res * (x_max + 1)
# special-function calls instead of n_res * (n_sims + 1), so the cost does not grow with
# --n-sims.
# ---------------------------------------------------------------------------

def poisson_pval_lookup(expected, x_max, direction='enrichment'):
    """
    L[i, k] = P(X >= k) for X ~ Poisson(expected[i]), or P(X <= k) for depletion.
    Shape (n_res, x_max + 1).
    """
    k = np.arange(x_max + 1)
    expected = np.maximum(np.asarray(expected, dtype=float), P_FLOOR)
    if direction == 'depletion':
        return poisson.cdf(k[np.newaxis, :], expected[:, np.newaxis])
    return poisson.sf(k[np.newaxis, :] - 1, expected[:, np.newaxis])


def binomial_pval_lookup(p_nbhd, n_trials, x_max, direction='enrichment'):
    """
    L[i, k] = P(X >= k) for X ~ Binomial(n_trials, p_nbhd[i]), or P(X <= k) for depletion.
    Shape (n_res, x_max + 1).
    """
    k = np.arange(x_max + 1)
    p_nbhd = np.clip(np.asarray(p_nbhd, dtype=float), 0.0, 1.0)
    if direction == 'depletion':
        return binom.cdf(k[np.newaxis, :], n_trials, p_nbhd[:, np.newaxis])
    return binom.sf(k[np.newaxis, :] - 1, n_trials, p_nbhd[:, np.newaxis])


# ---------------------------------------------------------------------------
# null simulation
# ---------------------------------------------------------------------------

def simulate_poisson_null(m, lambda_hat, n_sims, rng):
    """
    Test A null: de novos drawn independently per residue at the modelled rate.

    Deliberately NOT conditioned on the --min-denovo gene filter. That filter is there to
    spend compute where there is power, not as part of the inferential design, so the null
    stays the plain rate model.

    The cost of that choice has to be carried into interpretation rather than the code.
    --min-denovo keeps genes whose de novo count came out high, so the observed data sits
    above this null and Test A's empirical FDR/FWER are anti-conservative. Measured on the
    simulated-null ASD run: selected genes had a median observed/expected gene total of
    1.50 (1.39x in aggregate), and Test A's per-gene FWER came out at 14.3% against a
    nominal 5%.

    Two consequences:
      - Test A's standalone FDR/FWER are not calibrated and should not be quoted as
        significance. Use it as a direction-of-effect guard.
      - Because Test A IS the guard against regional constraint, a too-liberal Test A
        under-flags. "Passes Test B but not Test A" coming out empty is weak evidence,
        not a demonstration that no hit is a constraint artifact.

    Test B is unaffected: conditioning on N_g cancels the selection exactly (4.3% on the
    same run). And the primary criterion max(fdr_poisson, fdr_binomial) < q requires
    fdr_binomial < q, so the reported set is a subset of the Test B set and inherits its
    validity.
    """
    return rng.poisson(lambda_hat * m, size=(m.shape[0], n_sims))


def simulate_multinomial_null(m, n_g, n_sims, rng):
    """
    Test B null: the gene's N_g de novos redistributed across residues in proportion to
    mutation rate. Preserves N_g, the exact analogue of the multivariate hypergeometric
    step in the standard 3DNT.
    """
    total = m.sum()
    if total <= 0:
        return np.zeros((m.shape[0], n_sims), dtype=int)
    p = (m.flatten() / total)
    p = p / p.sum()  # guard against float drift
    return rng.multinomial(int(n_g), p, size=n_sims).T


# ---------------------------------------------------------------------------
# core computation
# ---------------------------------------------------------------------------

def _pvals_for_radius(adjacency_matrix, x_a, x_b, m, lambda_hat, n_g, m_g,
                      direction='enrichment'):
    """
    Returns (p_a, p_b, x_nbhd_obs, exp_a, exp_b) for one adjacency matrix.
    x_a / x_b are (n_res, n_sims + 1) with the observed counts in column 0.
    """
    # float32 rather than integer matmul: numpy has no BLAS path for integer dtypes, and
    # this product is n_res^2 * n_sims, the dominant cost of a run. Counts are small
    # integers and exactly representable, so rounding back is lossless.
    adj32 = adjacency_matrix.astype(np.float32)
    nbhd_a = np.rint(adj32 @ x_a.astype(np.float32)).astype(np.int64)
    nbhd_b = np.rint(adj32 @ x_b.astype(np.float32)).astype(np.int64)
    m_nbhd = (adjacency_matrix @ m).flatten()

    exp_a = lambda_hat * m_nbhd
    p_frac = m_nbhd / m_g if m_g > 0 else np.zeros_like(m_nbhd)
    exp_b = n_g * p_frac

    lookup_a = poisson_pval_lookup(exp_a, int(nbhd_a.max()), direction)
    lookup_b = binomial_pval_lookup(p_frac, n_g, int(nbhd_b.max()), direction)

    rows = np.arange(adjacency_matrix.shape[0])[:, np.newaxis]
    p_a = lookup_a[rows, nbhd_a]
    p_b = lookup_b[rows, nbhd_b]
    return p_a, p_b, nbhd_a[:, 0], exp_a, exp_b


def _harmonic_mean_combine(pval_mats):
    """Combine p-values across radii, as the standard 3DNT does for --neighborhood-radius multiple-*."""
    stacked = np.maximum(np.stack(pval_mats, axis=0), P_FLOOR)
    combined = stacked.shape[0] / np.sum(1.0 / stacked, axis=0)
    best_idx = np.argmin(stacked[:, :, 0], axis=0)
    return combined, best_idx


def compute_all_pvals_mu(
        df,
        pdb_file_pos_guide,
        pdb_dir,
        pae_dir,
        uniprot_id,
        n_sims,
        lambda_hat,
        mu_lookup,
        radius=15,
        pae_cutoff=15,
        seed=None,
        direction='enrichment',
):
    """
    Returns (df_pvals_poisson, df_pvals_binomial, adjacency_matrix, mu_coverage)
    for one protein.
    """
    rng = np.random.default_rng(seed)

    multi = radius in ('multiple-small', 'multiple-big')
    radii = (MULTI_RADII_SMALL if radius == 'multiple-small' else MULTI_RADII_BIG) if multi else [radius]

    adj_matrices = {
        r: get_adjacency_matrix(pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, r, pae_cutoff)
        for r in radii
    }
    if any(a is None for a in adj_matrices.values()):
        raise FileNotFoundError(f'No structure available for {uniprot_id}')

    n_res = adj_matrices[radii[0]].shape[0]
    df = restrict_to_structure(df, n_res, uniprot_id)

    x_obs = sum_per_residue(df, 'ac_case', n_res, dtype=int)
    # Opportunity comes from the full possible-missense universe, not from the observed
    # variants in df -- see the module docstring.
    m, mu_coverage = mu_vector_for_protein(mu_lookup, uniprot_id, n_res)
    n_g = int(x_obs.sum())
    m_g = float(m.sum())
    if m_g <= 0:
        raise ValueError(f'total mutation rate over the structure is zero for {uniprot_id}')

    # Observed counts sit in column 0 of both matrices, so the two tests see identical
    # observed data and differ only in their null.
    x_a = np.hstack([x_obs, simulate_poisson_null(m, lambda_hat, n_sims, rng)])
    x_b = np.hstack([x_obs, simulate_multinomial_null(m, n_g, n_sims, rng)])

    per_radius = {r: _pvals_for_radius(adj_matrices[r], x_a, x_b, m, lambda_hat, n_g, m_g,
                                       direction)
                  for r in radii}

    if multi:
        p_a, best_a = _harmonic_mean_combine([per_radius[r][0] for r in radii])
        p_b, best_b = _harmonic_mean_combine([per_radius[r][1] for r in radii])
        pos_idx = np.arange(n_res)
        nbhd_by_radius = np.stack([per_radius[r][2] for r in radii], axis=0)
        exp_a_by_radius = np.stack([per_radius[r][3] for r in radii], axis=0)
        exp_b_by_radius = np.stack([per_radius[r][4] for r in radii], axis=0)
        nbhd_a_obs, exp_a = nbhd_by_radius[best_a, pos_idx], exp_a_by_radius[best_a, pos_idx]
        nbhd_b_obs, exp_b = nbhd_by_radius[best_b, pos_idx], exp_b_by_radius[best_b, pos_idx]
        radius_a = np.array([radii[i] for i in best_a], dtype=float)
        radius_b = np.array([radii[i] for i in best_b], dtype=float)
        adjacency_matrix = adj_matrices.get(15, adj_matrices[radii[0]])
    else:
        p_a, p_b, nbhd_obs, exp_a, exp_b = per_radius[radii[0]]
        nbhd_a_obs = nbhd_b_obs = nbhd_obs
        radius_a = radius_b = float(radii[0])
        adjacency_matrix = adj_matrices[radii[0]]

    def _build(pval_matrix, nbhd_obs, expected, radius_col):
        pval_columns = ['p_value'] + [f'null_pval_{i}' for i in range(n_sims)]
        out = pd.DataFrame(columns=pval_columns, data=pval_matrix)
        out['nbhd_case'] = nbhd_obs
        out['nbhd_expected'] = expected
        out['radius'] = radius_col
        out['original_case'] = x_obs[:, 0]
        out['original_mu'] = m[:, 0]
        return out[['nbhd_case', 'nbhd_expected', 'radius', 'original_case', 'original_mu'] + pval_columns]

    return (_build(p_a, nbhd_a_obs, exp_a, radius_a),
            _build(p_b, nbhd_b_obs, exp_b, radius_b),
            adjacency_matrix, mu_coverage)


def write_df_pvals_mu(results_dir, uniprot_id, df_pvals, pval_file, group):
    with h5py.File(os.path.join(results_dir, pval_file), 'a') as fid:
        node = fid.require_group(group)
        null_pval_cols = [c for c in df_pvals.columns if c.startswith('null_pval')]
        write_dataset(node, f'{uniprot_id}', df_pvals[['p_value']])
        # float32 is plenty: null p-values are only ever rank-compared.
        write_dataset(node, f'{uniprot_id}_null_pval', df_pvals[null_pval_cols].astype(np.float32))
        write_dataset(node, f'{uniprot_id}_nbhd', df_pvals[['nbhd_case', 'nbhd_expected']])
        write_dataset(node, f'{uniprot_id}_radius', df_pvals[['radius']])
        write_dataset(node, f'{uniprot_id}_original', df_pvals[['original_case', 'original_mu']])


def simulate_null_input(df_rvas, mu_lookup, lambda_hat, seed=None):
    """
    Replace observed de novo counts with draws from the rate model, for the exome-wide
    null calibration check (--simulate-null-from-mu). Returns a frame shaped like the
    mapped input, one row per (gene, residue), so the rest of the pipeline -- including
    re-estimating lambda_hat and computing FDR -- runs exactly as it would on real data.
    """
    rng = np.random.default_rng(seed)
    genes = sorted(set(df_rvas.uniprot_id.unique()) & set(mu_lookup))
    parts = []
    for uniprot_id in genes:
        aa_pos, mu, _, _ = mu_lookup[uniprot_id]
        if mu.sum() <= 0:
            continue
        parts.append(pd.DataFrame({'uniprot_id': uniprot_id,
                                   'aa_pos': aa_pos,
                                   'ac_case': rng.poisson(lambda_hat * mu)}))
    df = pd.concat(parts, ignore_index=True)
    df['Variant ID'] = df.uniprot_id + ':' + df.aa_pos.astype(str)
    logger.info(
        f'Simulated null from the rate model: {int(df.ac_case.sum())} de novos across '
        f'{df.uniprot_id.nunique()} genes at lambda_hat = {lambda_hat:.6g}'
    )
    return df


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def _select_genes(df_rvas, df_fdr_filter, min_denovo, mu_lookup, min_mu_coverage=0.0):
    """
    Gene-level selection. Called AFTER estimate_lambda -- see the note there on why the
    order matters.

    Applied as strictly greater than min_denovo, matching the existing 3DNT gene filter
    in scan_test._filter_proteins_by_allele_count (which uses `> min_alleles`), so the
    default of 5 requires at least 6 de novos.
    """
    grouped = df_rvas.groupby('uniprot_id')['ac_case'].sum()
    uniprot_id_list = grouped[grouped > min_denovo].index.tolist()

    # A gene is only testable if it carries a nonzero total mutation rate. Genes with
    # rows in the table but zero total rate are those the rate model does not cover at
    # all -- the Roulette file is autosomes only, so every chrX gene lands here.
    missing = [u for u in uniprot_id_list if u not in mu_lookup]
    zero_rate = [u for u in uniprot_id_list
                 if u in mu_lookup and mu_lookup[u][1].sum() <= 0]
    if missing:
        logger.warning(f'{len(missing)} genes are absent from the rate table and are skipped')
    if zero_rate:
        logger.warning(
            f'{len(zero_rate)} genes have zero total mutation rate and are skipped. The '
            'Roulette rate file covers autosomes only, so chrX genes cannot be tested '
            'with it; supply a rate file with chrX to include them.'
        )
    skip = set(missing) | set(zero_rate)
    uniprot_id_list = [u for u in uniprot_id_list if u not in skip]

    # Rate-table coverage costs power rather than biasing the tests -- a variant's de novo
    # count and its rate are dropped together, so x_j and m_j stay on the same variant set
    # at every residue and the test simply covers a narrower region. Coverage gaps do
    # cluster spatially though (Moran's I p<0.05 in 29 of the 30 lowest-coverage analysis
    # genes), so in those genes some neighborhoods carry no rate data at all. Reported per
    # gene as mu_coverage; excluded only if asked for.
    if min_mu_coverage > 0:
        low = []
        for u in uniprot_id_list:
            _, _, n_with_mu, n_possible = mu_lookup[u]
            denom = n_possible.sum()
            if denom > 0 and (n_with_mu.sum() / denom) < min_mu_coverage:
                low.append(u)
        if low:
            logger.info(f'{len(low)} genes have mu_coverage < {min_mu_coverage} and are skipped')
            uniprot_id_list = [u for u in uniprot_id_list if u not in set(low)]

    if df_fdr_filter is not None:
        uniprot_id_list = list(np.intersect1d(uniprot_id_list, np.unique(df_fdr_filter.uniprot_id)))

    logger.info(
        f'Selected {len(uniprot_id_list)} of {len(grouped)} genes for analysis '
        f'(more than {min_denovo} de novos each)'
    )
    return uniprot_id_list


def _process_proteins_batch_mu(df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff,
                               results_dir, n_sims, pval_file, lambda_hat, mu_lookup,
                               seed=None, max_residues=None, direction='enrichment'):
    """Run both tests for each protein. Returns per-gene totals for the output table."""
    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'

    # Skip very long proteins up front. get_distance_matrix_structure allocates an
    # n_res x n_res float matrix, which is 9.4 GB for titin (34,350 residues); without
    # this the run thrashes before failing with MemoryError.
    struct_len = {}
    try:
        guide = pd.read_csv(pdb_file_pos_guide, sep='\t')
        ends = guide.pos_covered.str.extract(r'(\d+)\s*\]')[0].astype(float)
        struct_len = guide.assign(end=ends).groupby('uniprot_id')['end'].max().to_dict()
    except Exception as e:
        logger.warning(f'Could not read structure lengths for the size guard: {e}')

    if max_residues is not None:
        too_big = [u for u in uniprot_id_list if struct_len.get(u, 0) > max_residues]
        if too_big:
            logger.warning(
                f'{len(too_big)} proteins exceed --max-residues {max_residues} and are '
                f'skipped (the distance matrix scales as n_res^2): '
                f'{", ".join(sorted(too_big)[:10])}'
            )
            uniprot_id_list = [u for u in uniprot_id_list if u not in set(too_big)]

    gene_totals = {}
    n_proteins = len(uniprot_id_list)
    for i, uniprot_id in enumerate(uniprot_id_list):
        logger.info(f'Processing {uniprot_id} (protein {i+1} out of {n_proteins})')
        try:
            df = df_rvas[df_rvas.uniprot_id == uniprot_id]
            # Derive a per-protein stream from the seed so results do not depend on
            # how genes are split across parallel jobs.
            protein_seed = None if seed is None else [seed, i]
            df_a, df_b, _, mu_coverage = compute_all_pvals_mu(
                df, pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id,
                n_sims, lambda_hat, mu_lookup, radius, pae_cutoff, seed=protein_seed,
                direction=direction,
            )
            write_df_pvals_mu(results_dir, uniprot_id, df_a, pval_file, POISSON_GROUP)
            write_df_pvals_mu(results_dir, uniprot_id, df_b, pval_file, BINOMIAL_GROUP)
            gene_totals[uniprot_id] = (int(df_a['original_case'].sum()),
                                       float(df_a['original_mu'].sum()),
                                       mu_coverage)
        except FileNotFoundError as e:
            logger.error(f'{uniprot_id}: Required file not found - {e}')
        except KeyError as e:
            logger.error(f'{uniprot_id}: Missing required column or key - {e}')
        except ValueError as e:
            logger.error(f'{uniprot_id}: Invalid data or parameter - {e}')
        except MemoryError as e:
            logger.error(f'{uniprot_id}: Insufficient memory for processing - {e}')
        except Exception as e:
            logger.error(f'{uniprot_id}: Unexpected error - {e}')
    return gene_totals


def _merge_test_results(df_a, df_b, fdr_cutoff=0.05, fwer_cutoff=0.05):
    """Join the two independently corrected result tables and take the max FDR/FWER."""
    a = df_a.rename(columns={'p_value': 'p_poisson', 'fdr': 'fdr_poisson',
                             'fwer': 'fwer_poisson', 'nbhd_expected': 'exp_poisson',
                             'obs_exp': 'obs_exp_poisson', 'radius': 'radius_poisson'})
    b = df_b.rename(columns={'p_value': 'p_binomial', 'fdr': 'fdr_binomial',
                             'fwer': 'fwer_binomial', 'nbhd_expected': 'exp_binomial',
                             'obs_exp': 'obs_exp_binomial', 'radius': 'radius_binomial'})
    keep_b = ['uniprot_id', 'aa_pos', 'p_binomial', 'fdr_binomial', 'fwer_binomial',
              'exp_binomial', 'obs_exp_binomial', 'radius_binomial']
    merged = a.merge(b[keep_b], on=['uniprot_id', 'aa_pos'], how='inner')

    # A neighborhood passes only if it passes both tests.
    merged['fdr_max'] = merged[['fdr_poisson', 'fdr_binomial']].max(axis=1)
    merged['fwer_max'] = merged[['fwer_poisson', 'fwer_binomial']].max(axis=1)
    merged['sig_fwer_both'] = merged['fwer_max'] < fwer_cutoff
    merged['sig_fdr_both'] = merged['fdr_max'] < fdr_cutoff
    return merged


def _compute_both_fdrs(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file,
                       fwer_cutoff=0.05):
    logger.info('Computing FDR and FWER for Test A (Poisson, absolute)')
    df_a = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file,
                       group=POISSON_GROUP, mu_mode=True)
    logger.info('Computing FDR and FWER for Test B (binomial, vs rest of gene)')
    df_b = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file,
                       group=BINOMIAL_GROUP, mu_mode=True)
    return _merge_test_results(df_a, df_b, fdr_cutoff, fwer_cutoff)


def summarize_mu_results(df_results, fdr_cutoff):
    top = df_results.loc[df_results.groupby('uniprot_id')['fdr_max'].idxmin()]
    sig = top[top.fdr_max < fdr_cutoff].sort_values(by='fdr_max')
    logger.info('')
    logger.info(f'{len(sig)} of {len(top)} proteins have a neighborhood passing BOTH tests '
                f'at FDR < {fdr_cutoff}')
    if len(sig) > 0:
        logger.info(f'Top 20 hits:\n{sig[0:20].to_string()}')

    b_only = df_results[(df_results.fdr_binomial < fdr_cutoff) & (df_results.fdr_poisson >= fdr_cutoff)]
    a_only = df_results[(df_results.fdr_poisson < fdr_cutoff) & (df_results.fdr_binomial >= fdr_cutoff)]
    logger.info(
        f'{len(b_only)} neighborhoods pass Test B but NOT Test A -- these are the ones '
        'regional constraint would otherwise have handed us as false positives.'
    )
    logger.info(f'{len(a_only)} neighborhoods pass Test A but not Test B.')
    n_fwer = int(df_results.sig_fwer_both.sum())
    n_fwer_genes = int(df_results.loc[df_results.sig_fwer_both, 'uniprot_id'].nunique())
    logger.info(f'{n_fwer} neighborhoods in {n_fwer_genes} genes pass both tests at '
                f'FWER < 0.05 (column sig_fwer_both).')


def mutation_rate_scan_test(
    df_rvas,
    reference_dir,
    radius,
    pae_cutoff,
    results_dir,
    n_sims,
    no_fdr,
    fdr_only,
    fdr_cutoff,
    df_fdr_filter,
    fdr_file,
    pval_file,
    rate_calibration,
    rate_calibration_genes,
    min_denovo,
    n_trios,
    seed=None,
    mu_residue_file=None,
    mu_from_input=False,
    min_mu_coverage=0.0,
    max_residues=10000,
    simulate_null=False,
    direction='enrichment',
):
    """3D neighborhood test against a mutation-rate null. See module docstring."""

    if fdr_only:
        df_results = _compute_both_fdrs(results_dir, fdr_cutoff, df_fdr_filter,
                                        reference_dir, pval_file)
        summarize_mu_results(df_results, fdr_cutoff)
        df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
        return

    logger.info(f'Input dataset contains {len(df_rvas)} variants across '
                f'{df_rvas["uniprot_id"].nunique()} proteins')
    if direction == 'depletion':
        logger.info('Testing DEPLETION: lower-tail p-values, i.e. fewer variants in the '
                    'neighborhood than the mutation rate predicts.')

    calibration_genes = None
    if rate_calibration_genes is not None:
        calibration_genes = pd.read_csv(rate_calibration_genes, sep='\t')['uniprot_id'].unique()

    if mu_from_input:
        # --mu-col: the universe is whatever the input file lists. Correct only if that
        # file enumerates every possible missense variant, with zero-count rows kept.
        logger.warning(
            'Using --mu-col, so mutational opportunity is taken from the input file. '
            'This is only valid if the file lists ALL possible missense variants, '
            'including those with no observed de novos. A file restricted to observed '
            'variants will understate opportunity and bias both tests.'
        )
        df_mu = (df_rvas.groupby(['uniprot_id', 'aa_pos'], as_index=False)
                 .agg(mu=('mu', 'sum')))
        df_mu['n_with_mu'] = np.nan
        df_mu['n_possible'] = np.nan
    else:
        df_mu = load_mu_per_residue(mu_residue_file)

    mu_lookup = build_mu_lookup(df_mu)

    # lambda_hat BEFORE gene-level selection -- see estimate_lambda.
    lambda_hat = estimate_lambda(df_rvas, df_mu, rate_calibration, calibration_genes, n_trios)

    if simulate_null:
        # Generate with the real lambda_hat, then re-estimate from the synthetic counts
        # so the check exercises the calibration step too.
        df_rvas = simulate_null_input(df_rvas, mu_lookup, lambda_hat, seed)
        lambda_hat = estimate_lambda(df_rvas, df_mu, rate_calibration, calibration_genes, n_trios)

    uniprot_id_list = _select_genes(df_rvas, df_fdr_filter, min_denovo, mu_lookup, min_mu_coverage)
    if len(uniprot_id_list) == 0:
        raise ValueError('No genes passed the --min-denovo filter; nothing to test.')

    gene_totals = _process_proteins_batch_mu(
        df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff,
        results_dir, n_sims, pval_file, lambda_hat, mu_lookup, seed, max_residues,
        direction,
    )

    with h5py.File(os.path.join(results_dir, pval_file), 'a') as fid:
        fid.attrs['test_type'] = 'mutation_rate'
        fid.attrs['lambda_hat'] = lambda_hat
        fid.attrs['rate_calibration'] = rate_calibration
        fid.attrs['n_sims'] = n_sims
        fid.attrs['min_denovo'] = min_denovo
        fid.attrs['direction'] = direction
        if seed is not None:
            fid.attrs['seed'] = seed

    if no_fdr:
        return

    fdr_filter = df_fdr_filter
    if fdr_filter is None:
        fdr_filter = pd.DataFrame({'uniprot_id': list(gene_totals.keys())})
    df_results = _compute_both_fdrs(results_dir, fdr_cutoff, fdr_filter,
                                    reference_dir, pval_file)

    df_totals = pd.DataFrame(
        [(uid, n_g, m_g, cov) for uid, (n_g, m_g, cov) in gene_totals.items()],
        columns=['uniprot_id', 'n_denovo_gene', 'mu_gene', 'mu_coverage'],
    )
    df_results = df_results.merge(df_totals, on='uniprot_id', how='left')

    summarize_mu_results(df_results, fdr_cutoff)
    df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
