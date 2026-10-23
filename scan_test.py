import os
import pandas as pd
import numpy as np
import h5py
from scipy.stats import fisher_exact
from utils import (get_adjacency_matrix, write_dataset, structure_length,
                   check_keep_mask, neighborhood_retention, build_conditioning_mask)
from logger_config import get_logger
from empirical_fdr import compute_fdr

logger = get_logger(__name__)

MULTI_RADII_SMALL = [6, 9, 12, 15, 18, 21]
MULTI_RADII_BIG = [10, 15, 20, 30, 40]
P_FLOOR = np.finfo(float).tiny

# A conditioned-out neighborhood leaves its neighbors only partly intact. Below this
# fraction of surviving residues a centre is not re-tested: what is left is a sliver of
# the original neighborhood, so a small p-value there is driven by a handful of variants
# at the rim rather than by residual structure. Overridable per run.
DEFAULT_MIN_RETAINED = 0.5

# Gene-level filter for the case/control scan: the minimum case and control alleles a
# gene needs, inclusive. Mirrors --min-denovo in the mutation-rate path.
DEFAULT_MIN_ALLELES = 5


def neutralized_positions(adjacency_matrix, keep, min_retained):
    """
    Which centres are not re-tested under a conditioning mask: the masked residues
    themselves, plus any whose neighborhood has lost too much of itself.

    Depends only on the structure and the mask -- never on the counts -- so the observed
    and null columns are neutralized identically and the empirical null stays calibrated.
    """
    if keep is None:
        return None
    retention = neighborhood_retention(adjacency_matrix, keep)
    # retention <= 0 is checked separately so that --conditional-min-retained 0
    # still cannot test a neighborhood with nothing left in it. Under the
    # mutation-rate null such a centre has expected count 0, which would hand any
    # stray count an arbitrarily small p-value.
    return (~keep) | (retention < min_retained) | (retention <= 0)


def _neutralize(pval_matrix, dead):
    """Set every column, observed and null alike, to 1.0 at the dead centres.

    Rows are never removed: the row index carries aa_pos (see the indexing contract in
    utils.py), and empirical_fdr rebuilds positions from row order."""
    if dead is None or not dead.any():
        return pval_matrix
    out = np.array(pval_matrix, dtype=float, copy=True)
    out[dead, :] = 1.0
    return out

def get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_ctrl,
                                 include_depletion=False):
    """
    Fisher p-value for every (cases in neighborhood, controls in neighborhood) pair.

    Enrichment only by default: a neighborhood with fewer case variants than expected
    gets p = 1 and is never called. The tool is for finding clusters of case variants,
    and spending multiple-testing burden on the opposite direction costs power -- on the
    ClinVar results 56% of significant neighborhoods were depleted. Pass
    include_depletion=True to test both directions.

    Note what this is NOT. The cells that survive keep the p-value fisher_exact returns
    by default, which is the TWO-SIDED one; this does not compute a one-sided p-value
    and is not equivalent to alternative='greater'. Enriched p-values are therefore
    unchanged from the two-sided run, which makes the two modes directly comparable, and
    the retained values are conservative relative to a true one-sided test. Significance
    here is empirical -- observed and null go through this same grid -- so the monotone
    difference cancels and the discovery set is what a one-sided test would give.

    The rule is applied HERE, to the lookup grid, for two reasons.

    It is where the null gets it for free. This grid is indexed into by the observed
    column and by all --n-sims null columns alike, so transforming it transforms the
    empirical null identically and FDR/FWER stay calibrated. Applied anywhere downstream
    it would have to be applied to the nulls too, and forgetting that would leave the
    null pool full of small depleted p-values the observed data could no longer produce
    -- an anti-conservative FDR, and a silent one.

    And it is a skip rather than a computation. Depleted cells fail the pre-filter below
    and keep the default 1.0, so fisher_exact is never called for them: roughly half the
    exact calls disappear.

    For a 2x2 table with fixed margins, OR < 1 is exactly a < expected_a, which the
    chi-square block already computes.
    """
    max_n_case_nbhd = np.max(n_case_nbhd_mat)
    max_n_control_nbhd = np.max(n_control_nbhd_mat)
    
    # Create the output array - default to 1.0 for all p-values
    pvals = np.ones((max_n_case_nbhd+1, max_n_control_nbhd+1))
    
    # Create meshgrid for all combinations
    case_nbhd_range = np.arange(max_n_case_nbhd + 1)
    ctrl_nbhd_range = np.arange(max_n_control_nbhd + 1)
    n_case_grid, n_ctrl_grid = np.meshgrid(case_nbhd_range, ctrl_nbhd_range, indexing='ij')
    
    # Calculate all values needed for Fisher's exact test
    a = n_case_grid
    b = n_ctrl_grid
    c = n_case - a
    d = n_ctrl - b
    
    # Construct contingency tables for validity check
    col1_sum = a + c
    col2_sum = b + d
    valid_columns = (col1_sum > 0) & (col2_sum > 0)
    
    row1_sum = a + b
    row2_sum = c + d
    valid_rows = (row1_sum > 0) & (row2_sum > 0)
    
    all_non_neg = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    
    # Combine all validity criteria
    valid_mask = valid_columns & valid_rows & all_non_neg
    
    # First-pass filter: Use Chi-square approximation to identify potential candidates
    # Calculate expected values under null hypothesis
    n_total = n_case + n_ctrl
    expected_a = row1_sum * col1_sum / n_total
    expected_b = row1_sum * col2_sum / n_total
    expected_c = row2_sum * col1_sum / n_total
    expected_d = row2_sum * col2_sum / n_total
    
    # Chi-square statistic - zeroing out invalid positions
    valid_array = valid_mask.astype(float)
    chi2 = valid_array * (
        ((a - expected_a)**2 / np.maximum(1e-10, expected_a)) + 
        ((b - expected_b)**2 / np.maximum(1e-10, expected_b)) + 
        ((c - expected_c)**2 / np.maximum(1e-10, expected_c)) + 
        ((d - expected_d)**2 / np.maximum(1e-10, expected_d))
    )
    
    # Use a more lenient threshold for chi-square to catch all potential p < 0.05
    # chi2_threshold corresponds to p-value slightly higher than 0.05 (e.g., 0.1)
    # For df=1, chi2 value of 2.706 corresponds to p=0.1
    chi2_threshold = 2.706
    potential_significant = (chi2 > chi2_threshold) & valid_mask
    if not include_depletion:
        # chi2 is direction-blind, so without this the depleted half of the grid is
        # computed and then reported. a >= expected_a is OR >= 1 at fixed margins.
        potential_significant = potential_significant & (a >= expected_a)
    
    # Second pass: Calculate exact p-values only for potentially significant cells
    indices = np.where(potential_significant)
    for i, j in zip(indices[0], indices[1]):
        table = np.array([[i, j], [n_case - i, n_ctrl - j]])
        _, p = fisher_exact(table)
        pvals[i, j] = p
    
    return pvals

def _compute_pval_matrix_for_radius(adjacency_matrix, case_ac_matrix, control_ac_matrix, n_case, n_control,
                                    include_depletion=False):
    n_case_nbhd = get_nbhd_counts(adjacency_matrix, case_ac_matrix)
    n_control_nbhd = get_nbhd_counts(adjacency_matrix, control_ac_matrix)
    pval_lookup = get_pval_lookup_case_control(n_case_nbhd, n_control_nbhd, n_case, n_control,
                                               include_depletion)
    return pval_lookup[n_case_nbhd, n_control_nbhd], n_case_nbhd, n_control_nbhd

def get_nbhd_counts(adjacency_matrix, ac_per_residue):
    # allele count per residue has one row per residue and one
    # column per simulation
    nbhd_counts = adjacency_matrix @ ac_per_residue
    nbhd_counts = nbhd_counts.astype(int)
    return nbhd_counts

def get_ac_per_residue(df, colname, n_res, uniprot_id=''):
    """
    Sum a per-variant allele count onto residues, as an (n_res, 1) column.

    Positions outside the solved structure are dropped with a warning rather than
    indexed blindly: aa_pos > n_res would raise and lose the whole protein, and
    aa_pos < 1 would index from the end and silently credit the wrong residue. The
    mutation-rate path does the same thing in restrict_to_structure.
    """
    ac_per_residue = np.zeros((n_res, 1))
    ac_by_pos = df.groupby('aa_pos')[colname].sum().reset_index()
    in_range = (ac_by_pos.aa_pos >= 1) & (ac_by_pos.aa_pos <= n_res)
    n_out = int((~in_range).sum())
    if n_out:
        logger.warning(
            f'{uniprot_id}: dropping {n_out} {colname} positions outside the structure '
            f'(n_res = {n_res})'
        )
        ac_by_pos = ac_by_pos[in_range]
    ac_per_residue[ac_by_pos.aa_pos.astype(int) - 1, 0] = ac_by_pos[colname]
    return ac_per_residue

def get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim, seed=None):
    # seed is passed straight to default_rng: a previous version called np.random.seed(),
    # which seeds the legacy global RNG and has no effect on default_rng(), so the seed
    # was silently ignored and runs were not reproducible.
    n_alleles = int(  case_ac_per_residue.sum()  )
    gen = np.random.default_rng(seed)
    null_ac_per_residue = gen.multivariate_hypergeometric(total_ac_per_residue.astype(int), n_alleles, n_sim).T
    return null_ac_per_residue

def get_case_control_ac_matrix(df, n_res, n_sim, seed=None, keep=None, uniprot_id=''):
    case_ac_per_residue = get_ac_per_residue(df, 'ac_case', n_res, uniprot_id)
    control_ac_per_residue = get_ac_per_residue(df, 'ac_control', n_res, uniprot_id)
    if keep is not None:
        # Zero before the totals and before the null draw, so n_case / n_control and the
        # multivariate hypergeometric both see only the residues still in play. Zeroing
        # afterwards would leave the null redistributing alleles into masked residues.
        case_ac_per_residue = case_ac_per_residue * keep[:, np.newaxis]
        control_ac_per_residue = control_ac_per_residue * keep[:, np.newaxis]
    total_ac_per_residue = (case_ac_per_residue + control_ac_per_residue).flatten()
    null_case_ac_per_residue = get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim, seed)
    null_control_ac_per_residue = total_ac_per_residue[:, np.newaxis] - null_case_ac_per_residue
    case_ac_matrix = np.hstack([case_ac_per_residue, null_case_ac_per_residue])
    control_ac_matrix = np.hstack([control_ac_per_residue, null_control_ac_per_residue])
    return case_ac_matrix, control_ac_matrix

def compute_all_pvals(
        df,
        pdb_file_pos_guide,
        pdb_dir,
        pae_dir,
        uniprot_id,
        n_sims,
        radius = 15,
        pae_cutoff = 15,
        seed = None,
        keep_residues = None,
        min_retained = DEFAULT_MIN_RETAINED,
        include_depletion = False,
):
    if radius in ('multiple-small', 'multiple-big'):
        radii = MULTI_RADII_SMALL if radius == 'multiple-small' else MULTI_RADII_BIG
        adj_matrices = {
            r: get_adjacency_matrix(pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, r, pae_cutoff)
            for r in radii
        }
        n_res = adj_matrices[radii[0]].shape[0]
        keep = check_keep_mask(keep_residues, n_res, uniprot_id)
        case_ac_matrix, control_ac_matrix = get_case_control_ac_matrix(df, n_res, n_sims, seed, keep, uniprot_id)
        n_case = case_ac_matrix[:,0].sum()
        n_control = control_ac_matrix[:,0].sum()

        pval_mats = []
        nbhd_counts = {}
        dead_any = np.zeros(n_res, dtype=bool) if keep is not None else None
        # Intersection across radii, for the log only -- see the comment below.
        dead_all = None
        for r in radii:
            pval_mat, n_case_nbhd, n_ctrl_nbhd = _compute_pval_matrix_for_radius(
                adj_matrices[r], case_ac_matrix, control_ac_matrix, n_case, n_control,
                include_depletion
            )
            # Neutralize per radius, before combining: a centre can keep enough of its
            # small neighborhoods while its large ones are mostly masked, and the
            # harmonic mean should then rest on the radii that survived.
            dead_r = neutralized_positions(adj_matrices[r], keep, min_retained)
            if dead_r is not None:
                pval_mat = _neutralize(pval_mat, dead_r)
                dead_any |= dead_r
                # Only a centre dead at every radius is actually neutralized in the
                # combined p-value; dead_any is the union, so reporting it would count
                # centres the harmonic mean still rests on. Tracked separately for the
                # log rather than narrowing dead_any, which drives nothing here.
                dead_all = dead_r if dead_all is None else (dead_all & dead_r)
            pval_mats.append(pval_mat)
            nbhd_counts[r] = (n_case_nbhd, n_ctrl_nbhd)

        stacked = np.stack(pval_mats, axis=0)  # (n_radii, n_res, n_sims+1)
        stacked = np.maximum(stacked, P_FLOOR)
        pval_matrix = len(radii) / np.sum(1.0 / stacked, axis=0)

        # Per-position best radius: the radius with the smallest observed p-value
        best_radius_idx = np.argmin(stacked[:, :, 0], axis=0)   # (n_res,)
        radius_col = np.array([radii[i] for i in best_radius_idx], dtype=float)

        # Build nbhd counts from each position's best radius
        case_nbhd_by_radius = np.stack([nbhd_counts[r][0][:, 0] for r in radii], axis=0)  # (n_radii, n_res)
        ctrl_nbhd_by_radius = np.stack([nbhd_counts[r][1][:, 0] for r in radii], axis=0)
        _pos_idx = np.arange(case_nbhd_by_radius.shape[1])
        n_case_nbhd_mat    = case_nbhd_by_radius[best_radius_idx, _pos_idx].reshape(-1, 1)
        n_control_nbhd_mat = ctrl_nbhd_by_radius[best_radius_idx, _pos_idx].reshape(-1, 1)

    else:
        adjacency_matrix = get_adjacency_matrix(
            pdb_file_pos_guide,
            pdb_dir,
            pae_dir,
            uniprot_id,
            radius,
            pae_cutoff,
        )
        n_res = adjacency_matrix.shape[0]
        keep = check_keep_mask(keep_residues, n_res, uniprot_id)
        case_ac_matrix, control_ac_matrix = get_case_control_ac_matrix(df, n_res, n_sims, seed, keep, uniprot_id)
        n_case = case_ac_matrix[:,0].sum()
        n_control = control_ac_matrix[:,0].sum()
        n_case_nbhd_mat = get_nbhd_counts(adjacency_matrix, case_ac_matrix)
        n_control_nbhd_mat = get_nbhd_counts(adjacency_matrix, control_ac_matrix)
        pval_lookup = get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_control,
                                                   include_depletion)
        pval_matrix = pval_lookup[n_case_nbhd_mat, n_control_nbhd_mat]
        dead_any = neutralized_positions(adjacency_matrix, keep, min_retained)
        dead_all = dead_any      # one radius: the union and the intersection coincide
        pval_matrix = _neutralize(pval_matrix, dead_any)
        radius_col = float(radius)

    pval_columns = ['p_value'] + [f'null_pval_{i}' for i in range(n_sims)]
    df_pvals = pd.DataFrame(columns = pval_columns, data = pval_matrix)
    df_pvals['nbhd_case'] = n_case_nbhd_mat[:,0]
    df_pvals['nbhd_control'] = n_control_nbhd_mat[:,0]
    df_pvals['radius'] = radius_col
    # Add original per-residue mutation counts
    df_pvals['original_case'] = case_ac_matrix[:,0]
    df_pvals['original_control'] = control_ac_matrix[:,0]
    df_pvals = df_pvals[['nbhd_case', 'nbhd_control', 'radius', 'original_case', 'original_control'] + pval_columns]
    if dead_any is not None:
        n_dead = int(dead_all.sum()) if dead_all is not None else int(dead_any.sum())
        logger.info(
            f'{uniprot_id}: conditioning masks {int((~keep).sum())} of {n_res} residues; '
            f'{n_dead} centres not re-tested'
        )
    return df_pvals

def write_df_pvals(results_dir, uniprot_id, df_pvals, pval_file):
    with h5py.File(os.path.join(results_dir, pval_file), 'a') as fid:
        null_pval_cols = [c for c in df_pvals.columns if c.startswith('null_pval')]
        write_dataset(fid, f'{uniprot_id}', df_pvals[['p_value']])
        write_dataset(fid, f'{uniprot_id}_null_pval', df_pvals[null_pval_cols])
        write_dataset(fid, f'{uniprot_id}_nbhd', df_pvals[['nbhd_case', 'nbhd_control']])
        write_dataset(fid, f'{uniprot_id}_radius', df_pvals[['radius']])
        write_dataset(fid, f'{uniprot_id}_original', df_pvals[['original_case', 'original_control']])

def scan_test_one_protein(df, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, uniprot_id, radius, pae_cutoff, n_sims, pval_file, seed=None,
                          keep_residues=None, min_retained=DEFAULT_MIN_RETAINED,
                          include_depletion=False):
    df_pvals = compute_all_pvals(
        df,
        pdb_file_pos_guide,
        pdb_dir,
        pae_dir,
        uniprot_id,
        n_sims,
        radius,
        pae_cutoff,
        seed,
        keep_residues,
        min_retained,
        include_depletion,
    )
    write_df_pvals(results_dir, uniprot_id, df_pvals, pval_file)

def _preprocess_scan_data(df_rvas, ignore_ac):
    """Preprocess scan data based on ignore_ac flag."""
    if not ignore_ac:
        return df_rvas
    
    logger.debug("Applying ignore_ac preprocessing")
    df_processed = df_rvas.copy()
    df_processed['ac_case'] = (df_processed['ac_case'] > 0).astype(int)
    df_processed['ac_control'] = (df_processed['ac_control'] > 0).astype(int)
    df_processed['to_drop'] = df_processed['ac_case'] + df_processed['ac_control'] > 1
    df_processed = df_processed[~df_processed.to_drop].copy()
    df_processed.drop('to_drop', axis=1, inplace=True)
    
    return df_processed


def _filter_proteins_by_allele_count(df_rvas, df_fdr_filter, min_alleles=DEFAULT_MIN_ALLELES):
    """
    Genes with enough case AND control alleles to be worth testing.

    Inclusive: min_alleles is the minimum, so the default of 5 admits a gene with
    exactly 5 of each. --min-denovo in the mutation-rate path means the same thing.
    """
    grouped = df_rvas.groupby('uniprot_id')[['ac_case', 'ac_control']].sum()
    ac_high_enough = grouped[(grouped['ac_case'] >= min_alleles) & (grouped['ac_control'] >= min_alleles)]
    uniprot_id_list = ac_high_enough.index.tolist()

    if df_fdr_filter is not None:
        uniprot_id_list = np.intersect1d(uniprot_id_list, np.unique(df_fdr_filter.uniprot_id))

    logger.info(f"Selected {len(uniprot_id_list)} proteins for analysis "
                f"(at least {min_alleles} case and control alleles each)")
    return uniprot_id_list


def _process_proteins_batch(df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff, results_dir, n_sims, conditioning, pval_file, seed=None,
                            min_retained=DEFAULT_MIN_RETAINED, seed_salt=None,
                            include_depletion=False, min_alleles=DEFAULT_MIN_ALLELES):
    """
    Process each protein individually with scan test.

    `conditioning` maps uniprot_id -> [(aa_pos, radius), ...]: the neighborhoods to
    condition out for that protein. Conditioning zeroes the affected residues rather
    than deleting variant rows, so the per-residue arrays keep their length and row i
    keeps meaning residue i+1 (see the indexing contract in utils.py).
    """
    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'
    conditioning = conditioning or {}

    n_proteins = len(uniprot_id_list)
    for i, uniprot_id in enumerate(uniprot_id_list):
        logger.info(f'Processing {uniprot_id} (protein {i+1} out of {n_proteins})')
        try:
            df = df_rvas[df_rvas.uniprot_id == uniprot_id]

            keep_residues = None
            entry = conditioning.get(uniprot_id)
            if entry:
                keep_residues, _, removed = build_conditioning_mask(
                    pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, entry, pae_cutoff
                )
                for key, n_removed in removed.items():
                    what = ('component' if key == 'residues'
                            else f'neighborhood of residue {key}')
                    logger.info(f'{uniprot_id}: conditioning out {what} '
                                f'({n_removed} residues)')

            # Count what the scan will actually see, which is neither the raw frame
            # nor the conditioned one alone. Two things remove alleles before the test:
            #
            #  - conditioning, which masks residues rather than dropping rows, so an
            #    unmasked count lets a gene whose conditioned-out component carried
            #    nearly all its alleles through on almost nothing; and
            #  - get_ac_per_residue, which drops aa_pos outside 1..n_res with a warning
            #    (it used to raise and lose the gene), so a gene mapped against a longer
            #    isoform than the guide's structure is scanned on only the part that
            #    fits.
            #
            # Either way n_case can reach 0, and read_p_values then divides by it.
            df_counted = df
            try:
                n_res_guide = structure_length(pdb_file_pos_guide, uniprot_id)
                df_counted = df_counted[(df_counted.aa_pos >= 1)
                                        & (df_counted.aa_pos <= n_res_guide)]
            except (ValueError, FileNotFoundError, KeyError) as e:
                # No structure length to check against: fall back to counting
                # everything, which is what this guard did before.
                logger.debug(f'{uniprot_id}: no structure length for the allele guard - {e}')
            if keep_residues is not None:
                kept_pos = np.flatnonzero(keep_residues) + 1
                df_counted = df_counted[df_counted.aa_pos.isin(kept_pos)]
            if (sum(df_counted.ac_case) < min_alleles) or (
                    sum(df_counted.ac_control) < min_alleles):
                extra = '' if keep_residues is None else ' after conditioning'
                if len(df_counted) < len(df):
                    extra += (f' ({len(df) - len(df_counted)} of {len(df)} rows fall '
                              'outside the structure or the mask)')
                logger.warning(f'{uniprot_id}: needs at least {min_alleles} case and '
                               f'{min_alleles} control alleles{extra}. Skipping.')
                continue
            
            # Each protein gets its own derived stream. A single shared seed would
            # correlate the null draws across proteins, which matters because FWER takes
            # the minimum p-value across proteins within each simulation.
            protein_seed = None if seed is None else [seed, i]
            if protein_seed is not None and seed_salt is not None:
                # Conditional rounds re-scan the same protein: without a per-round salt
                # every round would replay the identical null draws.
                protein_seed = protein_seed + [seed_salt]
            scan_test_one_protein(
                df, pdb_file_pos_guide, pdb_dir, pae_dir, 
                results_dir, uniprot_id, radius, pae_cutoff, n_sims,
                pval_file, protein_seed, keep_residues, min_retained,
                include_depletion
            )
        except FileNotFoundError as e:
            logger.error(f'{uniprot_id}: Required file not found - {e}')
            continue
        except KeyError as e:
            logger.error(f'{uniprot_id}: Missing required column or key - {e}')
            continue
        except ValueError as e:
            logger.error(f'{uniprot_id}: Invalid data or parameter - {e}')
            continue
        except MemoryError as e:
            logger.error(f'{uniprot_id}: Insufficient memory for processing - {e}')
            continue
        except Exception as e:
            logger.error(f'{uniprot_id}: Unexpected error - {e}')
            continue


def scan_test(
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
    ignore_ac,
    fdr_file,
    pval_file,
    conditioning=None,
    seed=None,
    min_retained=DEFAULT_MIN_RETAINED,
    seed_salt=None,
    uniprot_id_list=None,
    include_depletion=False,
    min_alleles=DEFAULT_MIN_ALLELES,
):
    """
    Perform scan test analysis on protein structure data.
    
    Main orchestration function for the structure-informed rare variant association study.
    Processes variants across proteins and computes statistical associations with 3D neighborhoods.
    """
    
    # Handle FDR-only mode
    if fdr_only:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file)
        df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
        return

    logger.info("Starting scan test analysis")
    logger.info(f"Input dataset contains {len(df_rvas)} variants across {df_rvas['uniprot_id'].nunique()} proteins")

    # Preprocess data
    df_processed = _preprocess_scan_data(df_rvas, ignore_ac)
    
    # Filter proteins by allele count. A conditional round passes its own list so that
    # only the proteins being re-scanned are touched.
    if uniprot_id_list is None:
        uniprot_id_list = _filter_proteins_by_allele_count(df_processed, df_fdr_filter,
                                                          min_alleles)
    
    # Process each protein
    _process_proteins_batch(
        df_processed, uniprot_id_list, reference_dir, 
        radius, pae_cutoff, results_dir, n_sims, conditioning,
        pval_file, seed, min_retained, seed_salt, include_depletion, min_alleles
    )

    with h5py.File(os.path.join(results_dir, pval_file), 'a') as fid:
        # Record how this file was made. resolve_pval_node keys off test_type, and
        # include_depletion is not recoverable from the p-values themselves -- without it
        # an enrichment-only file is indistinguishable from a two-sided one afterwards.
        fid.attrs['test_type'] = 'case_control'
        fid.attrs['n_sims'] = n_sims
        fid.attrs['include_depletion'] = bool(include_depletion)
        fid.attrs['min_alleles'] = int(min_alleles)
        if seed is not None:
            fid.attrs['seed'] = seed
    
    # Compute FDR if requested
    if not no_fdr:
        # If no explicit filter was given, restrict FDR to the proteins just processed
        # so that proteins from previous runs with different n_sims don't cause errors
        fdr_filter = df_fdr_filter
        if fdr_filter is None:
            fdr_filter = pd.DataFrame({'uniprot_id': uniprot_id_list})
        df_results = compute_fdr(results_dir, fdr_cutoff, fdr_filter, reference_dir, pval_file)
        df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
    
