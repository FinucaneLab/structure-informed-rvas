import os
import pandas as pd
import numpy as np
import glob
import bisect
import h5py
from scipy.stats import fisher_exact, binom
from scipy import special
from utils import (DEFAULT_MIN_ALLELES, DEFAULT_SCREEN_CHI2, SCREEN_CHI2_GREATER, fisher_greater_pvals,
                   get_adjacency_matrix, valid_for_fisher, write_dataset, read_p_values,
                   get_case_control_ac_matrix, restrict_to_structure,
                   neighborhood_retention, check_keep_mask,
                   build_conditioning_mask)
from logger_config import get_logger
from empirical_fdr import compute_fdr
from value_tracking import (GeneSkipped, log_counts, record_neutralized, record_skip,
                            skip_reason, stats_fields)

logger = get_logger(__name__)

# A conditioned centre is not re-tested when its neighborhood has lost more than
# half of itself: what is left is a different neighborhood from the one the scan
# reported, and testing it would quietly change what the p-value refers to.
DEFAULT_MIN_RETAINED = 0.5

MULTI_RADII_SMALL = [6, 9, 12, 15, 18, 21]
MULTI_RADII_BIG = [10, 15, 20, 30, 40]
P_FLOOR = np.finfo(float).tiny

def get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_ctrl, alternative='greater'):
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
    
    if alternative == 'greater':
        potential_significant = (chi2 > SCREEN_CHI2_GREATER) & (a * n_ctrl > b * n_case) & valid_mask
        # only (a, b) pairs actually observed need the exact test
        observed = np.zeros_like(potential_significant)
        observed[n_case_nbhd_mat.astype(int).ravel(), n_control_nbhd_mat.astype(int).ravel()] = True
        sel = potential_significant & observed
        pvals[sel] = fisher_greater_pvals(a[sel], b[sel], n_case, n_ctrl)
        return pvals

    potential_significant = (chi2 > DEFAULT_SCREEN_CHI2) & valid_mask
    
    # Second pass: Calculate exact p-values only for potentially significant cells
    indices = np.where(potential_significant)
    for i, j in zip(indices[0], indices[1]):
        table = np.array([[i, j], [n_case - i, n_ctrl - j]])
        _, p = fisher_exact(table)
        pvals[i, j] = p
    
    return pvals

def _compute_pval_matrix_for_radius(adjacency_matrix, case_ac_matrix, control_ac_matrix, n_case, n_control, alternative='greater'):
    n_case_nbhd = get_nbhd_counts(adjacency_matrix, case_ac_matrix)
    n_control_nbhd = get_nbhd_counts(adjacency_matrix, control_ac_matrix)
    pval_lookup = get_pval_lookup_case_control(n_case_nbhd, n_control_nbhd, n_case, n_control, alternative)
    return pval_lookup[n_case_nbhd, n_control_nbhd], n_case_nbhd, n_control_nbhd

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
    # retention <= 0 is checked separately so that a min_retained of 0 still cannot
    # test a neighborhood with nothing left in it. Under the mutation-rate null such
    # a centre has expected count 0, which would hand any stray count an arbitrarily
    # small p-value. min_retained is a parameter, not yet a CLI flag.
    return (~keep) | (retention < min_retained) | (retention <= 0)


def _blank_counts(arr, dead):
    """Zero the reported counts at centres that were not re-tested.

    A dead centre's p-value is forced to 1.0, but its neighborhood counts and
    expectations would otherwise still describe the SURVIVING fragment of that
    neighborhood -- a real measurement, but not of the neighborhood the scan
    reported, and one nobody tested. Left in place they can outrank the hits that
    were tested: a fragment holding de novos next to a constrained residue can show
    obs/exp 4.0 beside p = 1.0, above a genuine 2.8. Zeroed, a dead row reads as
    nothing in every column.

    Rows are never removed; the row index carries aa_pos.
    """
    if dead is None or not dead.any():
        return arr
    out = np.array(arr, dtype=arr.dtype, copy=True)
    out[dead, ...] = 0
    return out


def _neutralize(pval_matrix, dead):
    """Set every column, observed and null alike, to 1.0 at the dead centres.

    Rows are never removed: the row index carries aa_pos (see the indexing contract in
    utils.py), and empirical_fdr rebuilds positions from row order."""
    if dead is None or not dead.any():
        return pval_matrix
    out = np.array(pval_matrix, dtype=float, copy=True)
    out[dead, :] = 1.0
    return out

def get_nbhd_counts(adjacency_matrix, ac_per_residue):
    # allele count per residue has one row per residue and one
    # column per simulation
    nbhd_counts = adjacency_matrix @ ac_per_residue
    nbhd_counts = nbhd_counts.astype(int)
    return nbhd_counts

def _restrict_and_recheck(df, n_res, uniprot_id, out_of_structure, min_alleles):
    """Drop positions outside the structure, then re-apply the allele floor.

    The floor was checked against the mapped variants, before anything knew how
    long the structure was. Dropping out-of-range positions can take a gene below
    it, or empty it outright -- a gene whose variants all sit past the end of a
    fragment structure clears the filter on counts the scan can never see.

    Without the re-check that gene is scanned with n_case = 0: every p-value comes
    back 1.0, the table is written anyway, and read_p_values then divides by zero
    and fills the ratio column with nan. That is the failure --min-alleles is
    bounded below 1 to prevent, and it would have been reintroduced here.

    Raising rather than returning a flag: _process_proteins_batch already catches
    ValueError per protein, logs it and moves on, which is the behaviour this had
    before the restriction existed (an out-of-range index raised IndexError there).
    """
    df, dropped = restrict_to_structure(df, n_res, uniprot_id)
    _record_out_of_structure(out_of_structure, uniprot_id, n_res, dropped)
    if not dropped:
        return df
    n_case, n_control = int(sum(df.ac_case)), int(sum(df.ac_control))
    if n_case < min_alleles or n_control < min_alleles:
        raise GeneSkipped(
            f'only {n_case} case and {n_control} control alleles fall inside the '
            f'structure (1..{n_res}), below --min-alleles {min_alleles}',
            'too_few_alleles_in_structure',
        )
    return df


def _record_out_of_structure(collector, uniprot_id, n_res, dropped):
    """Note a protein whose variants fell outside its structure, for the end-of-run report.

    Collected rather than only logged in place: restrict_to_structure warns as it
    happens, but that line lands in the middle of one protein's output among
    thousands and is easy to miss. The summary at the end of scan_test names every
    affected protein in one place.
    """
    if collector is not None and dropped:
        collector.append((uniprot_id, n_res, dropped))


def _refuse_if_nothing_testable(adj_by_radius, keep, min_retained, uniprot_id, n_res):
    """Raise when conditioning has left no testable neighborhood at any radius.

    Checked before any p-value work: writing the table anyway puts n_res rows of
    p = 1.0 into the FDR and FWER pools as tested hypotheses, inflating the
    multiple-testing burden for the genes that do still have signal -- the same
    argument the mutation-rate path makes for its post-conditioning de novo gate.
    Raised rather than returned, so _process_proteins_batch skips the gene and it
    stays out of `scanned`, which is what keeps its stale rows out of the results.

    Early also avoids computing a full table from all-zero counts, where the
    Fisher expectation divides by a zero total.
    """
    if keep is None:
        return
    dead = None
    for adj in adj_by_radius:
        d = neutralized_positions(adj, keep, min_retained)
        dead = d if dead is None else (dead & d)
    if dead is not None and dead.all():
        raise GeneSkipped(
            f'conditioning left no testable neighborhood in {uniprot_id}: all '
            f'{n_res} centres lost more than their share of residues',
            'nothing_testable_after_conditioning',
        )


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
        alternative = 'greater',
        out_of_structure = None,
        min_alleles = DEFAULT_MIN_ALLELES,
        keep_residues = None,
        min_retained = DEFAULT_MIN_RETAINED,
        stats = None,
):
    if radius in ('multiple-small', 'multiple-big'):
        radii = MULTI_RADII_SMALL if radius == 'multiple-small' else MULTI_RADII_BIG
        adj_matrices = {
            r: get_adjacency_matrix(pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, r, pae_cutoff)
            for r in radii
        }
        n_res = adj_matrices[radii[0]].shape[0]
        df = _restrict_and_recheck(df, n_res, uniprot_id, out_of_structure, min_alleles)
        keep = check_keep_mask(keep_residues, n_res, uniprot_id)
        _refuse_if_nothing_testable([adj_matrices[r] for r in radii], keep, min_retained, uniprot_id, n_res)
        case_ac_matrix, control_ac_matrix = get_case_control_ac_matrix(
            df, n_res, n_sims, seed, keep)
        n_case = case_ac_matrix[:,0].sum()
        n_control = control_ac_matrix[:,0].sum()

        pval_mats = []
        dead_all = None
        nbhd_counts = {}
        for r in radii:
            pval_mat, n_case_nbhd, n_ctrl_nbhd = _compute_pval_matrix_for_radius(
                adj_matrices[r], case_ac_matrix, control_ac_matrix, n_case, n_control, alternative
            )
            if keep is not None:
                # Per radius, before combining: a centre can keep enough of its
                # neighborhood at one radius and lose it at another, and the rule
                # reads only the structure and the mask, never the counts, so the
                # observed and null columns are neutralized identically.
                dead_r = neutralized_positions(adj_matrices[r], keep, min_retained)
                pval_mat = _neutralize(pval_mat, dead_r)
                # Only the intersection is reported: a centre is neutralized in the
                # combined p-value only when it is dead at every radius, since the
                # harmonic mean is allowed to rest on the surviving ones.
                dead_all = dead_r if dead_all is None else (dead_all & dead_r)
                n_case_nbhd = _blank_counts(n_case_nbhd, dead_r)
                n_ctrl_nbhd = _blank_counts(n_ctrl_nbhd, dead_r)
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

        adjacency_matrix = adj_matrices[15]  # used only for --remove-nbhd; radius 15 is a reasonable default
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
        df = _restrict_and_recheck(df, n_res, uniprot_id, out_of_structure, min_alleles)
        keep = check_keep_mask(keep_residues, n_res, uniprot_id)
        _refuse_if_nothing_testable([adjacency_matrix], keep, min_retained, uniprot_id, n_res)
        case_ac_matrix, control_ac_matrix = get_case_control_ac_matrix(
            df, n_res, n_sims, seed, keep)
        n_case = case_ac_matrix[:,0].sum()
        n_control = control_ac_matrix[:,0].sum()
        n_case_nbhd_mat = get_nbhd_counts(adjacency_matrix, case_ac_matrix)
        n_control_nbhd_mat = get_nbhd_counts(adjacency_matrix, control_ac_matrix)
        pval_lookup = get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_control, alternative)
        pval_matrix = pval_lookup[n_case_nbhd_mat, n_control_nbhd_mat]
        # One radius, so the union and the intersection coincide.
        dead_all = neutralized_positions(adjacency_matrix, keep, min_retained)
        pval_matrix = _neutralize(pval_matrix, dead_all)
        n_case_nbhd_mat = _blank_counts(n_case_nbhd_mat, dead_all)
        n_control_nbhd_mat = _blank_counts(n_control_nbhd_mat, dead_all)
        radius_col = float(radius)

    if keep is not None and dead_all is not None:
        # Only a centre dead at EVERY radius is neutralized in the combined p-value --
        # the harmonic mean is deliberately allowed to rest on the surviving radii --
        # so the count reported is the intersection, not the union.
        logger.info(
            f'{uniprot_id}: conditioning masks {int((~keep).sum())} of {n_res} '
            f'residues; {int(dead_all.sum())} centres not re-tested'
        )
        record_neutralized(stats, uniprot_id, dead_all.sum())

    pval_columns = ['p_value'] + [f'null_pval_{i}' for i in range(n_sims)]
    df_pvals = pd.DataFrame(columns = pval_columns, data = pval_matrix)
    df_pvals['nbhd_case'] = n_case_nbhd_mat[:,0]
    df_pvals['nbhd_control'] = n_control_nbhd_mat[:,0]
    df_pvals['radius'] = radius_col
    # Add original per-residue mutation counts
    df_pvals['original_case'] = case_ac_matrix[:,0]
    df_pvals['original_control'] = control_ac_matrix[:,0]
    df_pvals = df_pvals[['nbhd_case', 'nbhd_control', 'radius', 'original_case', 'original_control'] + pval_columns]
    return df_pvals, adjacency_matrix

def write_df_pvals(results_dir, uniprot_id, df_pvals, pval_file):
    with h5py.File(os.path.join(results_dir, pval_file), 'a') as fid:
        null_pval_cols = [c for c in df_pvals.columns if c.startswith('null_pval')]
        write_dataset(fid, f'{uniprot_id}', df_pvals[['p_value']])
        write_dataset(fid, f'{uniprot_id}_null_pval', df_pvals[null_pval_cols])
        write_dataset(fid, f'{uniprot_id}_nbhd', df_pvals[['nbhd_case', 'nbhd_control']])
        write_dataset(fid, f'{uniprot_id}_radius', df_pvals[['radius']])
        write_dataset(fid, f'{uniprot_id}_original', df_pvals[['original_case', 'original_control']])

def scan_test_one_protein(df, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir,
                          uniprot_id, radius, pae_cutoff, n_sims, pval_file, seed=None,
                          alternative='greater', out_of_structure=None,
                          min_alleles=DEFAULT_MIN_ALLELES, keep_residues=None,
                          min_retained=DEFAULT_MIN_RETAINED, stats=None):
    df_pvals, adj_mat = compute_all_pvals(
        df,
        pdb_file_pos_guide,
        pdb_dir,
        pae_dir,
        uniprot_id,
        n_sims,
        radius,
        pae_cutoff,
        seed,
        alternative,
        out_of_structure,
        min_alleles,
        keep_residues,
        min_retained,
        stats,
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
    """Filter proteins to include only those with sufficient case and control alleles.

    Applied inclusively, so the default of 5 keeps a gene with 5 case and 5 control
    alleles. _process_proteins_batch applies the same rule to whatever is left
    after any conditioning.
    """
    grouped = df_rvas.groupby('uniprot_id')[['ac_case', 'ac_control']].sum()
    ac_high_enough = grouped[(grouped['ac_case'] >= min_alleles) & (grouped['ac_control'] >= min_alleles)]
    uniprot_id_list = ac_high_enough.index.tolist()
    log_counts(f'Proteins with >= {min_alleles} case and control alleles',
               df_rvas[df_rvas.uniprot_id.isin(uniprot_id_list)])

    if df_fdr_filter is not None:
        uniprot_id_list = np.intersect1d(uniprot_id_list, np.unique(df_fdr_filter.uniprot_id))
        log_counts('Restrict to --df-filter proteins',
                   df_rvas[df_rvas.uniprot_id.isin(uniprot_id_list)])
    
    logger.info(
        f"Selected {len(uniprot_id_list)} proteins for analysis "
        f"(at least {min_alleles} case and control alleles each)"
    )
    return uniprot_id_list


def _process_proteins_batch(df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff,
                            results_dir, n_sims, conditioning, pval_file, seed=None,
                            alternative='greater', min_alleles=DEFAULT_MIN_ALLELES,
                            min_retained=DEFAULT_MIN_RETAINED, seed_salt=None, stats=None):
    """Process each protein individually with scan test.

    Returns the list of (uniprot_id, n_res, dropped_positions) for proteins whose
    variants fell outside their structure, for the caller to report at the end.
    `stats`, if a dict, collects why genes were skipped and how many centres
    conditioning neutralized (see value_tracking).
    """
    out_of_structure = []
    # Which genes actually got written. A gene can be skipped here for several
    # reasons, and a caller that overwrites a previous p-value file needs to know
    # which entries are stale rather than assuming every requested gene was redone.
    scanned = set()
    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'
    
    n_proteins = len(uniprot_id_list)
    for i, uniprot_id in enumerate(uniprot_id_list):
        logger.info(f'Processing {uniprot_id} (protein {i+1} out of {n_proteins})')
        try:
            df = df_rvas[df_rvas.uniprot_id == uniprot_id]
            keep_residues = None
            entry = conditioning.get(uniprot_id) if conditioning else None
            if entry:
                # Masked, not deleted. Every per-residue array in the scan is
                # positional -- entry i is residue i + 1 -- so dropping rows would
                # shift every downstream index. See the indexing contract in utils.
                keep_residues, _, removed = build_conditioning_mask(
                    pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, entry, pae_cutoff)
                logger.info(f'{uniprot_id}: conditioning removes '
                            f'{sum(removed.values())} residues {removed}')

            # Counted on what conditioning LEFT. The old code dropped the conditioned
            # rows before this point, so the gate was post-conditioning for free;
            # masking happens later, inside compute_all_pvals, so the gate has to
            # apply the mask itself. Without this a gene whose case alleles all sit
            # inside the masked region clears the floor and is scanned with n_case = 0,
            # and read_p_values then divides by that count -- the failure --min-alleles
            # is bounded below 1 to prevent.
            df_effective = df
            if keep_residues is not None:
                kept_pos = np.flatnonzero(np.asarray(keep_residues).ravel()) + 1
                df_effective = df[df.aa_pos.isin(kept_pos)]
            if (sum(df_effective.ac_case) < min_alleles
                    or sum(df_effective.ac_control) < min_alleles):
                logger.warning(
                    f'{uniprot_id}: There must be at least {min_alleles} case and '
                    f'{min_alleles} control alleles'
                    + (' after conditioning' if keep_residues is not None else '')
                    + f'. Skipping. (has {int(sum(df_effective.ac_case))} case, '
                    f'{int(sum(df_effective.ac_control))} control)'
                )
                record_skip(stats, 'too_few_alleles_after_conditioning', uniprot_id)
                continue
            
            # Each protein gets its own derived stream. A single shared seed would
            # correlate the null draws across proteins, which matters because FWER takes
            # the minimum p-value across proteins within each simulation.
            protein_seed = None if seed is None else [seed, i]
            if protein_seed is not None and seed_salt is not None:
                # Conditional rounds re-scan the same protein. Without a per-round
                # salt every round would replay the identical null draws, so a
                # neighborhood's p-value would be compared against the same simulated
                # data it was compared against before conditioning.
                protein_seed = protein_seed + [seed_salt]
            scan_test_one_protein(
                df, pdb_file_pos_guide, pdb_dir, pae_dir, 
                results_dir, uniprot_id, radius, pae_cutoff, n_sims,
                pval_file, protein_seed, alternative, out_of_structure, min_alleles,
                keep_residues, min_retained, stats
            )
            scanned.add(uniprot_id)
        except FileNotFoundError as e:
            logger.error(f'{uniprot_id}: Required file not found - {e}')
            record_skip(stats, skip_reason(e), uniprot_id)
            continue
        except KeyError as e:
            logger.error(f'{uniprot_id}: Missing required column or key - {e}')
            record_skip(stats, skip_reason(e), uniprot_id)
            continue
        except ValueError as e:
            logger.error(f'{uniprot_id}: Invalid data or parameter - {e}')
            record_skip(stats, skip_reason(e), uniprot_id)
            continue
        except MemoryError as e:
            logger.error(f'{uniprot_id}: Insufficient memory for processing - {e}')
            record_skip(stats, skip_reason(e), uniprot_id)
            continue
        except Exception as e:
            logger.error(f'{uniprot_id}: Unexpected error - {e}')
            record_skip(stats, skip_reason(e), uniprot_id)
            continue

    return out_of_structure, scanned


def _without_out_of_structure(df, out_of_structure):
    """df minus the variants restrict_to_structure dropped, for value tracking."""
    dropped = {(u, int(p)) for u, _, positions in out_of_structure for p in positions}
    if not dropped:
        return df
    return df[[(u, int(p)) not in dropped for u, p in zip(df.uniprot_id, df.aa_pos)]]


def _report_out_of_structure(out_of_structure):
    """Name every protein whose variants fell outside its structure, at the end of the run.

    restrict_to_structure already warns as each one happens, but that line lands in
    the middle of one protein's output among thousands and is easy to scroll past.
    This repeats them together as the last thing printed, so the run cannot finish
    without the reader being told which proteins were affected.

    A position outside 1..n_res means the variant was mapped against a UniProt
    isoform longer than the structure in the guide, or against a protein whose
    guide entry is a fragment. Those alleles are silently absent from the scan, so
    a gene can clear the allele filter on counts that the scan never sees.
    """
    if not out_of_structure:
        return
    total = sum(len(dropped) for _, _, dropped in out_of_structure)
    logger.warning(
        f'{len(out_of_structure)} protein(s) had {total} position(s) outside their '
        f'structure; those variants were dropped from the scan. The allele filters '
        f'count them, so these genes were scanned on fewer alleles than they passed on.'
    )
    for uniprot_id, n_res, dropped in sorted(out_of_structure):
        shown = ', '.join(str(d) for d in dropped[:10])
        more = f' (and {len(dropped) - 10} more)' if len(dropped) > 10 else ''
        logger.warning(f'  {uniprot_id}: structure is 1..{n_res}; dropped {shown}{more}')


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
    conditioning,
    seed=None,
    alternative='greater',
    min_alleles=DEFAULT_MIN_ALLELES,
    min_retained=DEFAULT_MIN_RETAINED,
    requested=None,
    seed_salt=None,
    stats=None,
):
    """
    Perform scan test analysis on protein structure data.

    `stats`: optional dict the scan fills with why genes were skipped and how many
    centres conditioning neutralized -- for a caller (the conditional rounds) that
    needs those after the scan's own count lines have been muted.

    `conditioning` maps uniprot_id to what should be masked out of that gene before
    it is re-scanned -- either [(aa_pos, radius), ...] or an explicit residue list.
    Keyed by gene, because a single --remove-nbhd position applied to every protein
    in the file is almost never what anyone means.

    `requested` restricts the scan to those genes; the rest of the p-value
    file is left alone. Returns the set of genes actually written.
    
    Main orchestration function for the structure-informed rare variant association study.
    Processes variants across proteins and computes statistical associations with 3D neighborhoods.
    """
    
    stats = {} if stats is None else stats

    # Handle FDR-only mode
    if fdr_only:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file)
        df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
        return set()

    logger.info("Starting scan test analysis")
    logger.info(f"Input dataset contains {len(df_rvas)} variants across {df_rvas['uniprot_id'].nunique()} proteins")

    # Preprocess data
    df_processed = _preprocess_scan_data(df_rvas, ignore_ac)
    if ignore_ac:
        log_counts('Remove variants with more than one allele (--ignore-ac)', df_processed)

    # Filter proteins by allele count
    uniprot_id_list = _filter_proteins_by_allele_count(df_processed, df_fdr_filter, min_alleles)
    if requested is not None:
        uniprot_id_list = [u for u in uniprot_id_list if u in set(requested)]
        log_counts('Restrict to requested genes',
                   df_processed[df_processed.uniprot_id.isin(uniprot_id_list)])

    # Process each protein
    out_of_structure, scanned = _process_proteins_batch(
        df_processed, uniprot_id_list, reference_dir,
        radius, pae_cutoff, results_dir, n_sims, conditioning,
        pval_file, seed, alternative, min_alleles, min_retained, seed_salt,
        stats=stats,
    )

    # Variants outside their structure were dropped from the scan, so they are not
    # counted as scanned either.
    df_scanned = _without_out_of_structure(
        df_processed[df_processed.uniprot_id.isin(scanned)], out_of_structure)
    log_counts('Scanned (skipped genes and out-of-structure variants removed)', df_scanned,
               **stats_fields(stats, scanned))

    skipped = [u for u in uniprot_id_list if u not in scanned]
    if skipped:
        logger.warning(
            f'{len(skipped)} of {len(uniprot_id_list)} genes were not scanned and are '
            f'excluded from the results: {skipped[:10]}'
            + (' ...' if len(skipped) > 10 else '')
            + '. Any p-values they already had in the file are left untouched and are '
              'NOT reported, since they describe a different analysis.'
        )

    with h5py.File(os.path.join(results_dir, pval_file), 'a') as fid:
        fid.attrs['n_sims'] = n_sims
        if seed is not None:
            fid.attrs['seed'] = seed
    
    # Compute FDR if requested. The out-of-structure summary runs either way: an
    # FDR failure is precisely when the reader most needs to know the protein set
    # was not what they thought, so it must not be skipped by the exception.
    try:
        if not no_fdr:
            # If no explicit filter was given, restrict FDR to the proteins just processed
            # so that proteins from previous runs with different n_sims don't cause errors
            # Restricted to the genes actually written, whether or not a filter was
            # supplied. A skipped gene keeps its PREVIOUS datasets in the p-value
            # file, and conditioning makes skips common -- the post-mask allele gate,
            # a mask length mismatch, an out-of-range centre. Reporting on a gene
            # this run did not write loads those stale, unconditioned values and
            # presents a conditioned-out hit at full significance, as if the re-scan
            # had confirmed it. An explicit --df-filter has to be intersected rather
            # than taken whole, or it reintroduces exactly that.
            if df_fdr_filter is None:
                fdr_filter = pd.DataFrame({'uniprot_id': sorted(scanned)})
            else:
                fdr_filter = df_fdr_filter[df_fdr_filter.uniprot_id.isin(scanned)]
            df_results = compute_fdr(results_dir, fdr_cutoff, fdr_filter, reference_dir, pval_file)
            df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
            log_counts('Neighborhoods tested (FDR/FWER pool)',
                       df_scanned[df_scanned.uniprot_id.isin(df_results.uniprot_id)],
                       n_nbhds=len(df_results))
    finally:
        _report_out_of_structure(out_of_structure)

    return scanned

