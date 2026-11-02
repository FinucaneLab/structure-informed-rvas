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
                   get_case_control_ac_matrix, restrict_to_structure)
from logger_config import get_logger
from empirical_fdr import compute_fdr

logger = get_logger(__name__)

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
        raise ValueError(
            f'only {n_case} case and {n_control} control alleles fall inside the '
            f'structure (1..{n_res}), below --min-alleles {min_alleles}'
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
):
    if radius in ('multiple-small', 'multiple-big'):
        radii = MULTI_RADII_SMALL if radius == 'multiple-small' else MULTI_RADII_BIG
        adj_matrices = {
            r: get_adjacency_matrix(pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, r, pae_cutoff)
            for r in radii
        }
        n_res = adj_matrices[radii[0]].shape[0]
        df = _restrict_and_recheck(df, n_res, uniprot_id, out_of_structure, min_alleles)
        case_ac_matrix, control_ac_matrix = get_case_control_ac_matrix(df, n_res, n_sims, seed)
        n_case = case_ac_matrix[:,0].sum()
        n_control = control_ac_matrix[:,0].sum()

        pval_mats = []
        nbhd_counts = {}
        for r in radii:
            pval_mat, n_case_nbhd, n_ctrl_nbhd = _compute_pval_matrix_for_radius(
                adj_matrices[r], case_ac_matrix, control_ac_matrix, n_case, n_control, alternative
            )
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
        case_ac_matrix, control_ac_matrix = get_case_control_ac_matrix(df, n_res, n_sims, seed)
        n_case = case_ac_matrix[:,0].sum()
        n_control = control_ac_matrix[:,0].sum()
        n_case_nbhd_mat = get_nbhd_counts(adjacency_matrix, case_ac_matrix)
        n_control_nbhd_mat = get_nbhd_counts(adjacency_matrix, control_ac_matrix)
        pval_lookup = get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_control, alternative)
        pval_matrix = pval_lookup[n_case_nbhd_mat, n_control_nbhd_mat]
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
    return df_pvals, adjacency_matrix

def write_df_pvals(results_dir, uniprot_id, df_pvals, pval_file):
    with h5py.File(os.path.join(results_dir, pval_file), 'a') as fid:
        null_pval_cols = [c for c in df_pvals.columns if c.startswith('null_pval')]
        write_dataset(fid, f'{uniprot_id}', df_pvals[['p_value']])
        write_dataset(fid, f'{uniprot_id}_null_pval', df_pvals[null_pval_cols])
        write_dataset(fid, f'{uniprot_id}_nbhd', df_pvals[['nbhd_case', 'nbhd_control']])
        write_dataset(fid, f'{uniprot_id}_radius', df_pvals[['radius']])
        write_dataset(fid, f'{uniprot_id}_original', df_pvals[['original_case', 'original_control']])

def scan_test_one_protein(df, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, uniprot_id, radius, pae_cutoff, n_sims, pval_file, seed=None, alternative='greater', out_of_structure=None, min_alleles=DEFAULT_MIN_ALLELES):
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
    
    if df_fdr_filter is not None:
        uniprot_id_list = np.intersect1d(uniprot_id_list, np.unique(df_fdr_filter.uniprot_id))
    
    logger.info(
        f"Selected {len(uniprot_id_list)} proteins for analysis "
        f"(at least {min_alleles} case and control alleles each)"
    )
    return uniprot_id_list


def _process_proteins_batch(df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff, results_dir, n_sims, remove_nbhd, pval_file, seed=None, alternative='greater', min_alleles=DEFAULT_MIN_ALLELES):
    """Process each protein individually with scan test.

    Returns the list of (uniprot_id, n_res, dropped_positions) for proteins whose
    variants fell outside their structure, for the caller to report at the end.
    """
    out_of_structure = []
    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'
    
    n_proteins = len(uniprot_id_list)
    for i, uniprot_id in enumerate(uniprot_id_list):
        logger.info(f'Processing {uniprot_id} (protein {i+1} out of {n_proteins})')
        try:
            df = df_rvas[df_rvas.uniprot_id == uniprot_id]
            if remove_nbhd is not None:
                    adj_radius = 15 if radius == 'multiple' else radius
                    adjacency_matrix = get_adjacency_matrix(
                        pdb_file_pos_guide,
                        pdb_dir,
                        pae_dir,
                        uniprot_id,
                        adj_radius,
                        pae_cutoff,
                    )
                    for to_remove in map(int, remove_nbhd.split(',')):
                        logger.info(f'Removing neighborhood of position {to_remove} for {uniprot_id}')
                        nbhd = set(np.where(adjacency_matrix[to_remove-1] == 1)[0] + 1)
                        df.drop(df[df['aa_pos'].isin(nbhd)].index, inplace=True)
                    df.reset_index(drop=True, inplace=True)

            if (sum(df.ac_case) < min_alleles) or (sum(df.ac_control) < min_alleles):
                logger.warning(
                    f'{uniprot_id}: There must be at least {min_alleles} case and '
                    f'{min_alleles} control alleles. Skipping.'
                )
                continue
            
            # Each protein gets its own derived stream. A single shared seed would
            # correlate the null draws across proteins, which matters because FWER takes
            # the minimum p-value across proteins within each simulation.
            protein_seed = None if seed is None else [seed, i]
            scan_test_one_protein(
                df, pdb_file_pos_guide, pdb_dir, pae_dir, 
                results_dir, uniprot_id, radius, pae_cutoff, n_sims,
                pval_file, protein_seed, alternative, out_of_structure, min_alleles
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

    return out_of_structure


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
    remove_nbhd,
    seed=None,
    alternative='greater',
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
    
    # Filter proteins by allele count
    uniprot_id_list = _filter_proteins_by_allele_count(df_processed, df_fdr_filter, min_alleles)
    
    # Process each protein
    out_of_structure = _process_proteins_batch(
        df_processed, uniprot_id_list, reference_dir, 
        radius, pae_cutoff, results_dir, n_sims, remove_nbhd,
        pval_file, seed, alternative, min_alleles
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
            fdr_filter = df_fdr_filter
            if fdr_filter is None:
                fdr_filter = pd.DataFrame({'uniprot_id': uniprot_id_list})
            df_results = compute_fdr(results_dir, fdr_cutoff, fdr_filter, reference_dir, pval_file)
            df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
    finally:
        _report_out_of_structure(out_of_structure)

