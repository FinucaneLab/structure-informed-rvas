import os
import pandas as pd
import numpy as np
import glob
import bisect
import h5py
from scipy.stats import fisher_exact, binom
from scipy import special
from utils import get_adjacency_matrix, valid_for_fisher, write_dataset, read_p_values
from logger_config import get_logger
from empirical_fdr import compute_fdr

logger = get_logger(__name__)

def get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_ctrl):
    # from Claude
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
    
    # Second pass: Calculate exact p-values only for potentially significant cells
    indices = np.where(potential_significant)
    for i, j in zip(indices[0], indices[1]):
        table = np.array([[i, j], [n_case - i, n_ctrl - j]])
        _, p = fisher_exact(table)
        pvals[i, j] = p
    
    return pvals

def get_nbhd_counts(adjacency_matrix, ac_per_residue):
    # allele count per residue has one row per residue and one
    # column per simulation
    nbhd_counts = adjacency_matrix @ ac_per_residue
    nbhd_counts = nbhd_counts.astype(int)
    return nbhd_counts

def get_ac_per_residue(df, colname, n_res):
    case_ac_per_residue = np.zeros((n_res, 1))
    ac_by_pos = df.groupby('aa_pos')[colname].sum().reset_index()
    case_ac_per_residue[ac_by_pos.aa_pos - 1, 0] = ac_by_pos[colname]
    return case_ac_per_residue

def get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim, seed=0):
    if seed is not None:
        np.random.seed(seed)
    n_alleles = int(  case_ac_per_residue.sum()  )
    gen = np.random.default_rng()
    null_ac_per_residue = gen.multivariate_hypergeometric(total_ac_per_residue.astype(int), n_alleles, n_sim).T
    return null_ac_per_residue

def get_case_control_ac_matrix(df, n_res, n_sim):
    case_ac_per_residue = get_ac_per_residue(df, 'ac_case', n_res)
    control_ac_per_residue = get_ac_per_residue(df, 'ac_control', n_res)
    total_ac_per_residue = (case_ac_per_residue + control_ac_per_residue).flatten()
    null_case_ac_per_residue = get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim)
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
):
    adjacency_matrix = get_adjacency_matrix(
        pdb_file_pos_guide,
        pdb_dir,
        pae_dir,
        uniprot_id,
        radius,
        pae_cutoff,
    )
    n_res = adjacency_matrix.shape[0]
    
    case_ac_matrix, control_ac_matrix = get_case_control_ac_matrix(df, n_res, n_sims)
    n_case = case_ac_matrix[:,0].sum()
    n_control = control_ac_matrix[:,0].sum()
    n_case_nbhd_mat = get_nbhd_counts(adjacency_matrix, case_ac_matrix)
    n_control_nbhd_mat = get_nbhd_counts(adjacency_matrix, control_ac_matrix)
    pval_lookup = get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_control)
    pval_matrix = pval_lookup[n_case_nbhd_mat, n_control_nbhd_mat]
    pval_columns = ['p_value'] + [f'null_pval_{i}' for i in range(n_sims)]
    df_pvals = pd.DataFrame(columns = pval_columns, data = pval_matrix)
    df_pvals['nbhd_case'] = n_case_nbhd_mat[:,0]
    df_pvals['nbhd_control'] = n_control_nbhd_mat[:,0]
    # Add original per-residue mutation counts
    df_pvals['original_case'] = case_ac_matrix[:,0]
    df_pvals['original_control'] = control_ac_matrix[:,0]
    df_pvals = df_pvals[['nbhd_case', 'nbhd_control', 'original_case', 'original_control'] + pval_columns]
    return df_pvals, adjacency_matrix

def write_df_pvals(results_dir, uniprot_id, df_pvals):
    with h5py.File(os.path.join(results_dir, 'p_values.h5'), 'a') as fid:
        null_pval_cols = [c for c in df_pvals.columns if c.startswith('null_pval')]
        write_dataset(fid, f'{uniprot_id}', df_pvals[['p_value']])
        write_dataset(fid, f'{uniprot_id}_null_pval', df_pvals[null_pval_cols])
        write_dataset(fid, f'{uniprot_id}_nbhd', df_pvals[['nbhd_case', 'nbhd_control']])
        write_dataset(fid, f'{uniprot_id}_original', df_pvals[['original_case', 'original_control']])

def scan_test_one_protein(df, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, uniprot_id, radius, pae_cutoff, n_sims):
    df_pvals, adj_mat = compute_all_pvals(
        df,
        pdb_file_pos_guide,
        pdb_dir,
        pae_dir,
        uniprot_id,
        n_sims,
        radius,
        pae_cutoff,
    )
    write_df_pvals(results_dir, uniprot_id, df_pvals)

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


def _filter_proteins_by_allele_count(df_rvas, df_fdr_filter, min_alleles=5):
    """Filter proteins to include only those with sufficient case and control alleles."""
    grouped = df_rvas.groupby('uniprot_id')[['ac_case', 'ac_control']].sum()
    ac_high_enough = grouped[(grouped['ac_case'] > min_alleles) & (grouped['ac_control'] > min_alleles)]
    uniprot_id_list = ac_high_enough.index.tolist()
    
    if df_fdr_filter is not None:
        uniprot_id_list = np.intersect1d(uniprot_id_list, np.unique(df_fdr_filter.uniprot_id))
    
    logger.info(f"Selected {len(uniprot_id_list)} proteins for analysis (min {min_alleles} alleles each)")
    return uniprot_id_list


def _process_proteins_batch(df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff, results_dir, n_sims, remove_nbhd):
    """Process each protein individually with scan test."""
    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'
    
    n_proteins = len(uniprot_id_list)
    for i, uniprot_id in enumerate(uniprot_id_list):
        logger.info(f'Processing {uniprot_id} (protein {i+1} out of {n_proteins})')
        try:
            df = df_rvas[df_rvas.uniprot_id == uniprot_id]
            if remove_nbhd is not None:
                    adjacency_matrix = get_adjacency_matrix(
                        pdb_file_pos_guide,
                        pdb_dir,
                        pae_dir,
                        uniprot_id,
                        radius,
                        pae_cutoff,
                    )
                    for to_remove in map(int, remove_nbhd.split(',')):
                        print(f'Removing neighborhood of position {to_remove} for {uniprot_id}')
                        nbhd = set(np.where(adjacency_matrix[to_remove-1] == 1)[0] + 1)
                        df.drop(df[df['aa_pos'].isin(nbhd)].index, inplace=True)
                    df.reset_index(drop=True, inplace=True)

            if (sum(df.ac_case) < 5) or (sum(df.ac_control) < 5):
                logger.warning(f'{uniprot_id}: There must be at least 5 case and 5 control alleles. Skipping.')
                continue
            
            scan_test_one_protein(
                df, pdb_file_pos_guide, pdb_dir, pae_dir, 
                results_dir, uniprot_id, radius, pae_cutoff, n_sims
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
    remove_nbhd,
):
    """
    Perform scan test analysis on protein structure data.
    
    Main orchestration function for the structure-informed rare variant association study.
    Processes variants across proteins and computes statistical associations with 3D neighborhoods.
    """
    
    # Handle FDR-only mode
    if fdr_only:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir)
        df_results.to_csv(fdr_file, sep='\t', index=False)
        return

    logger.info("Starting scan test analysis")
    logger.info(f"Input dataset contains {len(df_rvas)} variants across {df_rvas['uniprot_id'].nunique()} proteins")

    # Preprocess data
    df_processed = _preprocess_scan_data(df_rvas, ignore_ac)
    
    # Filter proteins by allele count
    uniprot_id_list = _filter_proteins_by_allele_count(df_processed, df_fdr_filter)
    
    # Process each protein
    _process_proteins_batch(
        df_processed, uniprot_id_list, reference_dir, 
        radius, pae_cutoff, results_dir, n_sims, remove_nbhd
    )
    
    # Compute FDR if requested
    if not no_fdr:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir)
        df_results.to_csv(fdr_file, sep='\t', index=False)
    