import os
import pandas as pd
import numpy as np
import glob
import bisect
import h5py
from scipy.stats import fisher_exact, binom
from scipy import special
from scipy import sparse
from utils import get_adjacency_matrix, valid_for_fisher, write_dataset, read_p_values
from logger_config import get_logger
from q_empirical_fdr import q_compute_fdr

logger = get_logger(__name__)


# def get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim, seed=0):
#     if seed is not None:
#         np.random.seed(seed)
#     n_alleles = int(  case_ac_per_residue.sum()  )
#     gen = np.random.default_rng()
#     null_ac_per_residue = gen.multivariate_hypergeometric(total_ac_per_residue.astype(int), n_alleles, n_sim).T
#     return null_ac_per_residue

def get_tstat_matrix(n_betahat, sum_betahat, sum_betahat_squared, total_n_betahat, total_sum_betahat, total_sum_betahat_squared, min_variants=10):
    '''
    Compute negative absolute value of t-statistic for each residue's neighborhood.
    n_betahat: L x 1
    sum_betahat: L x (n_sims + 1)
    sum_betahat_squared: L x (n_sims + 1)
    total_n_betahat: scalar
    total_sum_betahat: scalar
    total_sum_betahat_squared: scalar

    returns:
    neg_abs_tstat_matrix: L x (n_sims + 1)
    '''
    n_in = np.asarray(n_betahat).reshape(-1, 1) 
    n_out = (total_n_betahat - n_in).reshape(-1, 1)

    # Create a mask for valid neighborhoods (must have at least min_variants)
    # Also ensure there are at least 2 variants outside to calculate outside variance
    valid_mask = (n_in >= min_variants) & (total_n_betahat - n_in >= 2)

    with np.errstate(divide='ignore', invalid='ignore'):
        # 1. Neighborhood Stats
        mean_betahat_nbhd = sum_betahat / n_in
        var_betahat_nbhd = (sum_betahat_squared - n_in * (mean_betahat_nbhd ** 2)) / (n_in - 1)
        
        # 2. Outside Stats
        #n_out = total_n_betahat - n_in
        mean_betahat_outside = (total_sum_betahat - sum_betahat) / n_out
        var_betahat_outside = ((total_sum_betahat_squared - sum_betahat_squared) - n_out*(mean_betahat_outside**2)) / (n_out - 1)

        # 3. T-statistic
        se_diff = np.sqrt((var_betahat_nbhd / n_in) + (var_betahat_outside / n_out))
        t_stat_matrix = (mean_betahat_nbhd - mean_betahat_outside) / se_diff

    # Apply the mask: only keep values where the threshold was met
    # Everything else (including NaNs and low-count neighborhoods) becomes 0
    t_stat_matrix = np.where(valid_mask & np.isfinite(t_stat_matrix), t_stat_matrix, 0)
    neg_abs_tstat_matrix = -np.abs(t_stat_matrix)
    return neg_abs_tstat_matrix

def get_betahats_per_residue(df, n_res, n_sims, colname='betahat'):
    '''
    Aggregate statistics on a per-residue basis.
    Get betahat and null distribution
    Also betahat squared and null squared
    Then groupby aa_pos to get per-residue stats:
    1. count to get number of betahats per residue: L x 1
    2. sum to get base_per_residue and null_per_residue: L x (n_sims + 1)
    3. sum to get base_sq_per_residue and null_sq_per_residue: L x (n_sims + 1)
    In the null ensure that each residue is allocated the same number of betahats as in the original data,
    but the betahats are shuffled globally across the protein. 
    This preserves the distribution of betahats per residue while breaking any true association with position.
    This is done by creating a sparse indicator matrix M of shape (n_res, n_var) where M[i,j] = 1 if variant j maps to residue i, else 0.
    '''

    # Prepare indices and values
    pos = df['aa_pos'].to_numpy() - 1  # zero-based
    betas = df[colname].to_numpy()
    betas_sq = betas ** 2
    
    n_var = len(pos)

    # Create (sparse) matrix to map variants to residues
    # Shape: (n_res, n_var)
    # Since M is fixed, n_per_residue stays constant.
    M = sparse.csr_matrix(
        (np.ones(n_var), (pos, np.arange(n_var))),
        shape=(n_res, n_var)
    )

    # Summing betas per residue: result is (n_res, 1)
    base_per_residue = (M @ betas).reshape(-1, 1)
    base_sq_per_residue = (M @ betas_sq).reshape(-1, 1)

    # Permutations: (n_sims, n_var)
    # Apply permuations to the betas globally
    perm_idx = np.array([np.random.permutation(n_var) for _ in range(n_sims)])
    # null values: (n_sims, n_var) -> transpose for matrix mult
    null = betas[perm_idx].T 
    null_sq = betas_sq[perm_idx].T

    # # If memory becomes an issue, we can compute the nulls in a loop instead of all at once:
    # for i in range(n_sims):
    #     # Shuffling the array globally
    #     permuted_indices = np.random.permutation(n_var)
    #     null_betas[:, i] = betas[permuted_indices]
    #     null_betas_sq[:, i] = betas_sq[permuted_indices]

    # Map rermuted ralues to residues (retains the same number of betahats per residue, but breaks any true association with position)
    # Resulting nulls: (n_res, n_sims)
    null_per_residue = M @ null
    null_sq_per_residue = M @ null_sq
    
    # Also calculate number of variants per residue (n_betahat)
    # Summing the rows of the indicator matrix M
    n_per_residue = np.array(M.sum(axis=1)).flatten()

    return base_per_residue, base_sq_per_residue, null_per_residue, null_sq_per_residue, n_per_residue

def get_nbhd_stats(adjacency_matrix, n_per_residue, sum_betahat_matrix, sum_betahat_sq_matrix):
    '''
    Multiply residue-level statistics by the adjacency matrix to get per-neighborhood statistics
    '''

    # Multiply by Adjacency Matrix
    # Extract neighborhood-level stats by multiplying the per-residue stats with the adjacency matrix
    # Adjacency matrix should be (n_res, n_res)
    try:
        n_betahat_nbhd = adjacency_matrix @ n_per_residue
        sum_betahat_nbhd = adjacency_matrix @ sum_betahat_matrix
        sum_betahat_sq_nbhd = adjacency_matrix @ sum_betahat_sq_matrix
    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None
        
    return n_betahat_nbhd, sum_betahat_nbhd, sum_betahat_sq_nbhd


def compute_all_n_a_tstats(
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

    # Compute permutations (null distribution) once for the whole protein, 
    # then we can reuse for each neighborhood by multiplying with the adjacency matrix
    base, base_sq, nulls, nulls_sq, n_per_residue = get_betahats_per_residue(df, n_res, n_sims)

    # 2. Consolidate into matrices for neighborhood summation
    # We stack the 'actual' result with the 'null' results
    # Shape: (n_res, n_sims + 1)
    sum_betahat_matrix = np.column_stack([base, nulls])
    sum_betahat_sq_matrix = np.column_stack([base_sq, nulls_sq])
    
    # nbhd_size: L x 1
    # sum_beta: L x (n_sims+1)
    # sum_beta_squared: L x (n_sims+1)
    n_betahat, sum_betahat, sum_betahat_squared = get_nbhd_stats(adjacency_matrix, n_per_residue, sum_betahat_matrix, sum_betahat_sq_matrix)

    # get the totals for the whole protein
    total_n_betahat = df.shape[0]
    total_sum_betahat = df['betahat'].sum()
    total_sum_betahat_squared = (df['betahat'] ** 2).sum()

    # neg_abs_tstat_matrix: L x (n_sims+1)
    # negative absolute value of the t-statistic
    neg_abs_tstat_matrix = get_tstat_matrix(
        n_betahat,
        sum_betahat,
        sum_betahat_squared,
        total_n_betahat,
        total_sum_betahat,
        total_sum_betahat_squared,
        min_variants=10
    )

    # Create a mask to avoid division by zero
    valid_n = n_betahat > 0
    # Safely calculate mean_betahat
    # If n_betahat is 0, result will be 0.0
    mean_betahat = np.divide(
        sum_betahat[:, 0], 
        n_betahat, 
        out=np.zeros_like(n_betahat, dtype=float), 
        where=valid_n
    )

    n_a_tstat_columns = ['n_a_tstat'] + [f'null_n_a_tstat_{i}' for i in range(n_sims)]
    df_n_a_tstats = pd.DataFrame(columns = n_a_tstat_columns, data = neg_abs_tstat_matrix)

    #other stuff it's convenient to see in the output
    df_n_a_tstats['mean_betahat'] = mean_betahat
    df_n_a_tstats['n_betahat'] = n_betahat
    df_n_a_tstats = df_n_a_tstats[['mean_betahat', 'n_betahat'] + n_a_tstat_columns]
    return df_n_a_tstats

def write_df_n_a_tstats(results_dir, uniprot_id, df_n_a_tstats, n_a_tstat_file):
    with h5py.File(os.path.join(results_dir, n_a_tstat_file), 'a') as fid:
        null_n_a_tstat_cols = [c for c in df_n_a_tstats.columns if c.startswith('null_n_a_tstat')]
        # Save amino acid positions (dataframe index + 1, since index is 0-based)
        aa_positions = (df_n_a_tstats.index + 1).to_numpy().reshape(-1, 1)
        write_dataset(fid, f'{uniprot_id}_aa_pos', aa_positions)
        write_dataset(fid, f'{uniprot_id}', df_n_a_tstats[['n_a_tstat']])
        write_dataset(fid, f'{uniprot_id}_null_n_a_tstat', df_n_a_tstats[null_n_a_tstat_cols])
        write_dataset(fid, f'{uniprot_id}_mean_beta', df_n_a_tstats[['mean_betahat', 'n_betahat']])

def scan_test_one_protein(df, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, uniprot_id, radius, pae_cutoff, n_sims, large_threshold, n_a_tstat_file):
    df_n_a_tstats = compute_all_n_a_tstats(
        df,
        pdb_file_pos_guide,
        pdb_dir,
        pae_dir,
        uniprot_id,
        n_sims,
        radius,
        pae_cutoff,
    )

    # Zero out residues with n_a_tstat above a large_threshold
    # tmp_stat = df_n_a_tstats.n_a_tstat
    # tmp_mean = df_n_a_tstats.mean_betahat
    # tmp_n = df_n_a_tstats.n_betahat
    null_n_a_tstat_cols = [c for c in df_n_a_tstats.columns if c.startswith('null_n_a_tstat')]
    cols = null_n_a_tstat_cols #+ ['n_a_tstat']
    df_n_a_tstats.loc[:, cols] = df_n_a_tstats.loc[:, cols].mask(
        df_n_a_tstats.loc[:, cols] > large_threshold,
        0
    )
    ## Test:
    # mask_rows = df_n_a_tstats["n_betahat"] < 30
    # # Zero out selected columns for those rows
    # df_n_a_tstats.loc[mask_rows, cols] = 0

    write_df_n_a_tstats(results_dir, uniprot_id, df_n_a_tstats, n_a_tstat_file)

    if df_n_a_tstats.n_a_tstat.lt(large_threshold).any():
        return 1
    else:
        return 0
    # if not df_n_a_tstats.empty:
    #     write_df_n_a_tstats(results_dir, uniprot_id, df_n_a_tstats, n_a_tstat_file)
    #     return 1
    # else:
    #     return 0

def _filter_proteins_by_allele_count(df_rvas, df_fdr_filter, min_alleles=5):
    # needs to change to filter to enough betahats in the protein. at least 10?
    # or can we drop this altogether since we'll filter to enough betahats earlier?
    """Filter proteins to include only those with sufficient case and control alleles."""
    grouped = df_rvas.groupby('uniprot_id')[['ac_case', 'ac_control']].sum()
    ac_high_enough = grouped[(grouped['ac_case'] > min_alleles) & (grouped['ac_control'] > min_alleles)]
    uniprot_id_list = ac_high_enough.index.tolist()
    
    if df_fdr_filter is not None:
        uniprot_id_list = np.intersect1d(uniprot_id_list, np.unique(df_fdr_filter.uniprot_id))
    
    logger.info(f"Selected {len(uniprot_id_list)} proteins for analysis (min {min_alleles} alleles each)")
    return uniprot_id_list

def _filter_proteins_for_valid_tests(df_rvas, df_fdr_filter, threshold=10):
    """Filter proteins to include only those with sufficient number of variants."""
    grouped = df_rvas.groupby('uniprot_id')['betahat'].count()
    filtered = grouped[(grouped >= threshold)]
    uniprot_id_list = filtered.index.tolist()
    
    if df_fdr_filter is not None:
        uniprot_id_list = np.intersect1d(uniprot_id_list, np.unique(df_fdr_filter.uniprot_id))

    logger.info(f"Selected {len(uniprot_id_list)} proteins for analysis (min {threshold} variants each)")
    return uniprot_id_list


def _process_proteins_batch(df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff, results_dir, n_sims, remove_nbhd, large_threshold, n_a_tstat_file):
    """Process each protein individually with scan test."""
    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'
    
    n_proteins = len(uniprot_id_list)
    count_proteins = 0
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

            # add filter to at least 10 betahats
            
            valid_protein = scan_test_one_protein(
                df, pdb_file_pos_guide, pdb_dir, pae_dir, 
                results_dir, uniprot_id, radius, pae_cutoff, n_sims, large_threshold,
                n_a_tstat_file
            )
            count_proteins = count_proteins + valid_protein

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
    
    return count_proteins


def q_scan_test(
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
    n_a_tstat_file,
    remove_nbhd,
):
    """
    Perform scan test analysis on protein structure data.
    
    Main orchestration function for the structure-informed rare variant association study.
    Processes variants across proteins and computes statistical associations with 3D neighborhoods.
    """
    df_rvas = df_rvas.rename(columns={"ac_case": "betahat"})

    # logger.info("df_rvas columns: " + ", ".join(df_rvas.columns))
    # logger.info(f"df_rvas: {df_rvas.head()}")

    large_threshold = -2.0  # threshold for filtering residues with no variants

    # Handle FDR-only mode
    if fdr_only:
        df_results = q_compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, n_a_tstat_file, large_threshold)
        df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
        return

    logger.info("Starting scan test analysis")
    logger.info(f"Input dataset contains {len(df_rvas)} variants across {df_rvas['uniprot_id'].nunique()} proteins")

    # Preprocess data
    #df_processed = _preprocess_scan_data(df_rvas, ignore_ac)
    df_processed = df_rvas.copy()
    
    # Filter proteins by allele count
    uniprot_id_list = _filter_proteins_for_valid_tests(df_processed, df_fdr_filter)
    
    # Process each protein
    count_proteins = _process_proteins_batch(
        df_processed, uniprot_id_list, reference_dir, 
        radius, pae_cutoff, results_dir, n_sims, remove_nbhd, large_threshold,
        n_a_tstat_file
    )
    
    if count_proteins == 0:
        logger.warning("No protein neighborhoods pass the filtering criteria. Exiting scan test.")
        return
    else:
        logger.info(f"Completed scan test for {count_proteins} proteins with valid neighborhoods.")
    
    # Compute FDR if requested
    if not no_fdr:
        df_results = q_compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, n_a_tstat_file, large_threshold)
        df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
    
