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
from q_empirical_fdr import q_compute_fdr

logger = get_logger(__name__)


# def get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim, seed=0):
#     if seed is not None:
#         np.random.seed(seed)
#     n_alleles = int(  case_ac_per_residue.sum()  )
#     gen = np.random.default_rng()
#     null_ac_per_residue = gen.multivariate_hypergeometric(total_ac_per_residue.astype(int), n_alleles, n_sim).T
#     return null_ac_per_residue

def get_tstat_matrix(n_betahat, sum_betahat, sum_betahat_squared, total_n_betahat, total_sum_betahat, total_sum_betahat_squared):
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
    # Force n_in and n_out to be (L, 1) so they broadcast correctly across (L, 1001)
    n_in = np.asarray(n_betahat).reshape(-1, 1) 
    n_out = (total_n_betahat - n_in).reshape(-1, 1)
    
    # Error management to handle cases where n_in or n_out are 0 or 1, 
    # which would lead to division by zero or infinity values in variance calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        # Neighborhood stats
        mean_betahat_nbhd = sum_betahat / n_in
        # Use np.maximum(..., 1) or check for n < 2 to avoid divide by zero/negatives
        var_betahat_nbhd = (sum_betahat_squared - n_in * (mean_betahat_nbhd ** 2)) / (n_in - 1)
        
        # Outside stats
        mean_betahat_outside = (total_sum_betahat - sum_betahat) / n_out
        var_betahat_outside = ((total_sum_betahat_squared - sum_betahat_squared) - n_out*(mean_betahat_outside**2)) / (n_out - 1)

        # Standard Error and T-stat
        se_diff = np.sqrt((var_betahat_nbhd / n_in) + (var_betahat_outside / n_out))
        t_stat_matrix = (mean_betahat_nbhd - mean_betahat_outside) / se_diff

    # The mask should now match the shape (L, 1001) perfectly
    mask = ~np.isfinite(t_stat_matrix)
    t_stat_matrix[mask] = 0
    neg_abs_tstat_matrix = -np.abs(t_stat_matrix)
    return neg_abs_tstat_matrix

def get_betahats_per_residue(df, n_res, colname='betahat'):
    betahats_per_residue = np.zeros((n_res, 1))
    betahats_by_pos = df.groupby('aa_pos')[colname].sum().reset_index()
    betahats_per_residue[betahats_by_pos.aa_pos - 1, 0] = betahats_by_pos[colname]
    return betahats_per_residue

def get_nbhd_stats(adjacency_matrix, df, n_res, n_sims):
    '''
    add n_sims betahat_null_{i} columns to df by permuting betahat column
    add betahatsq columns
    then groupby aa_pos to get per-residue stats:
    1. count from df to get number of betahats per residue: L x 1
    2. sum from df to get sum_beta_per_residue: L x (n_sims + 1)
    3. sum from df to get sum_beta_squared_per_residue: L x (n_sims + 1)

    then multiply all three by the adjacency matrix to get outputs, which
    are per-neighborhood
    '''
    # 0. Prepare data
    base = get_betahats_per_residue(df, n_res).flatten()
    # base[df['aa_pos'].values - 1, 0] = df['betahat'].values
    # base = base.flatten()
    base_sq = base ** 2

    # 1. Obtain n_sims random permutations of betahat 
    null_values = np.array([np.random.permutation(base) for _ in range(n_sims)]).T
    #null_values = base[null_indices].T # Shape: (n_var, n_sims)
    null_sq = null_values ** 2

    # 2. Get the Grouping Key and create "Group IDs"
    # np.unique with return_inverse turns ['A', 'B', 'A'] into [0, 1, 0]
    # unique_groups, group_ids = np.unique(df['aa_pos'].values, return_inverse=True)
    # print(type(group_ids))
    # print(type(unique_groups))
    # print(unique_groups.shape)
    # print(group_ids.shape)

    group_ids = np.array(range(n_res))
    unique_groups = np.array(range(1, n_res + 1))
    n_groups = len(unique_groups)

    #aa_pos_arr = df['aa_pos'].values.reshape(-1, 1)

    # 3. Consolidate data into one matrix (Originals + Nulls)
    # We exclude 'aa_pos' here because we'll use 'group_ids' to index it
    data_to_sum = np.column_stack([
        base,                      # Original betahat
        null_values,               # All null betahats
        base_sq,                   # Original squared
        null_sq                    # All null squareds
    ])
    
    # 4. Perform the "Grouped Sum" using np.bincount
    # np.bincount(indices, weights=values) is the fastest way to do grouped sums
    group_sums = np.zeros((n_groups, data_to_sum.shape[1]))

    for i in range(data_to_sum.shape[1]):
        group_sums[:, i] = np.bincount(group_ids, weights=data_to_sum[:, i])

    # 5. Calculate Group Counts (the 'n_betahat' column)
    group_counts = np.bincount(group_ids)
    
    # 6. Wrap results into a single consolidated DataFrame
    betahat_cols = ['betahat'] + [f"betahat_null_{i}" for i in range(n_sims)]
    sq_betahat_cols = ['squared_betahat'] + [f"squared_betahat_null_{i}" for i in range(n_sims)]
    col_names = betahat_cols + sq_betahat_cols

    agg_df = pd.DataFrame(group_sums, columns=col_names)
    agg_df.insert(0, 'aa_pos', unique_groups)
    agg_df.insert(1, 'n_betahat', group_counts)

    # n_betahat = adjacency_matrix * agg_df['n_betahat'].to_numpy().reshape(-1, 1)  # L x 1
    # sum_betahat = adjacency_matrix @ agg_df[betahat_cols].to_numpy()  # L x (n_sims + 1)
    # sum_betahat_squared = adjacency_matrix @ agg_df[sq_betahat_cols].to_numpy()  # L x (n_sims + 1)

    try:
        n_betahat = adjacency_matrix @ group_counts  # L x 1
        sum_betahat = adjacency_matrix @ group_sums[:,:n_sims+1]  # L x (n_sims + 1)
        sum_betahat_squared = adjacency_matrix @ group_sums[:,n_sims+1:]  # L x (n_sims + 1)
    except Exception as e:
        print(f"Error occurred: {e}")
        
    return n_betahat, sum_betahat, sum_betahat_squared

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

    
    # nbhd_size: L x 1
    # sum_beta: L x (n_sims+1)
    # sum_beta_squared: L x (n_sims+1)
    n_betahat, sum_betahat, sum_betahat_squared = get_nbhd_stats(adjacency_matrix, df, n_res, n_sims)

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
    )
    n_a_tstat_columns = ['n_a_tstat'] + [f'null_n_a_tstat_{i}' for i in range(n_sims)]
    df_n_a_tstats = pd.DataFrame(columns = n_a_tstat_columns, data = neg_abs_tstat_matrix)

    #other stuff it's convenient to see in the output
    df_n_a_tstats['mean_betahat'] = sum_betahat[:,0] / n_betahat
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
    
