import os
import gc
import pandas as pd
import numpy as np
import h5py
from scipy.stats import fisher_exact, ttest_ind_from_stats
from utils import get_adjacency_matrix, write_dataset
from logger_config import get_logger
from empirical_fdr import compute_fdr

logger = get_logger(__name__)

def _protein_results_exist(results_dir, uniprot_id, quantitative=False):
    """Check if results already exist for a protein in the HDF5 file."""
    pval_file = 'p_values_quantitative.h5' if quantitative else 'p_values.h5'
    hdf5_path = os.path.join(results_dir, pval_file)

    if not os.path.exists(hdf5_path):
        return False

    try:
        with h5py.File(hdf5_path, 'r') as fid:
            # Check if the main p-values dataset exists for this protein
            return uniprot_id in fid
    except Exception as e:
        logger.warning(f"Error checking existing results for {uniprot_id}: {e}")
        return False

# --- Case-Control Analysis Functions (Original) ---

def get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_ctrl):
    max_n_case_nbhd = np.max(n_case_nbhd_mat)
    max_n_control_nbhd = np.max(n_control_nbhd_mat)
    
    pvals = np.ones((max_n_case_nbhd + 1, max_n_control_nbhd + 1))
    
    case_nbhd_range = np.arange(max_n_case_nbhd + 1)
    ctrl_nbhd_range = np.arange(max_n_control_nbhd + 1)
    n_case_grid, n_ctrl_grid = np.meshgrid(case_nbhd_range, ctrl_nbhd_range, indexing='ij')
    
    a = n_case_grid
    b = n_ctrl_grid
    c = n_case - a
    d = n_ctrl - b
    
    valid_mask = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    
    indices = np.where(valid_mask)
    for i, j in zip(indices[0], indices[1]):
        table = np.array([[i, j], [n_case - i, n_ctrl - j]])
        if table.min() >= 0:
            _, p = fisher_exact(table)
            pvals[i, j] = p
    
    return pvals

def get_nbhd_counts(adjacency_matrix, ac_per_residue):
    nbhd_counts = adjacency_matrix @ ac_per_residue
    nbhd_counts = nbhd_counts.astype(int)
    return nbhd_counts

def get_ac_per_residue(df, colname, n_res):
    ac_per_residue = np.zeros((n_res, 1))
    ac_by_pos = df.groupby('aa_pos')[colname].sum().reset_index()
    ac_per_residue[ac_by_pos.aa_pos - 1, 0] = ac_by_pos[colname]
    return ac_per_residue

def get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim, seed=0):
    if seed is not None:
        np.random.seed(seed)
    n_alleles = int(case_ac_per_residue.sum())
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

def compute_all_pvals(df, pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, n_sims, radius, pae_cutoff):
    adjacency_matrix = get_adjacency_matrix(
        pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, radius, pae_cutoff
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
    df_pvals = pd.DataFrame(columns=pval_columns, data=pval_matrix)
    df_pvals['nbhd_case'] = n_case_nbhd_mat[:,0]
    df_pvals['nbhd_control'] = n_control_nbhd_mat[:,0]
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
        df, pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, n_sims, radius, pae_cutoff
    )
    write_df_pvals(results_dir, uniprot_id, df_pvals)

# --- Quantitative Trait Analysis Functions (New) ---

def compute_all_pvals_quantitative(df, pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, n_sims, radius, pae_cutoff):
    adjacency_matrix = get_adjacency_matrix(
        pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, radius, pae_cutoff
    )
    n_residues = adjacency_matrix.shape[0]

    variants_df = df.drop_duplicates(subset=['Variant ID']).reset_index(drop=True)
    n_variants = len(variants_df)
    if n_variants < 5: # Not enough variants for a meaningful test
        logger.warning(f"{uniprot_id}: Not enough unique variants ({n_variants}) for quantitative test. Skipping.")
        return None, None

    variant_to_residue_map = np.zeros((n_residues, n_variants))
    variant_to_residue_map[variants_df['aa_pos'] - 1, np.arange(n_variants)] = 1

    beta_obs = variants_df['beta'].values
    is_variant_in_nbhd = (adjacency_matrix @ variant_to_residue_map) > 0

    n_in = is_variant_in_nbhd.sum(axis=1, keepdims=True).astype(float)
    total_n = float(n_variants)
    n_out = total_n - n_in

    # Process observed data first
    sum_beta_in_obs = is_variant_in_nbhd @ beta_obs.reshape(-1, 1)
    sum_sq_beta_in_obs = is_variant_in_nbhd @ (beta_obs**2).reshape(-1, 1)
    total_sum_beta_obs = beta_obs.sum()
    total_sum_sq_beta_obs = (beta_obs**2).sum()

    sum_beta_out_obs = total_sum_beta_obs - sum_beta_in_obs
    sum_sq_beta_out_obs = total_sum_sq_beta_obs - sum_sq_beta_in_obs

    # Process simulations in batches to reduce memory usage
    batch_size = min(100, n_sims)  # Process 100 simulations at a time
    sum_beta_in = np.zeros((n_residues, 1 + n_sims))
    sum_sq_beta_in = np.zeros((n_residues, 1 + n_sims))
    total_sum_beta = np.zeros(1 + n_sims)
    total_sum_sq_beta = np.zeros(1 + n_sims)

    # Set observed data (column 0)
    sum_beta_in[:, 0] = sum_beta_in_obs.flatten()
    sum_sq_beta_in[:, 0] = sum_sq_beta_in_obs.flatten()
    total_sum_beta[0] = total_sum_beta_obs
    total_sum_sq_beta[0] = total_sum_sq_beta_obs

    rng = np.random.default_rng()
    for batch_start in range(0, n_sims, batch_size):
        batch_end = min(batch_start + batch_size, n_sims)
        batch_n_sims = batch_end - batch_start

        # Create smaller batch matrix
        beta_batch = np.zeros((n_variants, batch_n_sims))
        for i in range(batch_n_sims):
            beta_batch[:, i] = rng.permutation(beta_obs)

        # Calculate neighborhood statistics for this batch
        sum_beta_in[:, batch_start + 1:batch_end + 1] = is_variant_in_nbhd @ beta_batch
        sum_sq_beta_in[:, batch_start + 1:batch_end + 1] = is_variant_in_nbhd @ (beta_batch**2)

        # Calculate total statistics for this batch
        total_sum_beta[batch_start + 1:batch_end + 1] = beta_batch.sum(axis=0)
        total_sum_sq_beta[batch_start + 1:batch_end + 1] = (beta_batch**2).sum(axis=0)

        # Clean up batch matrix to free memory
        del beta_batch
        gc.collect()  # Force garbage collection after each batch

    sum_beta_out = total_sum_beta - sum_beta_in
    sum_sq_beta_out = total_sum_sq_beta - sum_sq_beta_in

    epsilon = 1e-8
    mean_in = sum_beta_in / (n_in + epsilon)
    mean_out = sum_beta_out / (n_out + epsilon)
    
    var_in = (sum_sq_beta_in / (n_in + epsilon) - mean_in**2) * (n_in / (n_in - 1 + epsilon))
    var_out = (sum_sq_beta_out / (n_out + epsilon) - mean_out**2) * (n_out / (n_out - 1 + epsilon))

    var_in[var_in < 0] = 0
    var_out[var_out < 0] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        t_stat, p_vals = ttest_ind_from_stats(
            mean1=mean_in, std1=np.sqrt(var_in), nobs1=n_in,
            mean2=mean_out, std2=np.sqrt(var_out), nobs2=n_out,
            equal_var=False
        )

    # Filter out neighborhoods with fewer than 5 variants
    insufficient_variants = n_in.flatten() < 5
    p_vals[insufficient_variants] = np.nan
    t_stat[insufficient_variants] = np.nan

    # Set invalid p-values to 1.0 and corresponding t-stats to 0
    invalid_mask = np.isnan(p_vals)
    p_vals[invalid_mask] = 1.0
    t_stat[invalid_mask] = 0.0

    df_pvals = pd.DataFrame({
        'p_value': p_vals[:, 0], 't_stat': t_stat[:, 0],
        'mean_beta_in': mean_in[:, 0], 'mean_beta_out': mean_out[:, 0],
        'std_beta_in': np.sqrt(var_in[:, 0]), 'std_beta_out': np.sqrt(var_out[:, 0]),
        'n_variants_in': n_in.flatten(), 'n_variants_out': n_out.flatten()
    })
    
    null_pval_cols = [f'null_pval_{i}' for i in range(n_sims)]
    df_null_pvals = pd.DataFrame(p_vals[:, 1:], columns=null_pval_cols)
    df_pvals = pd.concat([df_pvals, df_null_pvals], axis=1)

    # Clean up large arrays to free memory
    del sum_beta_in, sum_sq_beta_in, sum_beta_out, sum_sq_beta_out
    del mean_in, mean_out, var_in, var_out, t_stat, p_vals
    del variant_to_residue_map, is_variant_in_nbhd
    gc.collect()

    return df_pvals, adjacency_matrix

def write_df_pvals_quantitative(results_dir, uniprot_id, df_pvals):
    with h5py.File(os.path.join(results_dir, 'p_values_quantitative.h5'), 'a') as fid:
        null_pval_cols = [c for c in df_pvals.columns if c.startswith('null_pval')]
        stat_cols = ['t_stat', 'mean_beta_in', 'mean_beta_out', 'std_beta_in', 'std_beta_out', 'n_variants_in', 'n_variants_out']
        
        write_dataset(fid, f'{uniprot_id}', df_pvals[['p_value']])
        write_dataset(fid, f'{uniprot_id}_null_pval', df_pvals[null_pval_cols])
        write_dataset(fid, f'{uniprot_id}_stats', df_pvals[stat_cols])

def scan_test_one_protein_quantitative(df, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, uniprot_id, radius, pae_cutoff, n_sims):
    df_pvals, adj_mat = compute_all_pvals_quantitative(
        df, pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, n_sims, radius, pae_cutoff
    )
    if df_pvals is not None:
        write_df_pvals_quantitative(results_dir, uniprot_id, df_pvals)

# --- Shared Logic & Main Entry Points ---

def _preprocess_scan_data(df_rvas, ignore_ac):
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
    grouped = df_rvas.groupby('uniprot_id')[['ac_case', 'ac_control']].sum()
    ac_high_enough = grouped[(grouped['ac_case'] >= min_alleles) & (grouped['ac_control'] >= min_alleles)]
    uniprot_id_list = ac_high_enough.index.tolist()
    
    if df_fdr_filter is not None:
        uniprot_id_list = list(np.intersect1d(uniprot_id_list, df_fdr_filter.uniprot_id.unique()))
    
    logger.info(f"Selected {len(uniprot_id_list)} proteins for case-control analysis (min {min_alleles} alleles each)")
    return uniprot_id_list

def _filter_proteins_by_variant_count(df_rvas, df_fdr_filter, min_variants=10):
    grouped = df_rvas.groupby('uniprot_id')['Variant ID'].nunique()
    variants_high_enough = grouped[grouped >= min_variants]
    uniprot_id_list = variants_high_enough.index.tolist()

    if df_fdr_filter is not None:
        uniprot_id_list = list(np.intersect1d(uniprot_id_list, df_fdr_filter.uniprot_id.unique()))
    
    logger.info(f"Selected {len(uniprot_id_list)} proteins for quantitative analysis (min {min_variants} variants each)")
    return uniprot_id_list

def _process_proteins_batch(df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff, results_dir, n_sims, test_function, quantitative=False):
    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'

    n_proteins = len(uniprot_id_list)
    n_processed = 0
    n_skipped = 0

    for i, uniprot_id in enumerate(uniprot_id_list):
        logger.info(f'Processing {uniprot_id} (protein {i+1} out of {n_proteins})')

        # Check if results already exist for this protein
        if _protein_results_exist(results_dir, uniprot_id, quantitative):
            logger.info(f'{uniprot_id}: Results already exist, skipping computation')
            n_skipped += 1
            continue

        try:
            df = df_rvas[df_rvas.uniprot_id == uniprot_id]
            test_function(df, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, uniprot_id, radius, pae_cutoff, n_sims)
            n_processed += 1

            # Force garbage collection every 50 proteins to prevent memory accumulation
            if (i + 1) % 50 == 0:
                gc.collect()
                logger.info(f'Garbage collection performed after processing {i+1} proteins')

        except Exception as e:
            logger.error(f'{uniprot_id}: Unexpected error - {e}', exc_info=True)
            continue

    logger.info(f'Batch processing complete: {n_processed} proteins processed, {n_skipped} proteins skipped (already computed)')

def scan_test(df_rvas, reference_dir, radius, pae_cutoff, results_dir, n_sims, no_fdr, fdr_only, fdr_cutoff, df_fdr_filter, ignore_ac, fdr_file):
    if fdr_only:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file='p_values.h5')
        df_results.to_csv(fdr_file, sep='\t', index=False)
        return

    logger.info("Starting case-control scan test analysis")
    df_processed = _preprocess_scan_data(df_rvas, ignore_ac)
    uniprot_id_list = _filter_proteins_by_allele_count(df_processed, df_fdr_filter)
    
    _process_proteins_batch(df_processed, uniprot_id_list, reference_dir, radius, pae_cutoff, results_dir, n_sims, scan_test_one_protein, quantitative=False)
    
    if not no_fdr:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file='p_values.h5')
        df_results.to_csv(fdr_file, sep='\t', index=False)

def scan_test_quantitative(df_rvas, reference_dir, radius, pae_cutoff, results_dir, n_sims, no_fdr, fdr_only, fdr_cutoff, df_fdr_filter, fdr_file):
    pval_filename = 'p_values_quantitative.h5'
    
    if fdr_only:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file=pval_filename, quantitative=True)
        df_results.to_csv(fdr_file, sep='\t', index=False)
        return
        
    logger.info("Starting quantitative trait scan test analysis")
    uniprot_id_list = _filter_proteins_by_variant_count(df_rvas, df_fdr_filter)
    
    _process_proteins_batch(df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff, results_dir, n_sims, scan_test_one_protein_quantitative, quantitative=True)
    
    if not no_fdr:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file=pval_filename, quantitative=True)
        df_results.to_csv(fdr_file, sep='\t', index=False)