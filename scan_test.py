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

logger = get_logger(__name__)
    
def get_pval_lookup_case_control_old(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_ctrl):
    max_n_case_nbhd = np.max(n_case_nbhd_mat)
    max_n_control_nbhd = np.max(n_control_nbhd_mat)
    pvals = np.ones((max_n_case_nbhd+1, max_n_control_nbhd+1))
    for n_case_nbhd in range(max_n_case_nbhd + 1):
        for n_ctrl_nbhd in range(max_n_control_nbhd + 1):
            contingency_table = np.array([
                [n_case_nbhd, n_ctrl_nbhd],
                [n_case - n_case_nbhd, n_ctrl - n_ctrl_nbhd]
            ])
            if valid_for_fisher(contingency_table):
                _, p = fisher_exact(contingency_table)
                pvals[n_case_nbhd, n_ctrl_nbhd] = p
    return pvals

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

def get_all_pvals(
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
    df_pvals['ratio'] = (df_pvals['nbhd_case'] + 2) / (df_pvals['nbhd_control'] + 2 * n_control / n_case)
    df_pvals = df_pvals[['nbhd_case', 'nbhd_control'] + pval_columns + ['ratio']]
    return df_pvals, adjacency_matrix

def write_df_pvals(results_dir, uniprot_id, df_pvals):
    with h5py.File(os.path.join(results_dir, 'p_values.h5'), 'a') as fid:
        # uniprot_id = f.split('/')[-1].split('.')[0]
        null_pval_cols = [c for c in df_pvals.columns if c.startswith('null_pval')]
        write_dataset(fid, f'{uniprot_id}', df_pvals[['p_value', 'ratio']])
        write_dataset(fid, f'{uniprot_id}_null_pval', df_pvals[null_pval_cols])
        write_dataset(fid, f'{uniprot_id}_nbhd', df_pvals[['nbhd_case', 'nbhd_control']])

def scan_test_one_protein(df, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, uniprot_id, radius, pae_cutoff, n_sims):
    results_prefix = os.path.join(results_dir, uniprot_id)
    df_pvals, adj_mat = get_all_pvals(
        df,
        pdb_file_pos_guide,
        pdb_dir,
        pae_dir,
        uniprot_id,
        n_sims,
        radius,
        pae_cutoff,
    )
    # np.save(f'{results_prefix}.adj_mat.npy', adj_mat)
    df.to_csv(f'{results_prefix}.df_rvas.tsv', sep='\t', index=False)
    write_df_pvals(results_dir, uniprot_id, df_pvals)

def summarize_results(df_results, fdr_cutoff, reference_dir, annot_file):

    if annot_file is not None:
        logger.info('')
        logger.info('Merging with annotations')
        df_ba = pd.read_csv(annot_file, sep='\t')
        df_ba_extra_cols = list([x for x in df_ba.columns if not x in ['uniprot_id', 'aa_pos']])
        df_results = pd.merge(df_results, df_ba, how='left', indicator=True)
        df_results['annotated'] = df_results['_merge'] == 'both'

    df_gene = pd.read_csv(f'{reference_dir}/gene_to_uniprot_id.tsv', sep='\t')
    df_results = df_results.merge(df_gene, how='left', on='uniprot_id')
    
    top_hits_all_genes = df_results.loc[df_results.groupby('uniprot_id')['fdr'].idxmin()]
    top_hits_sig = top_hits_all_genes[top_hits_all_genes.fdr<fdr_cutoff]
    top_hits_sig = top_hits_sig.sort_values(by='p_value')
    logger.info('')
    logger.info(f'{len(top_hits_sig)} out of {len(top_hits_all_genes)} proteins have a neighborhood significant at {fdr_cutoff}.')
    logger.info(f'Top 20 hits:\n{top_hits_sig[0:20].to_string()}')
    
    if annot_file is not None:
        overall_num_annot = df_results.annotated.sum()
        overall_prop_annot = overall_num_annot / len(df_results)
        df_sig = df_results[df_results.fdr<fdr_cutoff]
        prop_sig_annot = df_sig.annotated.mean()
        num_prot_with_sig_annot = df_sig.groupby('uniprot_id')['annotated'].max().sum()
        num_proteins_sig = len(top_hits_sig)
        top_hits_sig_annot = top_hits_sig.annotated.sum()
        
        c = [
            [
                top_hits_sig_annot,
                num_proteins_sig - top_hits_sig_annot
            ],[
                overall_num_annot - top_hits_sig_annot,
                len(df_results) - overall_num_annot - (num_proteins_sig - top_hits_sig_annot)
            ]
        ]
        _, p = fisher_exact(c)

        logger.info(f'Overall prop annotated: {overall_prop_annot}')
        logger.info(f'Proportion significant residues annotated: {prop_sig_annot}')
        logger.info(f'{num_prot_with_sig_annot}/{num_proteins_sig} ({num_prot_with_sig_annot/num_proteins_sig*100}%) have a significant annotated residue.')
        logger.info(f'{top_hits_sig_annot}/{num_proteins_sig} ({top_hits_sig_annot/num_proteins_sig*100}%) top significant residues annotated.')
        logger.info(f'P = {p}')
        logger.info('')

        df_to_print = df_results.loc[
                df_results.annotated & (df_results.fdr<fdr_cutoff),
                ['gene_name', 'uniprot_id', 'p_value', 'fdr']+df_ba_extra_cols
            ].drop_duplicates().copy()
        df_to_print = df_to_print.loc[df_to_print.groupby('uniprot_id')['fdr'].idxmin()]
        logger.info(f'Sample results:\n{df_to_print[0:20].to_string()}')


def _prepare_fdr_filters(df_fdr_filter):
    """Prepare filtering criteria for FDR computation."""
    if df_fdr_filter is None:
        return None, None
    
    uniprot_filter_list = np.unique(df_fdr_filter['uniprot_id'])
    aa_pos_filters = None
    
    # Extract amino acid positions to keep for each protein
    if 'aa_pos' in df_fdr_filter.columns:
        aa_pos_filters = {}
        for uniprot_id in uniprot_filter_list:
            aa_pos_keep = set(df_fdr_filter.loc[df_fdr_filter.uniprot_id == uniprot_id, 'aa_pos'].values)
            aa_pos_filters[uniprot_id] = aa_pos_keep
    
    return uniprot_filter_list, aa_pos_filters


def _load_and_filter_pvalues(results_dir, uniprot_filter_list, aa_pos_filters):
    """Load p-values from HDF5 file and apply filters."""
    to_concat = []
    n_sims = None
    
    with h5py.File(os.path.join(results_dir, 'p_values.h5'), 'a') as fid:
        uniprot_ids = [k for k in fid.keys() if '_' not in k]
        if uniprot_filter_list is not None:
            uniprot_ids = list(set(uniprot_ids) & set(uniprot_filter_list))

        logger.info('Reading p-values')
        for uniprot_id in uniprot_ids:
            df = read_p_values(fid, uniprot_id)
            
            # Apply amino acid position filter if specified
            if aa_pos_filters is not None and uniprot_id in aa_pos_filters:
                aa_pos_keep = aa_pos_filters[uniprot_id]
                # Create boolean mask: positions are 1-indexed, dataframe indices are 0-indexed
                mask = np.array([x+1 in aa_pos_keep for x in range(len(df))])
                df = df[mask]
            
            to_concat.append(df)
        
    # Ensure we have data to process
    if not to_concat:
        raise ValueError("No proteins found for FDR computation. Check filters and input data.")
    
    logger.info('Concatenating and sorting p-values')
    df_pvals = pd.concat(to_concat)
    df_pvals = df_pvals.sort_values(by='p_value').reset_index(drop=True)
    
    # Get n_sims from first protein (all should have same value)
    n_sims = fid[f'{uniprot_ids[0]}_null_pval'].shape[1]
    
    return df_pvals, uniprot_ids, n_sims


def _compute_false_discoveries(results_dir, uniprot_ids, aa_pos_filters, df_pvals, n_sims, large_p_threshold=0.05):
    """Compute false discovery statistics from null distributions."""
    logger.info('Computing average false discoveries')
    mask = df_pvals.p_value <= large_p_threshold
    false_discoveries_avg = np.zeros(df_pvals.shape[0])
    
    with h5py.File(os.path.join(results_dir, 'p_values.h5'), 'a') as fid:
        null_pvals = []
        for i, uniprot_id in enumerate(uniprot_ids):
            null_pvals_one_uniprot = fid[f'{uniprot_id}_null_pval'][:]
            
            # Apply same amino acid position filter to null p-values
            if aa_pos_filters is not None and uniprot_id in aa_pos_filters:
                aa_pos_keep = aa_pos_filters[uniprot_id]
                # Create boolean mask for null p-values (same logic as in _load_and_filter_pvalues)
                mask_positions = np.array([x+1 in aa_pos_keep for x in range(null_pvals_one_uniprot.shape[0])])
                null_pvals_one_uniprot = null_pvals_one_uniprot[mask_positions, :]
            
            null_pvals_one_uniprot = null_pvals_one_uniprot.flatten()
            null_pvals.extend(null_pvals_one_uniprot[null_pvals_one_uniprot < large_p_threshold])
            
            if (i % 100 == 99) or (i == (len(uniprot_ids) - 1)):
                if len(uniprot_ids) > 100:
                    logger.debug(f'Computing false discoveries from protein {i} out of {len(uniprot_ids)}')
                null_pvals = np.sort(np.array(null_pvals))
                false_disc = np.empty(len(df_pvals.p_value))
                if np.any(mask):
                    false_disc[mask] = np.searchsorted(null_pvals, df_pvals.p_value[mask], side='right') / n_sims
                if np.any(~mask):
                    false_disc[~mask] = df_pvals.shape[0]
                false_discoveries_avg += false_disc.tolist()
                null_pvals = []
    
    return false_discoveries_avg


def _apply_fdr_correction(df_pvals, false_discoveries_avg):
    """Apply FDR correction and format results."""
    logger.info('Computing FDR')
    df_pvals['false_discoveries_avg'] = false_discoveries_avg
    df_pvals['fdr'] = [x / (i+1) for i, x in enumerate(false_discoveries_avg)]
    df_pvals['fdr'] = df_pvals['fdr'][::-1].cummin()[::-1]
    
    return df_pvals[['uniprot_id', 'aa_pos', 'p_value', 'fdr', 'nbhd_case', 'nbhd_control', 'ratio']]


def compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, annot_file=None, large_p_threshold=0.05):
    """
    Compute False Discovery Rate correction for scan test results.
    
    Main orchestration function that coordinates the FDR computation workflow.
    """
    logger.info('Computing FDR')
    
    # Prepare filtering criteria
    uniprot_filter_list, aa_pos_filters = _prepare_fdr_filters(df_fdr_filter)
    
    # Load and filter p-values
    df_pvals, uniprot_ids, n_sims = _load_and_filter_pvalues(
        results_dir, uniprot_filter_list, aa_pos_filters
    )
    
    # Compute false discoveries from null distributions
    false_discoveries_avg = _compute_false_discoveries(
        results_dir, uniprot_ids, aa_pos_filters, df_pvals, n_sims, large_p_threshold
    )
    
    # Apply FDR correction
    df_results = _apply_fdr_correction(df_pvals, false_discoveries_avg)
    
    # Summarize and return results
    summarize_results(df_results, fdr_cutoff, reference_dir, annot_file)
    
    return df_results

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


def _process_proteins_batch(df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff, results_dir, n_sims):
    """Process each protein individually with scan test."""
    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'
    
    n_proteins = len(uniprot_id_list)
    for i, uniprot_id in enumerate(uniprot_id_list):
        logger.info(f'Processing {uniprot_id} (protein {i+1} out of {n_proteins})')
        try:
            df = df_rvas[df_rvas.uniprot_id == uniprot_id]
            if (sum(df.ac_case) < 5) or (sum(df.ac_control) < 5):
                logger.warning(f'{uniprot_id}: There must be at least 5 case and 5 control alleles. Skipping.')
                continue
            
            scan_test_one_protein(
                df, pdb_file_pos_guide, pdb_dir, pae_dir, 
                results_dir, uniprot_id, radius, pae_cutoff, n_sims
            )
        except Exception as e:
            logger.error(f'Error processing {uniprot_id}: {e}')
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
):
    """
    Perform scan test analysis on protein structure data.
    
    Main orchestration function for the structure-informed rare variant association study.
    Processes variants across proteins and computes statistical associations with 3D neighborhoods.
    """
    logger.info("Starting scan test analysis")
    logger.info(f"Input dataset contains {len(df_rvas)} variants across {df_rvas['uniprot_id'].nunique()} proteins")

    annot_file = f'{reference_dir}/annotations/g2p_binding_site_active_site.tsv'
    
    # Handle FDR-only mode
    if fdr_only:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, annot_file=annot_file)
        df_results.to_csv(fdr_file, sep='\t', index=False)
        return

    # Preprocess data
    df_processed = _preprocess_scan_data(df_rvas, ignore_ac)
    
    # Filter proteins by allele count
    uniprot_id_list = _filter_proteins_by_allele_count(df_processed, df_fdr_filter)
    
    # Process each protein
    _process_proteins_batch(
        df_processed, uniprot_id_list, reference_dir, 
        radius, pae_cutoff, results_dir, n_sims
    )
    
    # Compute FDR if requested
    if not no_fdr:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, annot_file=annot_file)
        df_results.to_csv(fdr_file, sep='\t', index=False)
    