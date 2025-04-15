import os
import pandas as pd
import numpy as np
import glob
import bisect
import h5py
from scipy.stats import fisher_exact, binom
from scipy import special
from utils import get_adjacency_matrix, valid_for_fisher, write_dataset, read_p_values
    
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

def get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim):
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
        pdb_file,
        n_sims,
        radius = 15,
):
    adjacency_matrix = get_adjacency_matrix(pdb_file, radius)
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

def compute_fdr(results_dir, df_fdr_filter, large_p_threshold = 0.05):
    print('computing fdr')
    to_concat = []

    index_filter = None
    uniprot_filter_list = None
    if df_fdr_filter is not None:
        uniprot_filter_list = np.unique(df_fdr_filter['uniprot_id'])
        if 'aa_pos' in df_fdr_filter.columns:
            index_filter = {}
    
    with h5py.File(os.path.join(results_dir, 'p_values.h5'), 'a') as fid:
        uniprot_ids = [k for k in fid.keys() if '_' not in k]
        if uniprot_filter_list is not None:
            uniprot_ids = list(set(uniprot_ids) & set(uniprot_filter_list))
        for uniprot_id in uniprot_ids:
            df = read_p_values(fid, uniprot_id)
            
            if index_filter is not None:
                aa_pos_keep = set(df_fdr_filter.loc[df_fdr_filter.uniprot_id == uniprot_id, 'aa_pos'].values)
                index_filter[uniprot_id] = np.array([x+1 in aa_pos_keep for x in range(len(df))])
                df = df[index_filter[uniprot_id]]
            to_concat.append(df)
        df_pvals = pd.concat(to_concat)
        to_concat = None
        df_pvals = df_pvals.sort_values(by='p_value').reset_index(drop=True)
        
        false_discoveries_avg = np.zeros(df_pvals.shape[0])
        num_large_p = 0
        for uniprot_id in uniprot_ids:
            null_pvals = fid[f'{uniprot_id}_null_pval'][:]
            if index_filter is not None:
                f = index_filter[uniprot_id]
                null_pvals = null_pvals[f,:]
            null_pvals = null_pvals.flatten()
            num_large_p += np.sum( null_pvals >= large_p_threshold )
            null_pvals = null_pvals[ null_pvals < large_p_threshold ]
            null_pvals = np.sort(null_pvals)
            n_sims = fid[f'{uniprot_id}_null_pval'].shape[1]
            false_discoveries_avg += [bisect.bisect_right(null_pvals, p)/n_sims if p<=large_p_threshold else df_pvals.shape[0] for p in df_pvals.p_value]
    df_pvals['false_discoveries_avg'] = false_discoveries_avg
    df_pvals['fdr'] = [x / (i+1) for i, x in enumerate(false_discoveries_avg)]
    df_pvals['fdr'] = df_pvals['fdr'][::-1].cummin()[::-1]
    df_results = df_pvals[['uniprot_id', 'aa_pos', 'p_value', 'fdr', 'nbhd_case', 'nbhd_control', 'ratio']]
    return df_results

def write_df_pvals(results_dir, uniprot_id, df_pvals):
    with h5py.File(os.path.join(results_dir, 'p_values.h5'), 'a') as fid:
        # uniprot_id = f.split('/')[-1].split('.')[0]
        null_pval_cols = [c for c in df_pvals.columns if c.startswith('null_pval')]
        write_dataset(fid, f'{uniprot_id}', df_pvals[['p_value', 'ratio']])
        write_dataset(fid, f'{uniprot_id}_null_pval', df_pvals[null_pval_cols])
        write_dataset(fid, f'{uniprot_id}_nbhd', df_pvals[['nbhd_case', 'nbhd_control']])

def scan_test_one_protein(df, pdb_file, results_dir, uniprot_id, radius, n_sims):
    results_prefix = os.path.join(results_dir, uniprot_id)
    df_pvals, adj_mat = get_all_pvals(df, pdb_file, n_sims, radius)
    np.save(f'{results_prefix}.adj_mat.npy', adj_mat)
    df.to_csv(f'{results_prefix}.df_rvas.tsv', sep='\t', index=False)
    write_df_pvals(results_dir, uniprot_id, df_pvals)

def scan_test(df_rvas, reference_dir, radius, results_dir, n_sims, no_fdr, fdr_only, df_fdr_filter):
    '''
    df_rvas is the output of map_to_protein. reference_dir has the pdb structures. this function
    should perform the scan test for all proteins and return a data frame with all the results.
    '''
    print('performing scan test')

    if fdr_only:
        df_results = compute_fdr(results_dir, df_fdr_filter)
        df_results.to_csv(os.path.join(results_dir, 'all_proteins.fdr.tsv'), sep='\t', index=False)
        return
    
    uniprot_id_list = np.unique(df_rvas.uniprot_id)
    if df_fdr_filter is not None:
        uniprot_id_list = np.intersect1d(uniprot_id_list, np.unique(df_fdr_filter.uniprot_id))
        

    n_proteins = len(uniprot_id_list)
    for i, uniprot_id in enumerate(uniprot_id_list):
        print('\n', uniprot_id, f'number {i} out of {n_proteins}')
        try:
            df = df_rvas[df_rvas.uniprot_id == uniprot_id]
            if len(np.unique(df.pdb_filename)) > 1:
                print('skipping when there is more than one pdb file')
                continue
            pdb_filename = np.unique(df.pdb_filename)[0]
            df = df[df.pdb_filename == pdb_filename].reset_index(drop=True)
            full_pdb_filename = os.path.join(reference_dir, 'pdb_files', pdb_filename)
            if not os.path.isfile(full_pdb_filename):
                print('missing pdb file. skipping.')
                continue
            scan_test_one_protein(df, full_pdb_filename, results_dir, uniprot_id, radius, n_sims)
        except Exception as e:
            print(f'Error for {uniprot_id}: {e}')
            continue
    if not no_fdr:
        df_results = compute_fdr(results_dir, df_fdr_filter)
        df_results.to_csv(os.path.join(results_dir, 'all_proteins.fdr.tsv'), sep='\t', index=False)
    