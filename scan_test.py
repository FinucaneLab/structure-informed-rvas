import os
import pandas as pd
import numpy as np
import glob
import bisect
import h5py
from scipy.stats import fisher_exact, binom
from utils import get_adjacency_matrix, valid_for_fisher, write_dataset, read_p_values
    
def get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_ctrl):
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
    
    print('getting case control ac matrix')
    case_ac_matrix, control_ac_matrix = get_case_control_ac_matrix(df, n_res, n_sims)
    print('done')
    n_case = case_ac_matrix[:,0].sum()
    n_control = control_ac_matrix[:,0].sum()
    print('getting nbhd counts')
    n_case_nbhd_mat = get_nbhd_counts(adjacency_matrix, case_ac_matrix)
    n_control_nbhd_mat = get_nbhd_counts(adjacency_matrix, control_ac_matrix)
    print('getting pval lookup')
    pval_lookup = get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_control)
    print('getting pval matrix')
    pval_matrix = pval_lookup[n_case_nbhd_mat, n_control_nbhd_mat]
    print('done')
    pval_columns = ['p_value'] + [f'null_pval_{i}' for i in range(n_sims)]
    df_pvals = pd.DataFrame(columns = pval_columns, data = pval_matrix)
    df_pvals['nbhd_case'] = n_case_nbhd_mat[:,0]
    df_pvals['nbhd_control'] = n_control_nbhd_mat[:,0]
    df_pvals['ratio'] = (df_pvals['nbhd_case'] + 2) / (df_pvals['nbhd_control'] + 2 * n_control / n_case)
    df_pvals = df_pvals[['nbhd_case', 'nbhd_control'] + pval_columns + ['ratio']]
    return df_pvals, adjacency_matrix

def compute_fdr(results_dir, large_p_threshold = 0.05):
    to_concat = []
    with h5py.File(os.path.join(results_dir, 'p_values.h5'), 'a') as fid:
        uniprot_ids = [k for k in fid.keys() if '_' not in k]
        for uniprot_id in uniprot_ids:
            df = read_p_values(fid, uniprot_id)
            to_concat.append(df)
        df_pvals = pd.concat(to_concat)
        to_concat = None
        df_pvals = df_pvals.sort_values(by='p_value').reset_index(drop=True)
        
        false_discoveries_avg = np.zeros(df_pvals.shape[0])
        num_large_p = 0
        for uniprot_id in uniprot_ids:
            null_pvals = fid[f'{uniprot_id}_null_pval'][:].flatten()
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
    results_prefix = os.path.join(results_dir, 'uniprot_id')
    df_pvals, adj_mat = get_all_pvals(df, pdb_file, n_sims, radius)
    print('saving adj mat')
    np.save(f'{results_prefix}.adj_mat.npy', adj_mat)
    print('printing df_rvas')
    df.to_csv(f'{results_prefix}.df_rvas.tsv', sep='\t', index=False)
    write_df_pvals(results_dir, uniprot_id, df_pvals)

def scan_test(df_rvas, reference_dir, radius, results_dir, n_sims, no_fdr):
    '''
    df_rvas is the output of map_to_protein. reference_dir has the pdb structures. this function
    should perform the scan test for all proteins and return a data frame with all the results.
    '''
    uniprot_id_list = np.unique(df_rvas.uniprot_id)
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
        df_results = compute_fdr(results_dir)
        df_results.to_csv(os.path.join(results_dir, 'all_proteins.fdr.tsv'), sep='\t', index=False)
    