import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, binom
from utils import get_adjacency_matrix, valid_for_fisher
    
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
                # _, p, _, _ = chi2_contingency(contingency_table)
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
        n_sim = 1000,
        radius = 15,
):
    adjacency_matrix = get_adjacency_matrix(pdb_file, radius)
    n_res = adjacency_matrix.shape[0]
    case_ac_matrix, control_ac_matrix = get_case_control_ac_matrix(df, n_res, n_sim)
    n_case = case_ac_matrix[:,0].sum()
    n_control = control_ac_matrix[:,0].sum()
    n_case_nbhd_mat = get_nbhd_counts(adjacency_matrix, case_ac_matrix)
    n_control_nbhd_mat = get_nbhd_counts(adjacency_matrix, control_ac_matrix)
    pval_lookup = get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_control)
    pval_matrix = pval_lookup[n_case_nbhd_mat, n_control_nbhd_mat]
    pval_columns = ['p_value'] + [f'null_pval_{i}' for i in range(n_sim)]
    df_pvals = pd.DataFrame(columns = pval_columns, data = pval_matrix)
    return df_pvals

def compute_fdr(df_pvals):
    null_ps = df_pvals.iloc[:, 1:].values
    n_sim = len(df_pvals.columns) - 1
    df_pvals['aa_pos'] = 1 + np.arange(len(df_pvals))
    df_pvals = df_pvals.sort_values(by='p_value').reset_index(drop=True)
    false_discoveries_avg = [np.sum(null_ps <= p)/n_sim for p in df_pvals.p_value]
    df_pvals['false_discoveries_avg'] = false_discoveries_avg
    df_pvals['fdr'] = [x / (i+1) for i, x in enumerate(false_discoveries_avg)]
    df_pvals['fdr'] = df_pvals['fdr'][::-1].cummin()[::-1]
    df_results = df_pvals[['aa_pos', 'p_value', 'fdr']]
    return df_results

def scan_test_one_protein(df, pdb_file, results_df_path, radius, n_sims):
    df_pvals = get_all_pvals(df, pdb_file, n_sims, radius)
    df_results = compute_fdr(df_pvals)
    df_results.to_csv(results_df_path, sep='\t', index=False)
    return df_results

def scan_test(df_rvas, reference_dir, radius, results_dir, n_sims=1000):
    '''
    df_rvas is the output of map_to_protein. reference_dir has the pdb structures. this function
    should perform the scan test for all proteins and return a data frame with all the results.
    '''
    uniprot_id_list = np.unique(df_rvas.uniprot_id)
    n_proteins = len(uniprot_id_list)
    min_fdr_filename = f'{results_dir}/min_fdr.tsv'
    with open(min_fdr_filename, 'w') as min_fdr_file:
        for i, uniprot_id in enumerate(uniprot_id_list):
            print('\n', uniprot_id, f'number {i} out of {n_proteins}')
            try:
                df = df_rvas[df_rvas.uniprot_id == uniprot_id]
                if len(np.unique(df.pdb_filename)) > 1:
                    print('skipping when there is more than one pdb file')
                    continue
                pdb_filename = np.unique(df.pdb_filename)[0]
                df = df[df.pdb_filename == pdb_filename].reset_index(drop=True)
                full_pdb_filename = f'{reference_dir}/pdb_files/{pdb_filename}'
                if not os.path.exists(full_pdb_filename):
                    print('missing pdb file. skipping.')
                    continue
                results_df_path = f'{results_dir}/{uniprot_id}.scan_test.results.tsv'
                df_results = scan_test_one_protein(df, full_pdb_filename, results_df_path, radius, n_sims)
                min_fdr = df_results.fdr.min()
                print(f'min fdr: {min_fdr}')
                min_fdr_file.write(f'{uniprot_id}\t{min_fdr}\n')
            except Exception as e:
                print(f'Error for {uniprot_id}: {e}')
                continue