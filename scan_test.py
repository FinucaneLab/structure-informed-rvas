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
    print('getting adjacency_matrix')
    adjacency_matrix = get_adjacency_matrix(pdb_file, radius)
    n_res = adjacency_matrix.shape[0]
    print('getting allele counts and permuting')
    case_ac_matrix, control_ac_matrix = get_case_control_ac_matrix(df, n_res, n_sim)
    n_case = case_ac_matrix[:,0].sum()
    n_control = control_ac_matrix[:,0].sum()
    print('aggregating counts in neighborhoods')
    n_case_nbhd_mat = get_nbhd_counts(adjacency_matrix, case_ac_matrix)
    n_control_nbhd_mat = get_nbhd_counts(adjacency_matrix, control_ac_matrix)
    print('precomputing pvals')
    pval_lookup = get_pval_lookup_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_control)
    print('computing pvals')
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
    df_results = df_pvals[['aa_pos', 'p_value', 'fdr']]
    print(df_results[0:20])
    return df_results

def scan_test_one_protein(df, pdb_file, results_df_path, radius, n_sims):
    df_pvals = get_all_pvals(df, pdb_file, n_sims, radius)
    df_results = compute_fdr(df_pvals)
    df_results.to_csv(results_df_path, sep='\t', index=False)

def scan_test(df_rvas, reference_dir, radius, results_dir, n_sims=1000):
    '''
    df_rvas is the output of map_to_protein. reference_dir has the pdb structures. this function
    should perform the scan test for all proteins and return a data frame with all the results.
    '''
    uniprot_id_list = np.unique(df_rvas.uniprot_id)
    for uniprot_id in uniprot_id_list:
        df = df_rvas[df_rvas.uniprot_id == uniprot_id]
        if len(np.unique(df.pdb_filename)) > 1:
            print('only using one pdb file for now')
        pdb_filename = np.unique(df.pdb_filename)[0]
        df = df[df.pdb_filename == pdb_filename].reset_index(drop=True)
        full_pdb_filename = f'{reference_dir}/pdb_files/{pdb_filename}'
        results_df_path = f'{results_dir}/{uniprot_id}.results.tsv'
        scan_test_one_protein(df, full_pdb_filename, results_df_path, radius, n_sims)