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



# def get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim, seed=0):
#     if seed is not None:
#         np.random.seed(seed)
#     n_alleles = int(  case_ac_per_residue.sum()  )
#     gen = np.random.default_rng()
#     null_ac_per_residue = gen.multivariate_hypergeometric(total_ac_per_residue.astype(int), n_alleles, n_sim).T
#     return null_ac_per_residue


def get_nbhd_stats(adjacency_matrix, df, n_res, n_sims):
    '''
    add n_sims null_betahat_{i} columns to df by permuting betahat column
    add betahatsq columns
    then groupby aa_pos to get per-residue stats:
    1. count from df to get number of betahats per residue: L x 1
    2. sum from df to get sum_beta_per_residue: L x (n_sims + 1)
    3. sum from df to get sum_beta_squared_per_residue: L x (n_sims + 1)

    then multiply all three by the adjacency matrix to get outputs, which
    are per-neighborhood
    '''
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
    total_n_betahat = 
    total_sum_betahat = 
    total_sum_betahat_squared =
    
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
        write_dataset(fid, f'{uniprot_id}', df_n_a_tstats[['n_a_tstat']])
        write_dataset(fid, f'{uniprot_id}_null_n_a_tstat', df_n_a_tstats[null_n_a_tstat_cols])
        write_dataset(fid, f'{uniprot_id}_mean_beta', df_n_a_tstats[['mean_betahat', 'n_betahat']])

def scan_test_one_protein(df, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, uniprot_id, radius, pae_cutoff, n_sims, n_a_tstat_file):
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
    write_df_n_a_tstats(results_dir, uniprot_id, df_n_a_tstats, n_a_tstat_file)

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


def _process_proteins_batch(df_rvas, uniprot_id_list, reference_dir, radius, pae_cutoff, results_dir, n_sims, remove_nbhd, n_a_tstat_file):
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

            # add filter to at least 10 betahats
            
            scan_test_one_protein(
                df, pdb_file_pos_guide, pdb_dir, pae_dir, 
                results_dir, uniprot_id, radius, pae_cutoff, n_sims,
                n_a_tstat_file
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
    n_a_tstat_file,
    remove_nbhd,
):
    """
    Perform scan test analysis on protein structure data.
    
    Main orchestration function for the structure-informed rare variant association study.
    Processes variants across proteins and computes statistical associations with 3D neighborhoods.
    """
    
    # Handle FDR-only mode
    if fdr_only:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, n_a_tstat_file)
        df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
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
        radius, pae_cutoff, results_dir, n_sims, remove_nbhd,
        n_a_tstat_file
    )
    
    # Compute FDR if requested
    if not no_fdr:
        df_results = compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, n_a_tstat_file)
        df_results.to_csv(f'{results_dir}/{fdr_file}', sep='\t', index=False)
    
