"""
Empirical False Discovery Rate (FDR) and Family-Wise Error Rate (FWER) correction for multiple testing.

This module implements FDR and FWER correction using null distributions from permutation testing,
specifically designed for structure-informed rare variant association studies.
"""

import os
import numpy as np
import pandas as pd
import h5py
from utils import read_stats
from logger_config import get_logger

logger = get_logger(__name__)


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


def _load_all_stats(results_dir, uniprot_filter_list, aa_pos_filters, pval_file):
    """Load both observed and null p-values from HDF5 file with consistent filtering."""
    to_concat = []
    null_stats_dict = {}
    n_sims = None
    
    with h5py.File(os.path.join(results_dir, pval_file), 'a') as fid:
        uniprot_ids = [k for k in fid.keys() if '_' not in k]
        if uniprot_filter_list is not None:
            uniprot_ids = list(set(uniprot_ids) & set(uniprot_filter_list))

        logger.info('Reading observed and null statistics (p-values or t-stats)')
        for uniprot_id in uniprot_ids:
            # Load observed p-values
            df = read_stats(fid, uniprot_id)
            
            # Load null p-values
            null_stat_one_uniprot = fid[f'{uniprot_id}_null_n_a_tstat'][:]
            
            # Apply same amino acid position filter to both datasets
            if aa_pos_filters is not None and uniprot_id in aa_pos_filters:
                aa_pos_keep = aa_pos_filters[uniprot_id]
                # Create boolean mask: positions are 1-indexed, dataframe indices are 0-indexed
                mask = np.array([x+1 in aa_pos_keep for x in range(len(df))])
                df = df[mask]
                null_stat_one_uniprot = null_stat_one_uniprot[mask, :]
            
            to_concat.append(df)
            null_stats_dict[uniprot_id] = null_stat_one_uniprot
            
            # Get n_sims from first protein (all should have same value)
            if n_sims is None:
                n_sims = null_stat_one_uniprot.shape[1]
        
    # Ensure we have data to process
    if not to_concat:
        raise ValueError("No proteins found for FDR computation. Check filters and input data.")
    
    logger.info('Concatenating and sorting observed statistics (p-values or t-stats)')
    df_stats = pd.concat(to_concat)
    #logger.info(f'Columns in df_stats: {df_stats.columns}')
    df_stats = df_stats.sort_values(by='n_a_tstat').reset_index(drop=True)
    
    return df_stats, null_stats_dict, uniprot_ids, n_sims


def _q_compute_false_discoveries(df_stats, null_stats_dict, uniprot_ids, n_sims, large_threshold=-2.0):
    """Compute false discovery statistics from null distributions."""
    logger.info('Computing false discoveries for FDR')
    mask = df_stats.n_a_tstat <= large_threshold
    
    # Aggregate ALL null p-values across all proteins
    null_stats = []
    for i, uniprot_id in enumerate(uniprot_ids):
        if len(uniprot_ids) > 100 and i % 100 == 0:
            logger.debug(f'Processing null stats from protein {i} out of {len(uniprot_ids)}')
        
        null_stat_one_uniprot = null_stats_dict[uniprot_id]
        null_stat_one_uniprot = null_stat_one_uniprot.flatten()
        null_stats.extend(null_stat_one_uniprot[null_stat_one_uniprot < large_threshold])
    
    # Sort the complete null distribution once
    null_stats = np.sort(np.array(null_stats))
    
    # Compute FDRs using the complete null distribution
    false_discoveries = np.empty(len(df_stats.n_a_tstat))
    
    if np.any(mask):
        false_discoveries[mask] = np.searchsorted(null_stats, df_stats.n_a_tstat[mask], side='right') / n_sims
    if np.any(~mask):
        false_discoveries[~mask] = df_stats.shape[0]
    
    return false_discoveries


def _q_compute_fwer(df_stats, null_stats_dict, uniprot_ids, n_sims, chunk_size=50):
    """
    Compute empirical FWER by processing proteins in chunks.
    
    Args:
        df_stats: DataFrame with observed p-values
        null_stats_dict: Dictionary mapping protein IDs to null p-value arrays
        uniprot_ids: List of protein IDs
        n_sims: Number of simulations
        chunk_size: Number of proteins to process at once (default 50)
        
    Returns:
        Array of FWER values
    """
    logger.info('Computing empirical FWER')
    
    # Initialize with ones (worst case p-value)
    min_stats_per_sim = np.ones(n_sims)
    
    # Process proteins in chunks
    for chunk_start in range(0, len(uniprot_ids), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(uniprot_ids))
        chunk_proteins = uniprot_ids[chunk_start:chunk_end]
        
        # Stack null p-values for this chunk of proteins
        null_stats_chunk = np.vstack([null_stats_dict[uid] for uid in chunk_proteins])
        
        # Find minimum p-value per simulation for this chunk
        min_pvals_chunk = np.min(null_stats_chunk, axis=0)
        
        # Update global minimums
        min_stats_per_sim = np.minimum(min_stats_per_sim, min_pvals_chunk)
    
    # Compute FWER for each observed p-value / t-statistic
    p_obs = df_stats['n_a_tstat'].values.reshape(-1, 1)
    min_stats_per_sim = min_stats_per_sim.reshape(1, -1)
    fwer = np.mean(min_stats_per_sim <= p_obs, axis=1)
    
    return fwer


def _q_apply_corrections(df_stats, false_discoveries, fwer):
    """Apply FDR and FWER corrections and format results."""
    logger.info('Applying FDR and FWER corrections')
    
    # Apply FDR correction
    df_stats['false_discoveries_avg'] = false_discoveries
    df_stats['fdr'] = [x / (i+1) for i, x in enumerate(false_discoveries)]
    df_stats['fdr'] = df_stats['fdr'][::-1].cummin()[::-1]
    
    # Add FWER
    df_stats['fwer'] = fwer
    
    return df_stats[['uniprot_id', 'aa_pos', 'n_a_tstat', 'fdr', 'fwer', 'mean_betahat', 'n_betahat']]


def summarize_results(df_results, fdr_cutoff, fwer_cutoff=0.05):
    """Summarize results with both FDR and FWER significance."""
    top_hits_all_genes = df_results.loc[df_results.groupby('uniprot_id')['fdr'].idxmin()]
    top_hits_fdr_sig = top_hits_all_genes[top_hits_all_genes.fdr < fdr_cutoff]
    top_hits_fdr_sig = top_hits_fdr_sig.sort_values(by='n_a_tstat')
    
    top_hits_fwer_sig = top_hits_all_genes[top_hits_all_genes.fwer < fwer_cutoff]
    top_hits_fwer_sig = top_hits_fwer_sig.sort_values(by='n_a_tstat')
    
    logger.info('')
    logger.info(f'{len(top_hits_fdr_sig)} out of {len(top_hits_all_genes)} proteins have a neighborhood significant at FDR < {fdr_cutoff}')
    logger.info(f'{len(top_hits_fwer_sig)} out of {len(top_hits_all_genes)} proteins have a neighborhood significant at FWER < {fwer_cutoff}')
    
    if len(top_hits_fdr_sig) > 0:
        logger.info(f'Top 20 FDR-significant hits:\n{top_hits_fdr_sig[0:20].to_string()}')
    
    if len(top_hits_fwer_sig) > 0:
        logger.info(f'Top FWER-significant hits:\n{top_hits_fwer_sig.to_string()}')


def q_compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file, large_threshold=0.05):
    """
    Compute False Discovery Rate and Family-Wise Error Rate corrections for scan test results.
    
    Main orchestration function that coordinates the FDR and FWER computation workflow using
    empirical null distributions from permutation testing.
    
    Args:
        results_dir: Directory containing HDF5 results file with observed and null p-values
        fdr_cutoff: FDR threshold for significance
        df_fdr_filter: Optional DataFrame to filter proteins and positions
        reference_dir: Directory with reference files for result annotation
        pval_file: HDF5 filename containing p-values
        large_threshold: P-value threshold for computational efficiency (default 0.05)
        
    Returns:
        DataFrame with FDR and FWER corrected results
    """
    logger.info('Computing FDR and FWER')
    
    # Prepare filtering criteria
    uniprot_filter_list, aa_pos_filters = _prepare_fdr_filters(df_fdr_filter)
    
    # Load both observed and null p-values
    df_stats, null_stats_dict, uniprot_ids, n_sims = _load_all_stats(
        results_dir, uniprot_filter_list, aa_pos_filters, pval_file
    )
    
    # Compute false discoveries from null distributions (for FDR)
    false_discoveries = _q_compute_false_discoveries(
        df_stats, null_stats_dict, uniprot_ids, n_sims, large_threshold
    )
    
    # Compute FWER from null distributions
    fwer = _q_compute_fwer(df_stats, null_stats_dict, uniprot_ids, n_sims)
    
    # Apply both corrections
    df_results = _q_apply_corrections(df_stats, false_discoveries, fwer)
    
    # Add gene name
    df_gene = pd.read_csv(f'{reference_dir}/protein_sequence_guide.tsv', sep='\t')
    df_results = df_results.merge(df_gene[['gene_name', 'uniprot_id']], how='left', on='uniprot_id')
    
    # Summarize and return results
    summarize_results(df_results, fdr_cutoff)
    
    return df_results