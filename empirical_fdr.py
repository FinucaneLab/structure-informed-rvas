"""
Empirical False Discovery Rate (FDR) correction for multiple testing.

This module implements FDR correction using null distributions from permutation testing,
specifically designed for structure-informed rare variant association studies.
"""

import os
import numpy as np
import pandas as pd
import h5py
from utils import read_p_values
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


def _load_all_pvalues(results_dir, uniprot_filter_list, aa_pos_filters):
    """Load both observed and null p-values from HDF5 file with consistent filtering."""
    to_concat = []
    null_pvals_dict = {}
    n_sims = None
    
    with h5py.File(os.path.join(results_dir, 'p_values.h5'), 'a') as fid:
        uniprot_ids = [k for k in fid.keys() if '_' not in k]
        if uniprot_filter_list is not None:
            uniprot_ids = list(set(uniprot_ids) & set(uniprot_filter_list))

        logger.info('Reading observed and null p-values')
        for uniprot_id in uniprot_ids:
            # Load observed p-values
            df = read_p_values(fid, uniprot_id)
            
            # Load null p-values
            null_pvals_one_uniprot = fid[f'{uniprot_id}_null_pval'][:]
            
            # Apply same amino acid position filter to both datasets
            if aa_pos_filters is not None and uniprot_id in aa_pos_filters:
                aa_pos_keep = aa_pos_filters[uniprot_id]
                # Create boolean mask: positions are 1-indexed, dataframe indices are 0-indexed
                mask = np.array([x+1 in aa_pos_keep for x in range(len(df))])
                df = df[mask]
                null_pvals_one_uniprot = null_pvals_one_uniprot[mask, :]
            
            to_concat.append(df)
            null_pvals_dict[uniprot_id] = null_pvals_one_uniprot
            
            # Get n_sims from first protein (all should have same value)
            if n_sims is None:
                n_sims = null_pvals_one_uniprot.shape[1]
        
    # Ensure we have data to process
    if not to_concat:
        raise ValueError("No proteins found for FDR computation. Check filters and input data.")
    
    logger.info('Concatenating and sorting observed p-values')
    df_pvals = pd.concat(to_concat)
    df_pvals = df_pvals.sort_values(by='p_value').reset_index(drop=True)
    
    return df_pvals, null_pvals_dict, uniprot_ids, n_sims


def _compute_false_discoveries(df_pvals, null_pvals_dict, uniprot_ids, n_sims, large_p_threshold=0.05):
    """Compute false discovery statistics from null distributions."""
    logger.info('Computing false discoveries')
    mask = df_pvals.p_value <= large_p_threshold
    
    # First loop: Aggregate ALL null p-values across all proteins
    logger.debug('Aggregating null p-values from all proteins')
    null_pvals = []
    for i, uniprot_id in enumerate(uniprot_ids):
        if len(uniprot_ids) > 100 and i % 100 == 0:
            logger.debug(f'Processing null p-values from protein {i} out of {len(uniprot_ids)}')
        
        null_pvals_one_uniprot = null_pvals_dict[uniprot_id]
        null_pvals_one_uniprot = null_pvals_one_uniprot.flatten()
        null_pvals.extend(null_pvals_one_uniprot[null_pvals_one_uniprot < large_p_threshold])
    
    # Sort the complete null distribution once
    logger.debug(f'Sorting {len(null_pvals)} null p-values')
    null_pvals = np.sort(np.array(null_pvals))
    
    # Second loop: Compute FDRs using the complete null distribution
    logger.debug('Computing false discovery rates')
    false_discoveries = np.empty(len(df_pvals.p_value))
    
    if np.any(mask):
        false_discoveries[mask] = np.searchsorted(null_pvals, df_pvals.p_value[mask], side='right') / n_sims
    if np.any(~mask):
        false_discoveries[~mask] = df_pvals.shape[0]
    
    return false_discoveries


def _apply_fdr_correction(df_pvals, false_discoveries):
    """Apply FDR correction and format results."""
    logger.info('Computing FDR')
    df_pvals['false_discoveries_avg'] = false_discoveries
    df_pvals['fdr'] = [x / (i+1) for i, x in enumerate(false_discoveries)]
    df_pvals['fdr'] = df_pvals['fdr'][::-1].cummin()[::-1]
    
    return df_pvals[['uniprot_id', 'aa_pos', 'p_value', 'fdr', 'nbhd_case', 'nbhd_control', 'ratio']]

def summarize_results(df_results, fdr_cutoff):

    top_hits_all_genes = df_results.loc[df_results.groupby('uniprot_id')['fdr'].idxmin()]
    top_hits_sig = top_hits_all_genes[top_hits_all_genes.fdr<fdr_cutoff]
    top_hits_sig = top_hits_sig.sort_values(by='p_value')
    logger.info('')
    logger.info(f'{len(top_hits_sig)} out of {len(top_hits_all_genes)} proteins have a neighborhood significant at {fdr_cutoff}.')
    logger.info(f'Top 20 hits:\n{top_hits_sig[0:20].to_string()}')

def compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, large_p_threshold=0.05):
    """
    Compute False Discovery Rate correction for scan test results.
    
    Main orchestration function that coordinates the FDR computation workflow using
    empirical null distributions from permutation testing.
    
    Args:
        results_dir: Directory containing HDF5 results file with observed and null p-values
        fdr_cutoff: FDR threshold for significance
        df_fdr_filter: Optional DataFrame to filter proteins and positions
        reference_dir: Directory with reference files for result annotation
        large_p_threshold: P-value threshold for computational efficiency (default 0.05)
        
    Returns:
        DataFrame with FDR-corrected results
    """
    logger.info('Computing FDR')
    
    # Prepare filtering criteria
    uniprot_filter_list, aa_pos_filters = _prepare_fdr_filters(df_fdr_filter)
    
    # Load both observed and null p-values
    df_pvals, null_pvals_dict, uniprot_ids, n_sims = _load_all_pvalues(
        results_dir, uniprot_filter_list, aa_pos_filters
    )
    
    # Compute false discoveries from null distributions
    false_discoveries = _compute_false_discoveries(
        df_pvals, null_pvals_dict, uniprot_ids, n_sims, large_p_threshold
    )
    
    # Apply FDR correction
    df_results = _apply_fdr_correction(df_pvals, false_discoveries)
    
    # Add gene name
    df_gene = pd.read_csv(f'{reference_dir}/gene_to_uniprot_id.tsv', sep='\t')
    df_results = df_results.merge(df_gene, how='left', on='uniprot_id')

    # Summarize and return results
    summarize_results(df_results, fdr_cutoff)
    
    return df_results