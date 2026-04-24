"""
Empirical False Discovery Rate (FDR) and Family-Wise Error Rate (FWER) correction for multiple testing.

This module implements FDR and FWER correction using null distributions from permutation testing,
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
        grouped = df_fdr_filter.groupby('uniprot_id')['aa_pos'].apply(set)
        aa_pos_filters = grouped.to_dict()
    
    return uniprot_filter_list, aa_pos_filters


def _load_all_pvalues(results_dir, uniprot_filter_list, aa_pos_filters, pval_file):
    """Load both observed and null p-values from HDF5 file with consistent filtering."""
    to_concat = []
    null_pvals_dict = {}
    n_sims = None
    
    with h5py.File(os.path.join(results_dir, pval_file), 'a') as fid:
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
    logger.info('Computing false discoveries for FDR')
    mask = df_pvals.p_value <= large_p_threshold
    
    # Aggregate ALL null p-values across all proteins
    null_pvals = []
    for i, uniprot_id in enumerate(uniprot_ids):
        if len(uniprot_ids) > 100 and i % 100 == 0:
            logger.debug(f'Processing null p-values from protein {i} out of {len(uniprot_ids)}')
        
        null_pvals_one_uniprot = null_pvals_dict[uniprot_id]
        null_pvals_one_uniprot = null_pvals_one_uniprot.flatten()
        null_pvals.extend(null_pvals_one_uniprot[null_pvals_one_uniprot < large_p_threshold])
    
    # Sort the complete null distribution once
    null_pvals = np.sort(np.array(null_pvals))
    
    # Compute FDRs using the complete null distribution
    false_discoveries = np.empty(len(df_pvals.p_value))
    
    if np.any(mask):
        false_discoveries[mask] = np.searchsorted(null_pvals, df_pvals.p_value[mask], side='right') / n_sims
    if np.any(~mask):
        false_discoveries[~mask] = df_pvals.shape[0]
    
    return false_discoveries


def _compute_fwer(df_pvals, null_pvals_dict, uniprot_ids, n_sims, chunk_size=50):
    """
    Compute empirical FWER by processing proteins in chunks.
    
    Args:
        df_pvals: DataFrame with observed p-values
        null_pvals_dict: Dictionary mapping protein IDs to null p-value arrays
        uniprot_ids: List of protein IDs
        n_sims: Number of simulations
        chunk_size: Number of proteins to process at once (default 50)
        
    Returns:
        Array of FWER values
    """
    logger.info('Computing empirical FWER')
    
    # Initialize with ones (worst case p-value)
    min_pvals_per_sim = np.ones(n_sims)
    
    # Process proteins in chunks
    for chunk_start in range(0, len(uniprot_ids), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(uniprot_ids))
        chunk_proteins = uniprot_ids[chunk_start:chunk_end]
        
        # Stack null p-values for this chunk of proteins
        null_pvals_chunk = np.vstack([null_pvals_dict[uid] for uid in chunk_proteins])
        
        # Find minimum p-value per simulation for this chunk
        min_pvals_chunk = np.min(null_pvals_chunk, axis=0)
        
        # Update global minimums
        min_pvals_per_sim = np.minimum(min_pvals_per_sim, min_pvals_chunk)
    
    # Compute FWER for each observed p-value
    p_obs = df_pvals['p_value'].values.reshape(-1, 1)
    min_pvals_per_sim = min_pvals_per_sim.reshape(1, -1)
    fwer = np.mean(min_pvals_per_sim <= p_obs, axis=1)
    
    return fwer


def _apply_corrections(df_pvals, false_discoveries, fwer):
    """Apply FDR and FWER corrections and format results."""
    logger.info('Applying FDR and FWER corrections')
    
    # Apply FDR correction
    df_pvals['false_discoveries_avg'] = false_discoveries
    df_pvals['fdr'] = [x / (i+1) for i, x in enumerate(false_discoveries)]
    df_pvals['fdr'] = df_pvals['fdr'][::-1].cummin()[::-1]
    
    # Add FWER
    df_pvals['fwer'] = fwer
    
    return df_pvals[['uniprot_id', 'aa_pos', 'p_value', 'fdr', 'fwer', 'nbhd_case', 'nbhd_control', 'ratio', 'radius']]


def summarize_results(df_results, fdr_cutoff, fwer_cutoff=0.05):
    """Summarize results with both FDR and FWER significance."""
    top_hits_all_genes = df_results.loc[df_results.groupby('uniprot_id')['fdr'].idxmin()]
    top_hits_fdr_sig = top_hits_all_genes[top_hits_all_genes.fdr < fdr_cutoff]
    top_hits_fdr_sig = top_hits_fdr_sig.sort_values(by='p_value')
    
    top_hits_fwer_sig = top_hits_all_genes[top_hits_all_genes.fwer < fwer_cutoff]
    top_hits_fwer_sig = top_hits_fwer_sig.sort_values(by='p_value')
    
    logger.info('')
    logger.info(f'{len(top_hits_fdr_sig)} out of {len(top_hits_all_genes)} proteins have a neighborhood significant at FDR < {fdr_cutoff}')
    logger.info(f'{len(top_hits_fwer_sig)} out of {len(top_hits_all_genes)} proteins have a neighborhood significant at FWER < {fwer_cutoff}')
    
    if len(top_hits_fdr_sig) > 0:
        logger.info(f'Top 20 FDR-significant hits:\n{top_hits_fdr_sig[0:20].to_string()}')
    
    if len(top_hits_fwer_sig) > 0:
        logger.info(f'Top FWER-significant hits:\n{top_hits_fwer_sig.to_string()}')


def compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file, large_p_threshold=0.05):
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
        large_p_threshold: P-value threshold for computational efficiency (default 0.05)
        
    Returns:
        DataFrame with FDR and FWER corrected results
    """
    logger.info('Computing FDR and FWER')
    
    # Prepare filtering criteria
    uniprot_filter_list, aa_pos_filters = _prepare_fdr_filters(df_fdr_filter)
    
    # Load both observed and null p-values
    df_pvals, null_pvals_dict, uniprot_ids, n_sims = _load_all_pvalues(
        results_dir, uniprot_filter_list, aa_pos_filters, pval_file
    )
    
    # Compute false discoveries from null distributions (for FDR)
    false_discoveries = _compute_false_discoveries(
        df_pvals, null_pvals_dict, uniprot_ids, n_sims, large_p_threshold
    )
    
    # Compute FWER from null distributions
    fwer = _compute_fwer(df_pvals, null_pvals_dict, uniprot_ids, n_sims)
    
    # Apply both corrections
    df_results = _apply_corrections(df_pvals, false_discoveries, fwer)
    
    # Add gene name
    df_gene = pd.read_csv(f'{reference_dir}/protein_sequence_guide.tsv', sep='\t')
    df_results = df_results.merge(df_gene[['gene_name', 'uniprot_id']], how='left', on='uniprot_id')
    
    # Summarize and return results
    summarize_results(df_results, fdr_cutoff)
    
    return df_results