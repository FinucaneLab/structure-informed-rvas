"""
Empirical False Discovery Rate (FDR) correction for multiple testing.

This module implements FDR correction using null distributions from permutation testing,
specifically designed for structure-informed rare variant association studies.
"""

import os
import gc
import numpy as np
import pandas as pd
import h5py
from utils import read_p_values, read_p_values_quantitative
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


def _load_all_pvalues(results_dir, uniprot_filter_list, aa_pos_filters, pval_file='p_values.h5', quantitative=False):
    """Load both observed and null p-values from HDF5 file with consistent filtering using batch processing."""
    batch_size = 500  # Process proteins in batches to manage memory
    null_pvals_dict = {}
    n_sims = None

    with h5py.File(os.path.join(results_dir, pval_file), 'a') as fid:
        uniprot_ids = [k for k in fid.keys() if '_' not in k]
        if uniprot_filter_list is not None:
            uniprot_ids = list(set(uniprot_ids) & set(uniprot_filter_list))

        logger.info(f'Reading observed and null p-values for {len(uniprot_ids)} proteins in batches of {batch_size}')

        # First pass: collect all observed p-values in batches
        all_dfs = []
        for i in range(0, len(uniprot_ids), batch_size):
            batch_ids = uniprot_ids[i:i + batch_size]
            logger.info(f'Processing batch {i//batch_size + 1}/{(len(uniprot_ids) + batch_size - 1)//batch_size}: proteins {i+1}-{min(i+batch_size, len(uniprot_ids))}')

            batch_dfs = []
            for uniprot_id in batch_ids:
                # Load observed p-values
                if quantitative:
                    df = read_p_values_quantitative(fid, uniprot_id)
                else:
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

                batch_dfs.append(df)
                null_pvals_dict[uniprot_id] = null_pvals_one_uniprot

                # Get n_sims from first protein (all should have same value)
                if n_sims is None:
                    n_sims = null_pvals_one_uniprot.shape[1]

            # Concatenate this batch and add to list
            if batch_dfs:
                batch_concat = pd.concat(batch_dfs, ignore_index=True)
                all_dfs.append(batch_concat)

                # Clean up batch data
                del batch_dfs
                gc.collect()

    # Ensure we have data to process
    if not all_dfs:
        raise ValueError("No proteins found for FDR computation. Check filters and input data.")

    logger.info('Concatenating and sorting all observed p-values')
    df_pvals = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    df_pvals = df_pvals.sort_values(by='p_value').reset_index(drop=True)

    return df_pvals, null_pvals_dict, uniprot_ids, n_sims


def _compute_false_discoveries(df_pvals, null_pvals_dict, uniprot_ids, n_sims, large_p_threshold=0.05):
    """Compute false discovery statistics from null distributions with memory optimization."""
    logger.info('Computing false discoveries')
    mask = df_pvals.p_value <= large_p_threshold

    # Process null p-values in batches to avoid memory overflow
    logger.info('Aggregating null p-values from all proteins in memory-efficient batches')
    batch_size = 500
    all_null_pvals = []

    for i in range(0, len(uniprot_ids), batch_size):
        batch_ids = uniprot_ids[i:i + batch_size]
        logger.info(f'Processing null p-values batch {i//batch_size + 1}/{(len(uniprot_ids) + batch_size - 1)//batch_size}')

        batch_null_pvals = []
        for uniprot_id in batch_ids:
            null_pvals_one_uniprot = null_pvals_dict[uniprot_id]
            null_pvals_one_uniprot = null_pvals_one_uniprot.flatten()
            significant_nulls = null_pvals_one_uniprot[null_pvals_one_uniprot < large_p_threshold]
            batch_null_pvals.append(significant_nulls)

        # Concatenate batch and add to list
        if batch_null_pvals:
            batch_concat = np.concatenate(batch_null_pvals)
            all_null_pvals.append(batch_concat)

            # Clean up batch data
            del batch_null_pvals
            gc.collect()

    # Concatenate all batches and sort once
    logger.info('Concatenating and sorting all null p-values')
    null_pvals = np.concatenate(all_null_pvals)
    del all_null_pvals
    gc.collect()

    logger.info(f'Sorting {len(null_pvals)} null p-values')
    null_pvals = np.sort(null_pvals)
    
    # Second loop: Compute FDRs using the complete null distribution
    logger.debug('Computing false discovery rates')
    false_discoveries = np.empty(len(df_pvals.p_value))
    
    if np.any(mask):
        false_discoveries[mask] = np.searchsorted(null_pvals, df_pvals.p_value[mask], side='right') / n_sims
    if np.any(~mask):
        false_discoveries[~mask] = df_pvals.shape[0]

    # Clean up large null_pvals array
    del null_pvals
    gc.collect()

    return false_discoveries


def _apply_fdr_correction(df_pvals, false_discoveries, quantitative=False):
    """Apply FDR correction and format results."""
    logger.info('Computing FDR')
    df_pvals['false_discoveries_avg'] = false_discoveries
    df_pvals['fdr'] = [x / (i+1) for i, x in enumerate(false_discoveries)]
    df_pvals['fdr'] = df_pvals['fdr'][::-1].cummin()[::-1]
    
    if quantitative:
        return df_pvals[['uniprot_id', 'aa_pos', 'p_value', 'fdr', 't_stat', 'mean_beta_in', 'mean_beta_out', 'std_beta_in', 'std_beta_out', 'n_variants_in', 'n_variants_out']]
    else:
        return df_pvals[['uniprot_id', 'aa_pos', 'p_value', 'fdr', 'nbhd_case', 'nbhd_control', 'ratio']]

def summarize_results(df_results, fdr_cutoff):

    top_hits_all_genes = df_results.loc[df_results.groupby('uniprot_id')['fdr'].idxmin()]
    top_hits_sig = top_hits_all_genes[top_hits_all_genes.fdr<fdr_cutoff]
    top_hits_sig = top_hits_sig.sort_values(by='p_value')
    logger.info('')
    logger.info(f'{len(top_hits_sig)} out of {len(top_hits_all_genes)} proteins have a neighborhood significant at {fdr_cutoff}.')
    logger.info(f'Top 20 hits:\n{top_hits_sig[0:20].to_string()}')

def compute_fdr(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, large_p_threshold=0.05, pval_file='p_values.h5', quantitative=False):
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
        results_dir, uniprot_filter_list, aa_pos_filters, pval_file, quantitative
    )
    
    # Compute false discoveries from null distributions
    false_discoveries = _compute_false_discoveries(
        df_pvals, null_pvals_dict, uniprot_ids, n_sims, large_p_threshold
    )

    # Clean up large null_pvals_dict to free memory
    del null_pvals_dict
    gc.collect()

    # Apply FDR correction
    df_results = _apply_fdr_correction(df_pvals, false_discoveries, quantitative)
    
    # Add gene name
    df_gene = pd.read_csv(f'{reference_dir}/gene_to_uniprot_id.tsv', sep='\t')
    df_results = df_results.merge(df_gene, how='left', on='uniprot_id')

    # Summarize and return results
    summarize_results(df_results, fdr_cutoff)
    
    return df_results