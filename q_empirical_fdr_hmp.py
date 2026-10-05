"""
Phase 2 of the Harmonic Mean P-value (HMP) pipeline for structure-informed RVAS.

Reads the master p_values.h5 produced by q_scan_test_hmp.build_hmp_master_h5
and computes FDR and FWER corrections jointly over all (M × N) entries using
a memory-efficient two-pass streaming algorithm.

Pass 1: loads only observed statistics into memory (~1-2 GB for typical inputs).
Pass 2: streams null statistics in batches of `batch_size` entries.  Within each
        batch the null arrays are loaded one entry at a time, processed, and
        immediately discarded.  Peak memory from null data is therefore
        O(batch_size × avg_n_res × n_sims).

FDR algorithm (reverse rank accumulation):
    For each null value y, find r = searchsorted(obs_sorted, y, 'left').
    y is a false discovery for every observation obs_sorted[i] with i >= r,
    so increment counts[r].  After streaming:
        false_discoveries[i] = cumsum(counts)[i] / n_sims
    This is mathematically identical to the original searchsorted-on-sorted-pool
    approach but requires only O(N) memory instead of O(total null values).
"""

import os
import numpy as np
import pandas as pd
import h5py
from logger_config import get_logger

logger = get_logger(__name__)

_NEUTRAL = {'tstat': 0.0, 'hill': 0.0, 'pval': 1.0, 'hillp': 1.0}


def _prepare_fdr_filters(df_fdr_filter):
    """Prepare optional protein / aa_pos filtering criteria."""
    if df_fdr_filter is None:
        return None, None

    uniprot_filter_list = np.unique(df_fdr_filter['uniprot_id'])
    aa_pos_filters = None

    if 'aa_pos' in df_fdr_filter.columns:
        aa_pos_filters = {}
        for uid in uniprot_filter_list:
            aa_pos_filters[uid] = set(
                df_fdr_filter.loc[df_fdr_filter['uniprot_id'] == uid, 'aa_pos'].values
            )

    return uniprot_filter_list, aa_pos_filters


def _load_observed_stats_hmp(results_dir, uniprot_filter_list, aa_pos_filters, pval_file):
    """
    Pass 1: load only observed statistics from the master HMP HDF5 file.

    Null statistics are NOT loaded here; their shapes are inspected only to
    determine n_sims.  The returned entry_ids_loaded list contains exactly the
    entries that survived all filters and will be visited again in Pass 2.

    Returns:
        df_stats:          DataFrame [uniprot_id, cluster, aa_pos, n_a_tstat,
                                      mean_betahat, n_betahat] sorted by n_a_tstat
        entry_ids_loaded:  list of entry keys that survived filtering
        h5_path:           absolute path to the HDF5 file (reused in Pass 2)
        n_sims:            number of null simulations
    """
    to_concat = []
    entry_ids_loaded = []
    n_sims = None
    h5_path = os.path.join(results_dir, pval_file)

    with h5py.File(h5_path, 'r') as fid:
        if 'entry_ids' not in fid:
            raise ValueError(
                "Master HDF5 file missing 'entry_ids' index. "
                "Was it created by build_hmp_master_h5?"
            )
        entry_ids = [e.decode('ascii') for e in fid['entry_ids'][:]]
        logger.info(f'Pass 1: reading observed stats for {len(entry_ids)} entries')

        for eid in entry_ids:
            dot_idx = eid.index('.')
            uniprot_id = eid[:dot_idx]
            cluster = eid[dot_idx + 1:]

            if uniprot_filter_list is not None and uniprot_id not in uniprot_filter_list:
                continue

            stat_data = fid[eid][:, 0]           # (n_res,)
            beta_info = fid[f'{eid}_mean_beta'][:]  # (n_res, 2)

            aa_pos_key = f'{eid}_aa_pos'
            aa_pos = (
                fid[aa_pos_key][:].flatten()
                if aa_pos_key in fid
                else np.arange(1, len(stat_data) + 1)
            )

            # Infer n_sims from dataset shape (metadata only, no data loaded)
            if n_sims is None:
                null_key = f'{eid}_null_n_a_tstat'
                if null_key in fid:
                    n_sims = fid[null_key].shape[1]

            # Apply aa_pos filter using np.isin (robust to non-contiguous positions)
            if aa_pos_filters is not None and uniprot_id in aa_pos_filters:
                mask = np.isin(aa_pos, list(aa_pos_filters[uniprot_id]))
                stat_data = stat_data[mask]
                beta_info = beta_info[mask]
                aa_pos = aa_pos[mask]

            if len(stat_data) == 0:
                continue

            to_concat.append(pd.DataFrame({
                'uniprot_id': uniprot_id,
                'cluster': cluster,
                'aa_pos': aa_pos,
                'n_a_tstat': stat_data,
                'mean_betahat': beta_info[:, 0],
                'n_betahat': beta_info[:, 1],
            }))
            entry_ids_loaded.append(eid)

    if not to_concat:
        raise ValueError("No entries found for FDR computation. Check filters and input data.")

    logger.info(
        f'Pass 1 complete: {len(entry_ids_loaded)} entries loaded; '
        f'sorting observed stats'
    )
    df_stats = pd.concat(to_concat, ignore_index=True)
    df_stats = df_stats.sort_values('n_a_tstat').reset_index(drop=True)

    return df_stats, entry_ids_loaded, h5_path, n_sims


def _hmp_streaming_null_pass(h5_path, entry_ids_loaded, obs_sorted, aa_pos_filters,
                              large_threshold, neutral, n_sims, batch_size=50):
    """
    Pass 2: stream null statistics in batches, computing FDR counts and FWER
    minimum simultaneously.

    FDR — reverse rank accumulation:
        For null value y, r = searchsorted(obs_sorted, y, 'left') is the index
        of the first observed stat >= y.  y is a false discovery for all
        obs_sorted[i] with i >= r.  Accumulating r values via bincount and
        taking a prefix sum yields false_discoveries without storing any nulls.

    FWER — running minimum:
        min_stats_per_sim tracks the global minimum null stat per simulation
        across all entries, updated incrementally.

    Args:
        h5_path:           path to the master HDF5 file
        entry_ids_loaded:  list of entry keys to process (from Pass 1)
        obs_sorted:        sorted observed stats array, shape (N,)
        aa_pos_filters:    optional {uniprot_id: set(aa_pos)} filter
        large_threshold:   null values >= this are neutral and excluded from FDR
        neutral:           initial value for FWER min accumulator
        n_sims:            number of null simulations
        batch_size:        number of entries whose null stats are loaded per batch

    Returns:
        counts:            (N+1,) int64 rank-accumulation array
        min_stats_per_sim: (n_sims,) float minimum null stat per simulation
    """
    N = len(obs_sorted)
    counts = np.zeros(N + 1, dtype=np.int64)
    min_stats_per_sim = np.full(n_sims, neutral)

    n_entries = len(entry_ids_loaded)
    logger.info(
        f'Pass 2: streaming {n_entries} entries '
        f'(batch_size={batch_size}, n_sims={n_sims})'
    )

    with h5py.File(h5_path, 'r') as fid:
        for batch_start in range(0, n_entries, batch_size):
            batch_end = min(batch_start + batch_size, n_entries)
            if batch_start % max(batch_size, 5000) == 0:
                logger.info(f'  Pass 2: entry {batch_start}/{n_entries}')

            for eid in entry_ids_loaded[batch_start:batch_end]:
                null_key = f'{eid}_null_n_a_tstat'
                if null_key not in fid:
                    continue

                null_stat = fid[null_key][:]  # (n_res, n_sims)

                # Apply the same aa_pos filter as Pass 1
                dot_idx = eid.index('.')
                uniprot_id = eid[:dot_idx]
                if aa_pos_filters is not None and uniprot_id in aa_pos_filters:
                    aa_pos_key = f'{eid}_aa_pos'
                    aa_pos = (
                        fid[aa_pos_key][:].flatten()
                        if aa_pos_key in fid
                        else np.arange(1, null_stat.shape[0] + 1)
                    )
                    mask = np.isin(aa_pos, list(aa_pos_filters[uniprot_id]))
                    null_stat = null_stat[mask, :]

                if null_stat.shape[0] == 0:
                    continue

                # FWER: update the running per-simulation minimum
                min_stats_per_sim = np.minimum(
                    min_stats_per_sim, np.min(null_stat, axis=0)
                )

                # FDR: accumulate reverse rank counts for values at or below threshold.
                # Using <= (not <) so that neutral null values (e.g. p=1.0 for pval)
                # are counted as false discoveries for neutral observations.
                null_flat = null_stat.flatten()
                null_valid = null_flat[null_flat <= large_threshold]
                if len(null_valid) > 0:
                    r = np.searchsorted(obs_sorted, null_valid, side='left')
                    # r == N bins null values larger than all observations;
                    # they contribute to no observation's false-discovery count.
                    counts += np.bincount(r, minlength=N + 1)

    logger.info('Pass 2 complete')
    return counts, min_stats_per_sim


def _hmp_apply_corrections(df_stats, false_discoveries, fwer):
    """Apply FDR and FWER corrections."""
    logger.info('Applying FDR and FWER corrections')

    df_stats = df_stats.copy()
    df_stats['false_discoveries_avg'] = false_discoveries
    df_stats['fdr'] = [x / (i + 1) for i, x in enumerate(false_discoveries)]
    df_stats['fdr'] = df_stats['fdr'][::-1].cummin()[::-1]
    df_stats['fwer'] = fwer

    return df_stats[['uniprot_id', 'cluster', 'aa_pos', 'n_a_tstat',
                      'false_discoveries_avg', 'fdr', 'fwer', 'mean_betahat', 'n_betahat']]


def summarize_results_hmp(df_results, fdr_cutoff, fwer_cutoff=0.05):
    """Log a brief summary of significant hits."""
    top_hits = df_results.loc[
        df_results.groupby(['uniprot_id', 'cluster'])['fdr'].idxmin()
    ]
    sig_fdr = top_hits[top_hits['fdr'] < fdr_cutoff].sort_values('n_a_tstat')
    sig_fwer = top_hits[top_hits['fwer'] < fwer_cutoff].sort_values('n_a_tstat')

    logger.info('')
    logger.info(
        f'{len(sig_fdr)} / {len(top_hits)} protein-cluster pairs '
        f'significant at FDR < {fdr_cutoff}'
    )
    logger.info(
        f'{len(sig_fwer)} / {len(top_hits)} protein-cluster pairs '
        f'significant at FWER < {fwer_cutoff}'
    )
    if len(sig_fdr) > 0:
        logger.info(f'Top 20 FDR hits:\n{sig_fdr.head(20).to_string()}')
    if len(sig_fwer) > 0:
        logger.info(f'Top FWER hits:\n{sig_fwer.to_string()}')


def q_compute_fdr_hmp(results_dir, fdr_cutoff, df_fdr_filter, reference_dir, pval_file,
                       stat_method='hillp', batch_size=50):
    """
    Compute FDR and FWER corrections over the HMP master h5.

    Uses a two-pass streaming algorithm to avoid loading the full null
    distribution into memory.  Peak memory scales with O(N + batch_size ×
    avg_n_res × n_sims) rather than O(total null values).

    Args:
        results_dir:    directory containing pval_file
        fdr_cutoff:     FDR threshold for significance reporting
        df_fdr_filter:  optional DataFrame [uniprot_id (, aa_pos)] to restrict entries
        reference_dir:  directory with reference files (for gene name annotation)
        pval_file:      HDF5 filename (typically 'p_values.h5')
        stat_method:    'tstat', 'hill', 'pval', or 'hillp' — determines large_threshold
        batch_size:     entries per batch in Pass 2 (controls peak null-stat memory)

    Returns:
        DataFrame [uniprot_id, cluster, aa_pos, n_a_tstat, fdr, fwer,
                   mean_betahat, n_betahat (, gene_name)]
    """
    logger.info('HMP Phase 2: computing FDR and FWER (two-pass streaming)')

    neutral = _NEUTRAL.get(stat_method, 0.0)
    large_threshold = neutral  # strict < in null filtering

    uniprot_filter_list, aa_pos_filters = _prepare_fdr_filters(df_fdr_filter)

    # Pass 1: observed stats only
    df_stats, entry_ids_loaded, h5_path, n_sims = _load_observed_stats_hmp(
        results_dir, uniprot_filter_list, aa_pos_filters, pval_file
    )
    obs_sorted = df_stats['n_a_tstat'].values  # ascending, most significant first

    # Pass 2: stream null stats, compute FDR counts + FWER min simultaneously
    counts, min_stats_per_sim = _hmp_streaming_null_pass(
        h5_path, entry_ids_loaded, obs_sorted,
        aa_pos_filters, large_threshold, neutral, n_sims, batch_size,
    )

    # False discoveries from reverse rank accumulation
    N = len(obs_sorted)
    false_discoveries = np.cumsum(counts)[:N] / n_sims

    # FWER for each observed stat
    fwer = np.mean(min_stats_per_sim.reshape(1, -1) <= obs_sorted.reshape(-1, 1), axis=1)

    df_results = _hmp_apply_corrections(df_stats, false_discoveries, fwer)

    # Optional gene name annotation
    guide_path = os.path.join(reference_dir, 'protein_sequence_guide.tsv')
    if os.path.exists(guide_path):
        df_gene = pd.read_csv(guide_path, sep='\t')
        df_results = df_results.merge(
            df_gene[['gene_name', 'uniprot_id']], how='left', on='uniprot_id'
        )
    else:
        logger.warning(
            f'protein_sequence_guide.tsv not found at {guide_path}; '
            'gene_name column will not be added'
        )

    summarize_results_hmp(df_results, fdr_cutoff)
    return df_results
