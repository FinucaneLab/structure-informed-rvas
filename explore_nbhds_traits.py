"""
Post-discovery exploration: for a batch of (protein, aa_pos, discovery_trait)
neighborhoods of interest, find every trait with a missense signal for that
protein and compute the average betahat in the neighborhood for each trait,
for later PyMOL visualization.

Two radius policies, selected via --fixed-radius in run.py:
  - stored (default): use each trait's own best radius, as already recorded
    in its p_values.h5 (`{uniprot_id}_radius` / `{uniprot_id}_mean_beta`).
    No raw recomputation; fast.
  - fixed (--fixed-radius): force every trait to use the discovery trait's
    best radius for that neighborhood. Since a trait's own h5 only stores
    results for its own best radius, this requires recomputing the
    neighborhood betahat aggregate from that trait's raw mapped variant data
    at the discovery radius.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import h5py

from utils import get_adjacency_matrix
from q_scan_test import get_betahats_per_residue, get_nbhd_stats
from read_data import map_to_protein_q
from logger_config import get_logger

logger = get_logger(__name__)

MISSENSE_LIST_PATTERN = 'ukbb_{trait}_missense_genes_p1e04_p1e02_uniprotid.ids_only'


def load_triplets(triplets_file):
    """
    Load a tab-delimited file of (protein, aa_pos, discovery_trait) triplets.
    Works whether or not the file has a header row.
    """
    df = pd.read_csv(triplets_file, sep='\t', header=None,
                      names=['protein', 'aa_pos', 'discovery_trait'], dtype=str)
    # If the first row isn't numeric in the aa_pos column, treat it as a header.
    if not df.iloc[0]['aa_pos'].strip().lstrip('-').isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    df['aa_pos'] = df['aa_pos'].astype(int)
    return df


def build_trait_to_proteins_index(missense_trait_dir, pattern=MISSENSE_LIST_PATTERN):
    """
    Scan missense_trait_dir once and build {trait: set(uniprot_id)} for every
    ukbb_{trait}_missense_genes_p1e04_p1e02_uniprotid.ids_only file found.
    """
    glob_pattern = os.path.join(missense_trait_dir, pattern.format(trait='*'))
    trait_regex = re.compile('^' + re.escape(pattern).replace(re.escape('{trait}'), '(.+)') + '$')

    trait_to_proteins = {}
    for f in glob.glob(glob_pattern):
        m = trait_regex.match(os.path.basename(f))
        if not m:
            continue
        trait = m.group(1)
        with open(f) as fh:
            ids = {line.strip() for line in fh if line.strip()}
        trait_to_proteins[trait] = ids

    logger.info(f"Indexed {len(trait_to_proteins)} traits from {missense_trait_dir}")
    return trait_to_proteins


def traits_with_missense_signal(protein, trait_to_proteins):
    return sorted(trait for trait, ids in trait_to_proteins.items() if protein in ids)


def read_stored_info(input_dir_hmp, trait, stat_method, pval_file, protein, aa_pos):
    """
    Read a trait's own best radius / n_betahat / mean_betahat for one
    (protein, aa_pos) from its per-trait p_values.h5, as already computed
    by --run-q3dnt with a multi-radius --neighborhood-radius.
    Returns None if the file, protein, or position isn't found.
    """
    h5_path = os.path.join(input_dir_hmp.format(trait=trait, stat_method=stat_method), pval_file)
    if not os.path.exists(h5_path):
        logger.warning(f"{trait}: p_values.h5 not found at {h5_path}")
        return None

    with h5py.File(h5_path, 'r') as fid:
        if protein not in fid:
            return None

        aa_pos_key = f'{protein}_aa_pos'
        if aa_pos_key in fid:
            aa_positions = fid[aa_pos_key][:].flatten()
        else:
            aa_positions = np.arange(1, fid[protein].shape[0] + 1)

        idx = np.where(aa_positions == aa_pos)[0]
        if len(idx) == 0:
            return None
        i = int(idx[0])

        mean_beta_ds = fid[f'{protein}_mean_beta'][:]
        mean_betahat = float(mean_beta_ds[i, 0])
        n_betahat = float(mean_beta_ds[i, 1])

        radius_key = f'{protein}_radius'
        radius = float(fid[radius_key][i, 0]) if radius_key in fid else None

    return {'n_betahat': n_betahat, 'mean_betahat': mean_betahat, 'radius': radius}


def recompute_mean_betahat(df_mapped_trait, protein, aa_pos, radius,
                            pdb_file_pos_guide, pdb_dir, pae_dir, pae_cutoff):
    """
    Recompute n_betahat / mean_betahat for one (protein, aa_pos) at a
    specific radius, from that trait's raw mapped variant data. Used when
    forcing a radius other than the trait's own stored best radius.
    """
    df_protein = df_mapped_trait[df_mapped_trait.uniprot_id == protein]
    if df_protein.empty:
        return None

    adjacency_matrix = get_adjacency_matrix(pdb_file_pos_guide, pdb_dir, pae_dir, protein, radius, pae_cutoff)
    n_res = adjacency_matrix.shape[0]
    if aa_pos < 1 or aa_pos > n_res:
        logger.warning(f"{protein}: aa_pos {aa_pos} out of range (n_res={n_res}) at radius {radius}")
        return None

    base, base_sq, nulls, nulls_sq, n_per_residue = get_betahats_per_residue(df_protein, n_res, n_sims=0)
    sum_betahat_matrix = np.column_stack([base, nulls])
    sum_betahat_sq_matrix = np.column_stack([base_sq, nulls_sq])
    n_betahat, sum_betahat, _ = get_nbhd_stats(adjacency_matrix, n_per_residue, sum_betahat_matrix, sum_betahat_sq_matrix)

    idx = aa_pos - 1
    n_bh = float(n_betahat[idx])
    mean_bh = float(sum_betahat[idx, 0] / n_bh) if n_bh > 0 else np.nan
    return {'n_betahat': n_bh, 'mean_betahat': mean_bh, 'radius': float(radius)}


def run_explore_nbhds_traits(
        triplets_file,
        missense_trait_dir,
        input_dir_hmp,
        stat_method,
        pval_file,
        reference_dir,
        pae_cutoff,
        rvas_data_to_map_template,
        beta_col,
        genome_build,
        fixed_radius,
        output_file,
        min_variants=5,
):
    triplets = load_triplets(triplets_file)
    trait_to_proteins = build_trait_to_proteins_index(missense_trait_dir)

    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'

    stored_rows = []
    pending_recompute = []  # only used when fixed_radius=True

    for _, trip in triplets.iterrows():
        protein, aa_pos, discovery_trait = trip['protein'], int(trip['aa_pos']), trip['discovery_trait']

        discovery_info = read_stored_info(input_dir_hmp, discovery_trait, stat_method, pval_file, protein, aa_pos)
        if discovery_info is None or discovery_info['radius'] is None:
            logger.warning(f"{protein} {aa_pos}: no stored radius for discovery trait {discovery_trait}; skipping.")
            continue
        discovery_radius = discovery_info['radius']

        traits = traits_with_missense_signal(protein, trait_to_proteins)
        if discovery_trait not in traits:
            traits = traits + [discovery_trait]

        for trait in traits:
            info = read_stored_info(input_dir_hmp, trait, stat_method, pval_file, protein, aa_pos)
            if info is None:
                logger.warning(f"{protein} {aa_pos}: no data for trait {trait}; skipping.")
                continue
            if info['n_betahat'] < min_variants:
                continue

            row = {
                'protein': protein,
                'aa_pos': aa_pos,
                'discovery_trait': discovery_trait,
                'trait': trait,
                'is_discovery_trait': trait == discovery_trait,
            }

            if not fixed_radius:
                row.update({
                    'radius_used': info['radius'],
                    'n_betahat': info['n_betahat'],
                    'mean_betahat': info['mean_betahat'],
                    'source': 'stored',
                })
                stored_rows.append(row)
            else:
                row.update({'radius_used': discovery_radius})
                pending_recompute.append(row)

    if not fixed_radius:
        df_out = pd.DataFrame(stored_rows)
    else:
        df_pending = pd.DataFrame(pending_recompute)
        final_rows = []
        for trait, group in df_pending.groupby('trait'):
            raw_path = rvas_data_to_map_template.format(trait=trait)
            if not os.path.exists(raw_path):
                logger.error(f"{trait}: raw rvas-to-map file not found at {raw_path}; skipping {len(group)} rows.")
                continue

            logger.info(f"{trait}: mapping raw variants from {raw_path} for {group['protein'].nunique()} proteins")
            df_mapped = map_to_protein_q(raw_path, beta_col, reference_dir, genome_build=genome_build)
            df_mapped = df_mapped[df_mapped.uniprot_id.isin(group['protein'].unique())]

            for _, r in group.iterrows():
                res = recompute_mean_betahat(
                    df_mapped, r['protein'], r['aa_pos'], r['radius_used'],
                    pdb_file_pos_guide, pdb_dir, pae_dir, pae_cutoff,
                )
                if res is None or res['n_betahat'] < min_variants:
                    continue
                final_rows.append({
                    **r.to_dict(),
                    'n_betahat': res['n_betahat'],
                    'mean_betahat': res['mean_betahat'],
                    'source': 'recomputed_fixed_radius',
                })
        df_out = pd.DataFrame(final_rows)

    df_out.to_csv(output_file, sep='\t', index=False)
    logger.info(f"Wrote {len(df_out)} rows to {output_file}")
    return df_out
