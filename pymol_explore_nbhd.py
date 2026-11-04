"""
Interactive PyMOL session for exploring the output of --explore-nhbds-traits.

Usage (inside a running PyMOL session, with this repo's directory on PYMOL's
path / cwd):

    run pymol_explore_nbhd.py
    load_data tsv_file=/path/to/nbhd_trait_betahat_exploration.tsv, reference_dir=/path/to/reference-data
    load_nbhd protein=Q92508, aa_pos=1250
    next_trait
    prev_trait
    show_trait trait=1697

load_nbhd loads the protein's structure (grey cartoon) and shows the first
trait's neighborhood (residues adjacent to aa_pos within that trait's
radius_used), colored by mean_betahat on a blue(negative)-white(zero)-red
(positive) scale shared across all traits for that neighborhood so colors
are comparable. The center residue is labeled with the trait id and value.
Step through traits with next_trait / prev_trait, or jump straight to one
with show_trait.
"""

import os
import sys
import ast
import re
import numpy as np
import pandas as pd
from pymol import cmd

REPO_DIR = '/Users/gpanagio/local_analysis/ppi/sir/structure-informed-rvas'
for p in (REPO_DIR, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)
from utils import get_adjacency_matrix

STATE = {
    'df': None,
    'reference_dir': None,
    'pdb_file_pos_guide': None,
    'pdb_dir': None,
    'pae_dir': None,
    'pae_cutoff': 4.0,
    'protein': None,
    'aa_pos': None,
    'rows': None,       # DataFrame of rows for current (protein, aa_pos)
    'idx': 0,
    'vabs': 1.0,        # symmetric color-scale bound shared across traits
    'adj_cache': {},    # (protein, radius) -> adjacency matrix
    'chunk_info': None, # pos_covered chunk picked for current protein/aa_pos
}


def load_data(tsv_file, reference_dir, pae_cutoff=4.0):
    """Load the --explore-nhbds-traits output tsv and reference paths."""
    df = pd.read_csv(tsv_file, sep='\t')
    STATE['df'] = df
    STATE['reference_dir'] = reference_dir
    STATE['pdb_file_pos_guide'] = os.path.join(reference_dir, 'pdb_pae_file_pos_guide.tsv')
    STATE['pdb_dir'] = os.path.join(reference_dir, 'pdb_files')
    STATE['pae_dir'] = os.path.join(reference_dir, 'pae_files')
    STATE['pae_cutoff'] = float(pae_cutoff)
    print(f"Loaded {len(df)} rows from {tsv_file}")


def _pick_chunk_for_pos(protein, aa_pos):
    """Find the pdb chunk covering aa_pos, and its (start, end, filename)."""
    info = pd.read_csv(STATE['pdb_file_pos_guide'], sep='\t')
    info = info[info['uniprot_id'] == protein] if 'uniprot_id' in info.columns \
        else info[info['pdb_filename'].str.contains(protein)]
    if len(info) == 0:
        raise ValueError(f"Protein {protein} not found in pdb_pae_file_pos_guide.tsv")

    for _, r in info.iterrows():
        pos_covered = r['pos_covered']
        if isinstance(pos_covered, str):
            pos_covered = ast.literal_eval(pos_covered)
        start, end = int(pos_covered[0]), int(pos_covered[1])
        if start <= aa_pos <= end:
            return {'pdb_filename': r['pdb_filename'], 'start': start, 'end': end}

    # fall back to the chunk whose range is closest, in case aa_pos is at
    # a boundary not cleanly covered
    best = None
    best_dist = None
    for _, r in info.iterrows():
        pos_covered = r['pos_covered']
        if isinstance(pos_covered, str):
            pos_covered = ast.literal_eval(pos_covered)
        start, end = int(pos_covered[0]), int(pos_covered[1])
        dist = 0 if start <= aa_pos <= end else min(abs(aa_pos - start), abs(aa_pos - end))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = {'pdb_filename': r['pdb_filename'], 'start': start, 'end': end}
    if best is None:
        raise ValueError(f"No pdb chunk found for {protein} aa_pos {aa_pos}")
    return best


def _get_adjacency(protein, radius):
    key = (protein, float(radius))
    if key not in STATE['adj_cache']:
        STATE['adj_cache'][key] = get_adjacency_matrix(
            STATE['pdb_file_pos_guide'], STATE['pdb_dir'], STATE['pae_dir'],
            protein, float(radius), STATE['pae_cutoff'],
        )
    return STATE['adj_cache'][key]


def load_nbhd(protein, aa_pos):
    """Load a protein's structure and the set of traits for one neighborhood."""
    aa_pos = int(aa_pos)
    df = STATE['df']
    if df is None:
        print("[ERROR] Call load_data first.")
        return

    rows = df[(df['protein'] == protein) & (df['aa_pos'] == aa_pos)].copy()
    if len(rows) == 0:
        print(f"[ERROR] No rows found for protein={protein} aa_pos={aa_pos}")
        return
    rows = rows.sort_values(by='mean_betahat').reset_index(drop=True)

    STATE['protein'] = protein
    STATE['aa_pos'] = aa_pos
    STATE['rows'] = rows
    STATE['idx'] = 0
    STATE['vabs'] = max(1e-6, float(np.nanmax(np.abs(rows['mean_betahat'].values))))
    STATE['chunk_info'] = _pick_chunk_for_pos(protein, aa_pos)

    cmd.reinitialize()
    pdb_path = os.path.join(STATE['pdb_dir'], STATE['chunk_info']['pdb_filename'])
    cmd.load(pdb_path, protein)
    cmd.hide('everything')
    cmd.show('cartoon', protein)
    cmd.color('grey80', protein)
    cmd.bg_color('white')

    print(f"Loaded {protein} ({len(rows)} traits at aa_pos {aa_pos}); "
          f"chunk {STATE['chunk_info']['pdb_filename']} "
          f"[{STATE['chunk_info']['start']}-{STATE['chunk_info']['end']}]")
    print("Traits (sorted by mean_betahat): " +
          ", ".join(f"{t}({b:.3f})" for t, b in zip(rows['trait'], rows['mean_betahat'])))

    show_trait(idx=0)


def _to_file_resi(aa_pos_full):
    chunk = STATE['chunk_info']
    return aa_pos_full - chunk['start'] + 1


def _show_row(row):
    protein = STATE['protein']
    aa_pos = STATE['aa_pos']
    chunk = STATE['chunk_info']
    radius = row['radius_used']

    cmd.hide('spheres', protein)
    cmd.color('grey80', protein)

    adj = _get_adjacency(protein, radius)
    if adj is None:
        print(f"[WARNING] No adjacency matrix for {protein} at radius {radius}; "
              "showing only the center residue.")
        neighbor_positions_full = [aa_pos]
    else:
        n_res = adj.shape[0]
        if aa_pos < 1 or aa_pos > n_res:
            print(f"[WARNING] aa_pos {aa_pos} out of range for adjacency (n_res={n_res})")
            neighbor_positions_full = [aa_pos]
        else:
            neighbor_idx = np.where(adj[aa_pos - 1] == 1)[0]
            neighbor_positions_full = sorted(set([aa_pos] + (neighbor_idx + 1).tolist()))

    # restrict to residues present in the loaded chunk
    neighbor_positions_full = [p for p in neighbor_positions_full
                                if chunk['start'] <= p <= chunk['end']]
    resi_list = [str(_to_file_resi(p)) for p in neighbor_positions_full]

    beta = float(row['mean_betahat'])
    vabs = STATE['vabs']
    norm = max(-1.0, min(1.0, beta / vabs))  # in [-1, 1]

    if norm >= 0:
        color = [1.0, 1.0 - norm, 1.0 - norm]       # white -> red
    else:
        color = [1.0 + norm, 1.0 + norm, 1.0]       # white -> blue
    color_name = f"betacolor_{row['trait']}"
    cmd.set_color(color_name, color)

    sel_name = 'nbhd_sel'
    cmd.select(sel_name, f"{protein} and resi {'+'.join(resi_list)} and name CA")
    cmd.show('spheres', sel_name)
    cmd.color(color_name, sel_name)

    cmd.label(f"{protein} and resi {_to_file_resi(aa_pos)} and name CA",
               f'"{row["trait"]}: {beta:.3f}"')
    cmd.zoom(sel_name, buffer=8)

    tag = 'DISCOVERY' if row['is_discovery_trait'] else 'trait'
    title = (f"[{STATE['idx']+1}/{len(STATE['rows'])}] {protein} aa_pos {aa_pos} | "
             f"{tag} {row['trait']} | radius={radius} | n_betahat={row['n_betahat']:.0f} | "
             f"mean_betahat={beta:.4f} | source={row['source']}")
    cmd.set('text', 1)
    print(title)


def show_trait(idx=None, trait=None):
    """Jump to a specific trait by index (0-based) or trait id."""
    rows = STATE['rows']
    if rows is None:
        print("[ERROR] Call load_nbhd first.")
        return
    if trait is not None:
        matches = rows.index[rows['trait'].astype(str) == str(trait)].tolist()
        if not matches:
            print(f"[ERROR] trait {trait} not found in current neighborhood.")
            return
        idx = matches[0]
    idx = int(idx)
    idx = max(0, min(idx, len(rows) - 1))
    STATE['idx'] = idx
    _show_row(rows.iloc[idx])


def next_trait():
    rows = STATE['rows']
    if rows is None:
        print("[ERROR] Call load_nbhd first.")
        return
    show_trait(idx=(STATE['idx'] + 1) % len(rows))


def prev_trait():
    rows = STATE['rows']
    if rows is None:
        print("[ERROR] Call load_nbhd first.")
        return
    show_trait(idx=(STATE['idx'] - 1) % len(rows))


cmd.extend('load_data', load_data)
cmd.extend('load_nbhd', load_nbhd)
cmd.extend('show_trait', show_trait)
cmd.extend('next_trait', next_trait)
cmd.extend('prev_trait', prev_trait)
