#!/usr/bin/env python3
"""
visualize_and_interpret.py - Protein structure visualization and interpretation tools

Subcommands:
  run_all    Full pipeline (visualize, annotate, neighborhoods) for all significant
             proteins, or for every component with --components
  visualize  Generate PyMOL visualizations for a single protein
  annotate   Transfer functional site annotations from protein family members
  nbhd-features  Cross-reference a neighborhood with functional features
  color      Color a chain in a PyMOL session by per-residue ratio values

Usage:
  python visualize_and_interpret.py run_all --fdr-file <tsv> --results-dir <dir> --reference-dir <dir>
  python visualize_and_interpret.py annotate <uniprot_id> <output_tsv> [options]
  python visualize_and_interpret.py color --tsv <tsv> --input-cif <cif> --output-pse <pse> --chain <chain> --uniprot <id>
"""

import argparse
import csv
import os
import re
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from Bio.Align import PairwiseAligner

try:
    import requests
except ImportError:
    # Only the UniProt annotation paths need network access. Visualization, the
    # neighborhood residue tables and the feature cross-reference all work offline, so
    # a missing `requests` must not stop them being imported.
    class _RequestsUnavailable:
        def __getattr__(self, name):
            raise ImportError(
                'The annotation steps need the `requests` package (pip install requests). '
                'Visualization and neighborhood feature tables do not.'
            )

    requests = _RequestsUnavailable()


# ---------------------------------------------------------------------------
# Alignment utilities
# ---------------------------------------------------------------------------

def get_pairwise_alignment(seq1, seq2):
    """Global pairwise alignment with gap-open=-10, gap-extend=-0.5 to penalize fragmented gaps."""
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(seq1, seq2)
    return alignments[0]


def alignment_to_strings(alignment):
    """Convert a Biopython Alignment object to two equal-length strings with '-' for gaps."""
    coords = alignment.coordinates
    seq1 = alignment.sequences[0]
    seq2 = alignment.sequences[1]

    aligned1 = []
    aligned2 = []

    for i in range(len(coords[0]) - 1):
        start1, end1 = coords[0][i], coords[0][i+1]
        start2, end2 = coords[1][i], coords[1][i+1]

        len1 = end1 - start1
        len2 = end2 - start2

        if len1 == 0:
            aligned1.extend(['-'] * len2)
            aligned2.extend(list(seq2[start2:end2]))
        elif len2 == 0:
            aligned1.extend(list(seq1[start1:end1]))
            aligned2.extend(['-'] * len1)
        else:
            aligned1.extend(list(seq1[start1:end1]))
            aligned2.extend(list(seq2[start2:end2]))

    return ''.join(aligned1), ''.join(aligned2)


def get_alignment_position(aligned_seq, query_pos):
    """
    Return the index in the gapped string `aligned_seq` corresponding to
    1-indexed position `query_pos` in the ungapped sequence. Returns None if
    `query_pos` is beyond the end of the sequence.
    """
    current_query_pos = 0
    for align_pos in range(len(aligned_seq)):
        if aligned_seq[align_pos] != '-':
            current_query_pos += 1
        if current_query_pos == query_pos:
            return align_pos
    return None


def calculate_local_identity(alignment, position, window=5):
    """Fraction of identical residue pairs in a ±window window around `position` in seq1."""
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)
    align_pos = get_alignment_position(aligned_seq1, position)
    if align_pos is None:
        return 0.0

    start = max(0, align_pos - window)
    end = min(len(aligned_seq1), align_pos + window + 1)
    matches = 0
    comparisons = 0

    for i in range(start, end):
        if aligned_seq1[i] != '-' and aligned_seq2[i] != '-':
            comparisons += 1
            if aligned_seq1[i] == aligned_seq2[i]:
                matches += 1

    return matches / comparisons if comparisons > 0 else 0.0


def calculate_gap_density(alignment, position, window=5):
    """Fraction of positions that are gaps in either sequence in a ±window window around `position`."""
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)
    align_pos = get_alignment_position(aligned_seq1, position)
    if align_pos is None:
        return 1.0

    start = max(0, align_pos - window)
    end = min(len(aligned_seq1), align_pos + window + 1)
    total_positions = end - start
    if total_positions == 0:
        return 1.0

    gaps = sum(1 for i in range(start, end)
               if aligned_seq1[i] == '-' or aligned_seq2[i] == '-')
    return gaps / total_positions


def calculate_position_score(alignment, position):
    """1.0 if the residue at `position` is an exact match, 0.0 if gapped or mismatched."""
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)
    align_pos = get_alignment_position(aligned_seq1, position)
    if align_pos is None:
        return 0.0
    if aligned_seq1[align_pos] == '-' or aligned_seq2[align_pos] == '-':
        return 0.0
    return 1.0 if aligned_seq1[align_pos] == aligned_seq2[align_pos] else 0.0


def calculate_confidence_score(alignment, position):
    """
    Weighted annotation-transfer confidence for a single position:
      0.4 * local_identity + 0.3 * (1 - gap_density) + 0.3 * position_score
    Ranges 0–1; used to filter transferred annotations before writing output.
    """
    local_identity = calculate_local_identity(alignment, position)
    gap_density = calculate_gap_density(alignment, position)
    position_score = calculate_position_score(alignment, position)
    return 0.4 * local_identity + 0.3 * (1 - gap_density) + 0.3 * position_score


def get_aligned_position(alignment, query_pos):
    """
    Map 1-indexed `query_pos` in seq1 to its 1-indexed position in seq2 via the alignment.
    Returns None if the aligned column is a gap in seq2.
    """
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)
    align_pos = get_alignment_position(aligned_seq1, query_pos)
    if align_pos is None:
        return None
    if aligned_seq2[align_pos] == '-':
        return None
    target_pos = sum(1 for i in range(align_pos + 1) if aligned_seq2[i] != '-')
    return target_pos


def transfer_all_annotations(query_seq, target_seq, target_annotations):
    """
    Align query to target and transfer every annotation in `target_annotations`
    to the corresponding query position. Each returned record includes the
    alignment confidence score so low-quality transfers can be filtered later.
    `target_annotations` is a dict mapping str(position) -> list of annotation dicts.
    """
    alignment = get_pairwise_alignment(query_seq, target_seq)
    transferred = []

    for query_pos in range(1, len(query_seq) + 1):
        target_pos = get_aligned_position(alignment, query_pos)
        if target_pos is None:
            continue
        target_pos_str = str(target_pos)
        if target_pos_str not in target_annotations:
            continue
        confidence = calculate_confidence_score(alignment, query_pos)
        for annotation in target_annotations[target_pos_str]:
            transferred.append({
                'query_position': query_pos,
                'target_position': target_pos,
                'feature_type': annotation['feature_type'],
                'feature_description': annotation['description'],
                'alignment_confidence': confidence
            })

    return transferred


# ---------------------------------------------------------------------------
# UniProt API utilities
# ---------------------------------------------------------------------------

def get_protein_family(uniprot_id):
    """
    Return the raw SIMILARITY comment from UniProt (e.g. "Belongs to the Kv1
    potassium channel family"), or None if the protein has no such comment.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    try:
        for comment in data.get('comments', []):
            if comment.get('commentType') == 'SIMILARITY':
                return comment['texts'][0]['value']
    except (KeyError, IndexError):
        pass
    return None


def extract_family_name(family_text):
    """
    Parse the family name from "Belongs to the X (super)family" text.
    Returns X (stripped), or None if the pattern does not match.
    """
    if not family_text:
        return None
    match = re.search(r'Belongs to the (.+?)(super)?family', family_text)
    if match:
        return match.group(1).strip()
    return None


def search_family_members_with_annotations(family_name, is_superfamily=False):
    """
    Search UniProt for proteins in `family_name` that have PDB structures.
    Tries "family" then "superfamily" suffix (or the reverse if `is_superfamily`).
    Returns a list of UniProt accessions, or [] if nothing is found.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    if not family_name.endswith(' '):
        family_name = family_name + ' '
    suffixes = ['superfamily', 'family'] if is_superfamily else ['family', 'superfamily']

    for suffix in suffixes:
        params = {
            'query': f'family:"{family_name}{suffix}" AND database:pdb',
            'fields': 'accession,organism_name,ft_binding,ft_act_site,ft_site,xref_pdb',
            'size': 500,
            'format': 'json'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        protein_list = [p['primaryAccession'] for p in response.json().get('results', [])]
        if protein_list:
            return protein_list

    return []


def get_protein_list(uniprot_id):
    """
    Return a list of UniProt accessions for PDB-annotated family members of
    `uniprot_id`. Raises ValueError if no family information is found or if the
    family name cannot be parsed from the SIMILARITY comment.
    """
    family_text = get_protein_family(uniprot_id)
    if not family_text:
        raise ValueError(
            f"No family information found for {uniprot_id}. "
            f"Check https://www.uniprot.org/uniprotkb/{uniprot_id}"
        )
    family_name = extract_family_name(family_text)
    if not family_name:
        raise ValueError(f"Could not extract family name from: {family_text}")

    is_superfamily = 'superfamily' in family_text.lower()
    print(f'Family name: {family_name}')
    if is_superfamily:
        print('Type: superfamily')

    protein_list = search_family_members_with_annotations(family_name, is_superfamily)
    print(f'Found {len(protein_list)} family members with PDB structures')
    return protein_list


def get_uniprot_sequence(uniprot_id):
    """Fetch the canonical amino acid sequence for `uniprot_id` from UniProt."""
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}")
    response.raise_for_status()
    return response.json().get('sequence', {}).get('value', '')


def get_protein_function(uniprot_id):
    """Return the first FUNCTION comment text from UniProt, or None if absent."""
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}")
    response.raise_for_status()
    try:
        return response.json()['comments'][0]['texts'][0]['value']
    except (KeyError, IndexError):
        return None


def get_functional_sites_detailed(uniprot_id, cache_dir=None):
    """
    Fetch functional site annotations from UniProt for `uniprot_id`.

    Returns a dict mapping str(residue_position) -> list of
    {position, feature_type, description} dicts. Results are cached to
    {cache_dir}/{uniprot_id}.json so repeated calls are free.

    Included feature types: Binding site, Active site, Site, Metal binding,
    Calcium binding, DNA binding, Nucleotide binding, Modified residue,
    Cross-link, Transmembrane, Motif, Region, Domain, Topological domain,
    Zinc finger.

    Range features spanning >250 residues are excluded to avoid capturing
    whole-protein features (e.g. Chain, Propeptide).
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{uniprot_id}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

    functional_sites = {}

    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        features = data.get('features', [])

        functional_types = {
            'Binding site', 'Active site', 'Site', 'Metal binding',
            'Calcium binding', 'DNA binding', 'Nucleotide binding',
            'Modified residue', 'Cross-link', 'Transmembrane', 'Motif',
            'Region', 'Domain', 'Topological domain', 'Zinc finger',
        }

        for feature in features:
            if feature.get('type') not in functional_types:
                continue
            location = feature.get('location', {})
            feature_type = feature.get('type')
            description = feature.get('description', '')
            if 'ligand' in feature:
                ligand_name = feature['ligand'].get('name', '')
                if ligand_name:
                    description = f"{description} Ligand: {ligand_name}".strip()

            positions_to_add = []
            if 'position' in location:
                pos = location['position'].get('value')
                if pos:
                    positions_to_add.append(int(pos))
            elif 'start' in location and 'end' in location:
                start_pos = location['start'].get('value')
                end_pos = location['end'].get('value')
                if start_pos and end_pos and (end_pos - start_pos < 250):
                    positions_to_add.extend(range(int(start_pos), int(end_pos) + 1))

            for pos in positions_to_add:
                site = {'position': pos, 'feature_type': feature_type, 'description': description}
                pos_str = str(pos)
                if pos_str not in functional_sites:
                    functional_sites[pos_str] = []
                functional_sites[pos_str].append(site)

        time.sleep(0.1)

    except Exception as e:
        print(f"Warning: Error fetching data for {uniprot_id}: {e}")

    if cache_dir:
        with open(cache_file, 'w') as f:
            json.dump(functional_sites, f, indent=2)

    return functional_sites


def save_alignment_cache(query_id, target_id, aligned_seq1, aligned_seq2, cache_dir):
    """Save a gapped alignment string pair to {cache_dir}/alignment_{query_id}_{target_id}.json."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"alignment_{query_id}_{target_id}.json")
    with open(cache_file, 'w') as f:
        json.dump({'seqA': aligned_seq1, 'seqB': aligned_seq2}, f)


def load_alignment_cache(query_id, target_id, cache_dir):
    """Load a cached alignment string pair, or return None if not cached."""
    cache_file = os.path.join(cache_dir, f"alignment_{query_id}_{target_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Annotation transfer logic
# ---------------------------------------------------------------------------

def filter_family_by_annotation_count(family_members, cache_dir, max_members=50):
    """
    Trim `family_members` to the `max_members` proteins with the most annotated
    positions. Used to keep runtime manageable when a family has hundreds of members.
    """
    functional_cache = os.path.join(cache_dir, 'functional_sites')
    os.makedirs(functional_cache, exist_ok=True)
    print(f"\nFiltering {len(family_members)} family members to top {max_members} by annotation count...")

    annotation_counts = []
    for member_id in family_members:
        try:
            annotations = get_functional_sites_detailed(member_id, functional_cache)
            annotation_counts.append((member_id, len(annotations)))
        except Exception:
            annotation_counts.append((member_id, 0))

    annotation_counts.sort(key=lambda x: x[1], reverse=True)
    top_members = [mid for mid, _ in annotation_counts[:max_members]]

    print(f"Selected {len(top_members)} family members with most annotations")
    print(f"  Top member: {annotation_counts[0][0]} with {annotation_counts[0][1]} positions")
    if len(annotation_counts) > 1:
        print(f"  Median: {annotation_counts[len(annotation_counts)//2][1]} positions")
        print(f"  Bottom: {annotation_counts[-1][1]} positions")

    return top_members


def _load_gene_map(reference_dir):
    """Return {uniprot_id: gene_name} from protein_sequence_guide.tsv."""
    path = os.path.join(reference_dir, 'protein_sequence_guide.tsv')
    try:
        df = pd.read_csv(path, sep='\t')
        return dict(zip(df['uniprot_id'], df['gene_name']))
    except Exception:
        return {}


def _load_mut_data(pval_path, uniprot_id, test='binomial'):
    """
    Read one protein's per-residue mutation data, from either flavour of p-value file.
    Returns (df_mut, mu_mode); in mutation-rate mode df_mut carries `mu` rather than
    `ac_control`.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import read_original_mutation_data, resolve_pval_node

    with h5py.File(pval_path, 'r') as fid:
        node, mu_mode = resolve_pval_node(fid, test)
        return read_original_mutation_data(node, uniprot_id, mu_mode=mu_mode), mu_mode


# Column layouts of the two kinds of FDR table. The mutation-rate run corrects each test
# separately and reports the max, so it has no plain `fdr` / `p_value` column at all.
MU_FDR_SCHEMA = {
    'significance_column': 'fdr_max',
    'p_column': {'binomial': 'p_binomial', 'poisson': 'p_poisson'},
    'radius_columns': {'binomial': ('radius_binomial', 'radius_poisson'),
                       'poisson': ('radius_poisson', 'radius_binomial')},
}
STD_FDR_SCHEMA = {
    'significance_column': 'fdr',
    'p_column': {'binomial': 'p_value', 'poisson': 'p_value'},
    'radius_columns': {'binomial': ('radius',), 'poisson': ('radius',)},
}


def _fdr_schema(df_fdr):
    """Pick the column layout for an FDR table by looking at what it actually contains."""
    return MU_FDR_SCHEMA if 'fdr_max' in df_fdr.columns else STD_FDR_SCHEMA


def _compute_nbhd(uniprot_id, center, reference_dir, radius, pae_cutoff):
    """Return set of residue positions in the neighborhood centered at `center`."""
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import get_adjacency_matrix
    pdb_pae_guide = os.path.join(reference_dir, 'pdb_pae_file_pos_guide.tsv')
    pdb_dir = os.path.join(reference_dir, 'pdb_files')
    pae_dir = os.path.join(reference_dir, 'pae_files')
    adj_mat = get_adjacency_matrix(pdb_pae_guide, pdb_dir, pae_dir, uniprot_id, radius, pae_cutoff)
    if adj_mat is None:
        raise RuntimeError(f"No structure found for {uniprot_id}")
    return set(int(x) for x in np.where(adj_mat[int(center) - 1, :] == 1)[0] + 1)


def _nbhd_features_table(nbhd, df_mut, features_df, mu_mode=False):
    """
    Cross-reference `nbhd` (set of residue positions) with `features_df`
    (output of generate_simple_feature_table).

    `df_mut` must have columns aa_pos, ac_case, and either ac_control (standard 3DNT)
    or mu (mutation-rate mode).

    For each feature that overlaps the neighborhood, computes:
      - prop_nbhd_in_feature: fraction of neighborhood residues in the feature
      - prop_feature_in_nbhd: fraction of feature residues in the neighborhood

    and then, in standard mode:
      - prop_case_only_in_feature: fraction of case-only alleles (in the nbhd)
        that fall inside the feature
      - case_only_positions: comma-separated list of case-only-mutated positions

    or, in mutation-rate mode, where there is no control cohort to define "case-only":
      - prop_denovo_in_feature: fraction of the neighborhood's de novos inside the feature
      - prop_mu_in_feature: fraction of the neighborhood's mutational opportunity inside
        the feature -- the null for the line above. A feature that simply covers more of
        the neighborhood, or covers a mutational hotspot, carries a high
        prop_denovo_in_feature on its own; the comparison between the two columns is what
        says whether the de novos are concentrated there beyond the rate.
      - denovo_positions: comma-separated list of de novo positions inside the feature

    Returns a DataFrame sorted descending by the relevant proportion column.
    """
    df_nbhd = df_mut[df_mut['aa_pos'].isin(nbhd)]
    if mu_mode:
        df_hit = df_nbhd[df_nbhd['ac_case'] > 0]
        mu_nbhd = float(df_nbhd['mu'].sum())
    else:
        df_hit = df_nbhd[(df_nbhd['ac_case'] > 0) & (df_nbhd['ac_control'] == 0)]
    n_hit_nbhd = int(df_hit['ac_case'].sum())

    rows = []
    for _, row in features_df.iterrows():
        feature_pos = parse_pymol_positions(row['residues_pymol'])
        overlap = nbhd & feature_pos
        if not overlap:
            continue
        df_hit_in_feature = df_hit[df_hit['aa_pos'].isin(overlap)]
        n_hit_in_feature = int(df_hit_in_feature['ac_case'].sum())
        hit_positions = sorted(df_hit_in_feature['aa_pos'].tolist())
        prop_hit = round(n_hit_in_feature / n_hit_nbhd, 3) if n_hit_nbhd > 0 else None
        out = {
            'feature_name': row['feature_name'],
            'functional_site_category': row['functional_site_category'],
            'prop_nbhd_in_feature': round(len(overlap) / len(nbhd), 3),
            'prop_feature_in_nbhd': round(len(overlap) / row['num_positions'], 3),
        }
        if mu_mode:
            mu_in_feature = float(df_nbhd[df_nbhd['aa_pos'].isin(overlap)]['mu'].sum())
            out['prop_denovo_in_feature'] = prop_hit
            out['prop_mu_in_feature'] = round(mu_in_feature / mu_nbhd, 3) if mu_nbhd > 0 else None
            out['denovo_positions'] = ','.join(str(p) for p in hit_positions) if hit_positions else ''
        else:
            out['prop_case_only_in_feature'] = prop_hit
            out['case_only_positions'] = ','.join(str(p) for p in hit_positions) if hit_positions else ''
        rows.append(out)

    result_df = pd.DataFrame(rows)
    if len(result_df) > 0:
        sort_col = 'prop_denovo_in_feature' if mu_mode else 'prop_case_only_in_feature'
        result_df = result_df.sort_values(sort_col, ascending=False)
    return result_df


def _annotate_and_save(uniprot_id, features_path, cache_dir, confidence_threshold=0.8):
    """
    Convenience wrapper used by cmd_run_all. If `features_path` already exists,
    load and return it (no network calls). Otherwise run the full annotation
    pipeline (family search → alignment → transfer → feature table) and save
    the result to `features_path`.
    """
    if os.path.exists(features_path):
        print(f"  Loading existing features from {features_path}")
        return pd.read_csv(features_path, sep='\t')

    os.makedirs(os.path.dirname(os.path.abspath(features_path)), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    no_family_info = False
    try:
        family_members = get_protein_list(uniprot_id)
    except ValueError as e:
        print(f"  Warning: {e}")
        print("  No family info; using direct annotations only.")
        family_members = []
        no_family_info = True

    if not family_members and not no_family_info:
        family_text = get_protein_family(uniprot_id)
        family_name = extract_family_name(family_text)
        if family_name:
            is_superfamily = 'superfamily' in (family_text or '').lower()
            suffixes = ['superfamily', 'family'] if is_superfamily else ['family', 'superfamily']
            for suffix in suffixes:
                params = {'query': f'family:"{family_name} {suffix}" AND reviewed:true',
                          'fields': 'accession', 'size': 50, 'format': 'json'}
                r = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params)
                if r.status_code == 200 and r.json().get('results'):
                    family_members = [x['primaryAccession'] for x in r.json()['results']]
                    break
            if not family_members:
                for suffix in suffixes:
                    params = {'query': f'family:"{family_name} {suffix}"',
                              'fields': 'accession', 'size': 50, 'format': 'json'}
                    r = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params)
                    if r.status_code == 200 and r.json().get('results'):
                        family_members = [x['primaryAccession'] for x in r.json()['results']]
                        break

    annotations_df = aggregate_annotations_across_family(
        uniprot_id, family_members, cache_dir, confidence_threshold=0.7
    )
    features_df = generate_simple_feature_table(annotations_df, confidence_threshold)
    features_df.to_csv(features_path, sep='\t', index=False)
    print(f"  Saved features to {features_path}")
    return features_df


def parse_pymol_positions(pymol_str):
    """Parse PyMOL residue notation like '10+20-25+30' back to a set of integer positions."""
    if not pymol_str:
        return set()
    positions = set()
    for part in str(pymol_str).split('+'):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            positions.update(range(int(start), int(end) + 1))
        else:
            positions.add(int(part))
    return positions


def positions_to_pymol_notation(positions):
    """
    Encode a collection of integer positions as a compact PyMOL selection string
    using run-length range syntax, e.g. [1,2,3,5,6] -> "1-3+5-6".
    """
    if not positions:
        return ""
    positions = sorted(set(positions))
    ranges = []
    start = end = positions[0]
    for pos in positions[1:]:
        if pos == end + 1:
            end = pos
        else:
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = end = pos
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return "+".join(ranges)


def aggregate_annotations_across_family(query_id, family_members, cache_dir, confidence_threshold=0.7):
    """
    Core annotation pipeline. First adds the query protein's own annotations
    at confidence 1.0, then for each family member: fetches its sequence and
    functional sites, aligns it to the query, and transfers annotations to
    query positions using calculate_confidence_score.

    Returns a flat DataFrame with one row per (query_position, feature, source_protein),
    including all transfers regardless of confidence — filtering happens downstream
    in generate_simple_feature_table.
    """
    print("\n=== Annotation Transfer ===")
    print(f"Query protein: {query_id}")
    print(f"Family members: {len(family_members)}")

    query_seq = get_uniprot_sequence(query_id)
    print(f"Query sequence length: {len(query_seq)}")

    functional_cache = os.path.join(cache_dir, 'functional_sites')
    alignment_cache = os.path.join(cache_dir, 'alignments')
    os.makedirs(functional_cache, exist_ok=True)
    os.makedirs(alignment_cache, exist_ok=True)

    all_transferred = []

    print(f"\n[0/{len(family_members)}] Processing {query_id} (query itself)...")
    query_annotations = get_functional_sites_detailed(query_id, functional_cache)
    print(f"  Functional sites: {len(query_annotations)} positions annotated")

    for pos_str, annotations in query_annotations.items():
        pos = int(pos_str)
        for annot in annotations:
            all_transferred.append({
                'query_position': pos,
                'target_position': pos,
                'feature_type': annot['feature_type'],
                'feature_description': annot['description'],
                'alignment_confidence': 1.0,
                'source_protein': query_id
            })
    print(f"  Added: {len([a for a in all_transferred if a['source_protein'] == query_id])} annotations from query itself")

    for i, target_id in enumerate(family_members, 1):
        print(f"\n[{i}/{len(family_members)}] Processing {target_id}...")
        if target_id == query_id:
            print("  Skipping (already processed as query)")
            continue

        try:
            target_seq = get_uniprot_sequence(target_id)
            print(f"  Target sequence length: {len(target_seq)}")

            target_annotations = get_functional_sites_detailed(target_id, functional_cache)
            print(f"  Functional sites: {len(target_annotations)} positions annotated")

            if len(target_annotations) == 0:
                print("  Skipping (no functional sites)")
                continue

            cached_alignment = load_alignment_cache(query_id, target_id, alignment_cache)
            if cached_alignment:
                print("  Using cached alignment")
                alignment = get_pairwise_alignment(query_seq, target_seq)
                aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)
            else:
                print("  Computing alignment...")
                alignment = get_pairwise_alignment(query_seq, target_seq)
                aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)
                save_alignment_cache(query_id, target_id, aligned_seq1, aligned_seq2, alignment_cache)

            transferred = transfer_all_annotations(query_seq, target_seq, target_annotations)
            print(f"  Transferred: {len(transferred)} annotations")

            for item in transferred:
                item['source_protein'] = target_id

            feature_groups = defaultdict(list)
            for item in transferred:
                feature_groups[(item['feature_type'], item['feature_description'])].append(item)

            print("  Feature details:")
            for (feat_type, feat_desc), items in feature_groups.items():
                source_positions = set()
                for pos_str, annotations in target_annotations.items():
                    for annot in annotations:
                        if annot['feature_type'] == feat_type and annot['description'] == feat_desc:
                            source_positions.add(int(pos_str))

                high_conf_items = [it for it in items if it['alignment_confidence'] >= confidence_threshold]
                query_positions = sorted(set(it['query_position'] for it in high_conf_items))
                target_positions = sorted(set(it['target_position'] for it in high_conf_items))

                print(f"    [{feat_type}] {feat_desc[:60]}...")
                print(f"      Source: {len(source_positions)} positions in {target_id}")
                print(f"      Transferred: {len(high_conf_items)} high-confidence (≥{confidence_threshold}) positions to query")
                if query_positions:
                    qr = f"{min(query_positions)}-{max(query_positions)}" if len(query_positions) > 1 else str(query_positions[0])
                    tr = f"{min(target_positions)}-{max(target_positions)}" if len(target_positions) > 1 else str(target_positions[0])
                    print(f"      Query positions: {qr} (from target {tr})")

            all_transferred.extend(transferred)

        except Exception as e:
            print(f"  Error processing {target_id}: {e}")
            continue

    if not all_transferred:
        print("\nWarning: No annotations transferred!")
        return pd.DataFrame(columns=['query_position', 'source_protein', 'feature_type',
                                     'feature_description', 'alignment_confidence'])

    df = pd.DataFrame(all_transferred)
    print(f"\n=== Transfer Complete ===")
    print(f"Total annotations transferred: {len(df)}")
    print(f"Unique query positions annotated: {df['query_position'].nunique()}")
    print(f"Mean confidence: {df['alignment_confidence'].mean():.3f}")
    return df


def generate_simple_feature_table(annotations_df, confidence_threshold=0.8):
    """
    Collapse the per-position annotation DataFrame from aggregate_annotations_across_family
    into one row per unique (feature_type, feature_description), keeping only
    annotations with alignment_confidence >= confidence_threshold.

    Output columns: feature_name, functional_site_category, is_major_functional_site,
    residues_pymol, num_positions, num_supporting_proteins, source_uniprot_ids,
    mean_confidence.
    """
    print("\n=== Generating Feature Table ===")
    print(f"Confidence threshold: {confidence_threshold}")

    total_before = len(annotations_df)
    annotations_df = annotations_df[annotations_df['alignment_confidence'] >= confidence_threshold].copy()
    total_after = len(annotations_df)
    print(f"Filtered annotations: {total_before} → {total_after} (removed {total_before - total_after} low-confidence)")

    if len(annotations_df) == 0:
        print("\nWarning: No annotations remain after confidence filtering!")
        return pd.DataFrame(columns=['feature_name', 'functional_site_category', 'is_major_functional_site',
                                     'residues_pymol', 'num_positions', 'num_supporting_proteins',
                                     'source_uniprot_ids', 'mean_confidence'])

    features = []
    for (feat_type, feat_desc), group in annotations_df.groupby(['feature_type', 'feature_description']):
        positions = group['query_position'].unique().tolist()
        source_proteins = sorted(group['source_protein'].unique())
        features.append({
            'feature_name': feat_desc,
            'functional_site_category': feat_type,
            'is_major_functional_site': feat_type in ['Binding site', 'Domain', 'Active site', 'Site'],
            'residues_pymol': positions_to_pymol_notation(positions),
            'num_positions': len(positions),
            'num_supporting_proteins': len(source_proteins),
            'source_uniprot_ids': ', '.join(source_proteins),
            'mean_confidence': round(group['alignment_confidence'].mean(), 3)
        })

    features_df = pd.DataFrame(features)
    features_df = features_df.sort_values(['functional_site_category', 'num_positions'], ascending=[True, False])
    print(f"\nGenerated {len(features_df)} unique features")
    return features_df


# ---------------------------------------------------------------------------
# Subcommand: annotate
# ---------------------------------------------------------------------------

def cmd_annotate(args):
    """
    CLI handler for `annotate`. Fetches family members from UniProt, transfers
    their functional site annotations to the query protein via pairwise alignment,
    and writes a features TSV. Falls back to the query protein's own annotations
    if no family information is available. Results are cached so re-runs are fast.
    """
    print(f"Simple functional site annotation for {args.uniprot_id}")
    print(f"Output will be saved to: {args.output_file}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")

    os.makedirs(args.cache_dir, exist_ok=True)

    print("\n=== Fetching Family Members ===")
    no_family_info = False
    try:
        family_members = get_protein_list(args.uniprot_id)
    except ValueError as e:
        print(f"Warning: {e}")
        print("No family information found; using direct annotations from the protein itself.")
        family_members = []
        no_family_info = True

    if len(family_members) == 0 and not no_family_info:
        print("No family members with PDB found. Searching without PDB requirement...")
        family_text = get_protein_family(args.uniprot_id)
        family_name = extract_family_name(family_text)
        is_superfamily = 'superfamily' in family_text.lower()
        suffixes = ['superfamily', 'family'] if is_superfamily else ['family', 'superfamily']

        for suffix in suffixes:
            params = {
                'query': f'family:"{family_name} {suffix}" AND reviewed:true',
                'fields': 'accession',
                'size': 50,
                'format': 'json'
            }
            response = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params)
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    family_members = [r['primaryAccession'] for r in results]
                    print(f"Found {len(family_members)} reviewed family members (max 50)")
                    break

        if len(family_members) == 0:
            for suffix in suffixes:
                params = {
                    'query': f'family:"{family_name} {suffix}"',
                    'fields': 'accession',
                    'size': 50,
                    'format': 'json'
                }
                response = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    if results:
                        family_members = [r['primaryAccession'] for r in results]
                        print(f"Found {len(family_members)} family members (reviewed + unreviewed, max 50)")
                        break

    annotations_df = aggregate_annotations_across_family(
        args.uniprot_id, family_members, args.cache_dir,
        confidence_threshold=0.7
    )

    features_df = generate_simple_feature_table(annotations_df, args.confidence_threshold)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    features_df.to_csv(args.output_file, sep='\t', index=False)

    print(f"\n{'='*80}")
    print(f"Output saved to: {args.output_file}")
    print(f"{'='*80}")

    print("\nFeature summary by category:")
    type_counts = features_df.groupby('functional_site_category').size().sort_values(ascending=False)
    for feat_type, count in type_counts.items():
        print(f"  {feat_type}: {count} unique features")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: nbhd-features
# ---------------------------------------------------------------------------

def cmd_nbhd_features(args):
    """
    CLI handler for `nbhd-features`. Loads a features TSV (from `annotate`),
    computes the structural neighborhood centered at --aa-pos, then cross-references
    the two. Writes two files to {results_dir}/neighborhoods/:
      - {gene}_{uniprot}_{center}_nbhd.tsv: per-residue mutation counts in the neighborhood
      - {gene}_{uniprot}_{center}_nbhd_features.tsv: overlap of each feature with the
        neighborhood, sorted by fraction of case-only mutations in feature
    """
    features_df = pd.read_csv(args.features, sep='\t')

    gene_name = _load_gene_map(args.reference_dir).get(args.uniprot, args.uniprot)
    center = int(args.aa_pos)

    nbhd = _compute_nbhd(args.uniprot, center, args.reference_dir, args.radius, args.pae_cutoff)
    print(f"Neighborhood centered at {gene_name} ({args.uniprot}) aa {center}: {len(nbhd)} residues")

    df_mut, mu_mode = _load_mut_data(args.pval_file, args.uniprot, args.test)
    if mu_mode:
        print(f"Mutation-rate results; using the {args.test} null.")

    nbhd_dir = os.path.join(args.results_dir, 'neighborhoods')
    os.makedirs(nbhd_dir, exist_ok=True)

    # Save neighborhood residue file
    second_col = 'mu' if mu_mode else 'ac_control'
    nbhd_path = os.path.join(nbhd_dir, f'{gene_name}_{args.uniprot}_{center}_nbhd.tsv')
    df_mut[df_mut['aa_pos'].isin(nbhd)][['aa_pos', 'ac_case', second_col]].to_csv(
        nbhd_path, sep='\t', index=False
    )
    print(f"Saved neighborhood residues to {nbhd_path}")

    result_df = _nbhd_features_table(nbhd, df_mut, features_df, mu_mode=mu_mode)

    if len(result_df) == 0:
        print("No features overlap with this neighborhood.")

    out_path = args.output or os.path.join(
        nbhd_dir, f'{gene_name}_{args.uniprot}_{center}_nbhd_features.tsv'
    )
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    result_df.to_csv(out_path, sep='\t', index=False)
    print(f"Saved {len(result_df)} overlapping features to {out_path}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: color
# ---------------------------------------------------------------------------

def _make_mut_pse(pymol_cmd, input_cif, chain, uniprot_id, pval_file, output_pse,
                  test='binomial'):
    """
    Create a _mut-style PSE: grey cartoon with mutation spheres on one chain.

    Standard 3DNT shows the case/control classes; a mutation-rate run has no controls and
    shows de novo positions instead, darkened where a residue carries more than one.
    """
    df_mut, mu_mode = _load_mut_data(pval_file, uniprot_id, test)

    pymol_cmd.reinitialize()
    pymol_cmd.load(input_cif, "complex")
    pymol_cmd.hide("everything", "all")
    pymol_cmd.show("cartoon", "all")
    pymol_cmd.show("spheres", "hetatm and not resn HOH")
    pymol_cmd.color("grey70", "all")
    pymol_cmd.bg_color("white")

    # Extract target chain so it can have independent transparency
    pymol_cmd.extract("target_chain", f"chain {chain}")
    pymol_cmd.set("cartoon_transparency", 0.6, "complex")
    pymol_cmd.set("cartoon_transparency", 0.0, "target_chain")

    pymol_cmd.set("cartoon_loop_radius", 0.4)
    pymol_cmd.set("specular", 0.0)
    pymol_cmd.set("ambient", 0.4)
    pymol_cmd.set("ray_trace_mode", 1)
    pymol_cmd.set("ray_trace_gain", 0.15)

    def _select_show(name, positions, color):
        if not positions:
            return
        resi_str = '+'.join(str(p) for p in positions)
        pymol_cmd.select(name, f"target_chain and resi {resi_str} and name CA")
        pymol_cmd.show("spheres", name)
        pymol_cmd.color(color, name)

    if mu_mode:
        _select_show("denovo",           df_mut[df_mut['ac_case'] == 1]['aa_pos'].tolist(), "salmon")
        _select_show("denovo_recurrent", df_mut[df_mut['ac_case'] >= 2]['aa_pos'].tolist(), "firebrick")
    else:
        control_mask = (df_mut['ac_control'] >= 1) & (df_mut['ac_case'] == 0)
        case_mask    = (df_mut['ac_case']    >= 1) & (df_mut['ac_control'] == 0)
        both_mask    = (df_mut['ac_case']    >= 1) & (df_mut['ac_control'] >= 1)

        _select_show("control_only",     df_mut[control_mask]['aa_pos'].tolist(), "blue")
        _select_show("case_only",        df_mut[case_mask]['aa_pos'].tolist(),    "red")
        _select_show("case_and_control", df_mut[both_mask]['aa_pos'].tolist(),    "purple")

    pymol_cmd.save(output_pse)
    print(f"[OK] Saved mutations session to {output_pse}")


def cmd_color(args):
    """
    CLI handler for `color`. Loads a CIF structure, greys out all chains, then
    colors the specified chain by per-residue case/control ratio using a
    blue (low) to red (high) spectrum stored as B-factors. Ratios are read
    from p_values.h5 (preferred — smooth gradient over all residues) or a TSV.
    Saves a PyMOL session file (.pse).
    """
    from pymol import cmd as pymol_cmd

    if not args.tsv and not args.pval_file:
        raise ValueError("Either --tsv or --pval-file must be provided")

    pymol_cmd.reinitialize()
    pymol_cmd.load(args.input_cif, "complex")
    pymol_cmd.hide("everything", "all")
    pymol_cmd.show("cartoon", "all")
    pymol_cmd.show("spheres", "hetatm and not resn HOH")
    pymol_cmd.color("grey70", "all")
    pymol_cmd.bg_color("white")

    # Extract the target chain into its own object so it can have independent transparency
    pymol_cmd.extract("target_chain", f"chain {args.chain}")
    pymol_cmd.set("cartoon_transparency", 0.6, "complex")
    pymol_cmd.set("cartoon_transparency", 0.0, "target_chain")

    residue_ratios = {}
    mu_mode = False

    if args.pval_file:
        import h5py
        import hdf5plugin  # noqa: F401
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils import read_p_values, resolve_pval_node
        pval_file = args.pval_file
        with h5py.File(pval_file, 'r') as fid:
            node, mu_mode = resolve_pval_node(fid, args.test)
            df_eff = read_p_values(node, args.uniprot, mu_mode=mu_mode)
        # obs/exp against the chosen null in mutation-rate mode; the regularized
        # case/control ratio otherwise.
        effect = df_eff['obs_exp'] if mu_mode else df_eff['ratio']
        residue_ratios = {int(pos): float(v)
                          for pos, v in zip(df_eff['aa_pos'], effect)}
    else:
        with open(args.tsv, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["uniprot_id"] == args.uniprot:
                    try:
                        residue_ratios[int(row["aa_pos"])] = float(row["ratio"])
                    except ValueError:
                        continue

    if not residue_ratios:
        raise RuntimeError(f"No ratio data found for UniProt ID {args.uniprot}")

    if mu_mode:
        from pymol_code import _obs_exp_to_b
        b_values = {resi: float(_obs_exp_to_b(v)) for resi, v in residue_ratios.items()}
        palette = "ratio_blue white ratio_red"
        spectrum_range = {'minimum': 0.0, 'maximum': 1.0}
    else:
        max_ratio = max(residue_ratios.values()) or 1.0
        b_values = {resi: (v / max_ratio) ** 2 for resi, v in residue_ratios.items()}
        palette = "ratio_blue ratio_red"
        spectrum_range = {}

    for resi, b in b_values.items():
        pymol_cmd.alter(f"target_chain and resi {resi}", f"b={b}")

    pymol_cmd.rebuild()
    pymol_cmd.set_color("ratio_blue", [0.45, 0.78, 1.0])
    pymol_cmd.set_color("ratio_red",  [1.0, 0.55, 0.55])
    pymol_cmd.spectrum("b", palette, "target_chain", byres=1, **spectrum_range)
    pymol_cmd.set("specular", 0.0)
    pymol_cmd.set("ambient", 0.4)
    pymol_cmd.set("ray_trace_mode", 1)
    pymol_cmd.set("ray_trace_gain", 0.15)
    pymol_cmd.set("cartoon_loop_radius", 0.4)
    pymol_cmd.save(args.output_pse)
    print(f"[OK] Saved colored session to {args.output_pse}")

    if args.pval_file:
        stem = args.output_pse[:-len("_ratio.pse")] if args.output_pse.endswith("_ratio.pse") else args.output_pse[:-4]
        mut_pse = stem + "_mut.pse"
        _make_mut_pse(pymol_cmd, args.input_cif, args.chain, args.uniprot, pval_file, mut_pse,
                      test=args.test)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: visualize
# ---------------------------------------------------------------------------

def cmd_visualize(args):
    """CLI handler for `visualize`. Delegates to pymol_code.run_all for one protein."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pymol_code import run_all
    run_all(args.uniprot, args.results_dir, args.reference_dir, pval_file=args.pval_file,
            test=args.test)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------

def _units_from_components(components_file, gene_map):
    """
    One unit per component, read straight from <stem>.components.tsv.

    The residue set is taken from the file rather than recomputed: a component is the
    union of many overlapping neighborhoods, the conditional analysis already masked
    exactly that set, and rebuilding it here from centres and radii would risk
    annotating a different region from the one that was conditioned on. It also means
    this path needs no structures at all.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import parse_pymol_positions
    from components import parse_centers_detail

    df = pd.read_csv(components_file, sep='\t')
    required = {'uniprot_id', 'component_id', 'residues_pymol', 'lead_aa_pos'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f'{components_file} is missing {sorted(missing)}; expected the '
            'components.tsv written by a --conditional run.'
        )

    units = []
    for row in df.sort_values(['round', 'lead_p'] if 'lead_p' in df.columns
                              else ['round']).itertuples():
        uniprot_id = row.uniprot_id
        gene_name = getattr(row, 'gene_name', None) or gene_map.get(uniprot_id, uniprot_id)
        residues = parse_pymol_positions(row.residues_pymol)
        lead = int(row.lead_aa_pos)
        rnd = int(getattr(row, 'round', 0))
        info = (f'component {row.component_id} (round {rnd}), lead {lead}, '
                f'{len(residues)} residues, {getattr(row, "n_centers", "?")} neighborhoods, '
                f'{getattr(row, "n_case_alleles", "?")} case alleles')
        units.append({
            'uniprot_id': uniprot_id,
            'gene_name': gene_name,
            'center': lead,
            'residues': residues,
            # member centres with their own radii, so PyMOL can redraw the component as
            # the union of spheres it actually is
            'centers': parse_centers_detail(row),
            'label': f'c{int(row.component_id)}_{lead}',
            'kind': 'component',
            'info': info,
        })
    return units


def _units_from_fdr(args, df_fdr, sig_col, p_col, radius_cols, gene_map):
    """One unit per called neighborhood (or per protein, without --all-signals)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import best_radius_for_hit

    df_sig = df_fdr[df_fdr[sig_col] < args.significance_cutoff].copy()
    if len(df_sig) == 0:
        return []

    if args.all_signals:
        # One row per independent signal, as written by --conditional. Collapsing to the
        # best per protein would throw away exactly what that analysis found.
        hits = df_sig.sort_values(p_col).reset_index(drop=True)
    else:
        hits = (df_sig.sort_values(p_col)
                      .groupby('uniprot_id', as_index=False)
                      .first())

    units = []
    for row in hits.itertuples():
        uniprot_id = row.uniprot_id
        center = int(row.aa_pos)
        # The scan records which radius it chose for this position under
        # --neighborhood-radius multiple-*; re-deriving the neighborhood at a fixed 15 A
        # would annotate a different neighborhood from the one that was tested.
        radius = best_radius_for_hit(df_fdr, uniprot_id, center,
                                     default=args.radius, radius_cols=radius_cols)
        units.append({
            'uniprot_id': uniprot_id,
            'gene_name': gene_map.get(uniprot_id, uniprot_id),
            'center': center,
            'residues': None,            # computed from the structure in the loop
            'radius': radius,
            'label': str(center),
            'kind': 'neighborhood',
            'info': (f'neighborhood centre {center}, radius {radius}, '
                     f'{sig_col}={getattr(row, sig_col):.4g} {p_col}={getattr(row, p_col):.4g}'),
        })
    return units


def cmd_analyze(args):
    """
    Full pipeline for every called region:
      1. Visualization (PyMOL session, once per protein)
      2. Annotation (functional sites, once per protein)
      3. Per-residue table for the region
      4. Region x features table

    The region is a component when --components is given, and a single neighborhood
    otherwise. A component is the unit the conditional analysis actually masked, so
    annotating it describes the whole signal rather than the one window that scored best.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    gene_map = _load_gene_map(args.reference_dir)

    if args.components:
        units = _units_from_components(args.components, gene_map)
        print(f"Components mode: {len(units)} components from {args.components}")
        if args.significance_column or args.all_signals:
            print("  Note: --significance-column and --all-signals do not apply with "
                  "--components; the file already holds the called components.")
    else:
        # Load FDR file and work out which layout it has. A mutation-rate run corrects
        # each test separately and reports the max, so it has no plain `fdr` or
        # `p_value` column.
        df_fdr = pd.read_csv(args.fdr_file, sep='\t')
        schema = _fdr_schema(df_fdr)
        sig_col = args.significance_column or schema['significance_column']
        p_col = schema['p_column'][args.test]
        radius_cols = schema['radius_columns'][args.test]
        if schema is MU_FDR_SCHEMA:
            print(f"Mutation-rate results detected; ranking by {p_col}, "
                  f"significance from {sig_col}.")
        for col in (sig_col, p_col):
            if col not in df_fdr.columns:
                raise ValueError(
                    f"Column '{col}' not found in {args.fdr_file}. "
                    f"Available: {list(df_fdr.columns)}"
                )
        units = _units_from_fdr(args, df_fdr, sig_col, p_col, radius_cols, gene_map)
        if not units:
            print(f"No proteins with {sig_col} < {args.significance_cutoff} "
                  f"in {args.fdr_file}")
            return 0

    annotations_dir = args.annotations_dir or os.path.join(args.results_dir, 'annotations')
    cache_dir = args.cache_dir or os.path.join(annotations_dir, 'cache')
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    pval_path = args.pval_file
    nbhd_dir = os.path.join(args.results_dir, 'neighborhoods')
    os.makedirs(nbhd_dir, exist_ok=True)

    # The _mut PSE is per protein but a protein can carry several components, so they
    # are collected up front and all drawn into it as separate toggleable objects.
    components_by_protein = defaultdict(list)
    for unit in units:
        if unit['kind'] == 'component' and unit.get('centers'):
            components_by_protein[unit['uniprot_id']].append(
                {'label': unit['label'], 'centers': unit['centers']})

    n = len(units)
    visualized = set()
    annotated = {}
    for i, unit in enumerate(units, 1):
        uniprot_id = unit['uniprot_id']
        gene_name = unit['gene_name']
        center = unit['center']
        kind = unit['kind']

        print(f"\n{'='*60}")
        print(f"[{i}/{n}] {gene_name} ({uniprot_id})")
        print(f"  {unit['info']}")
        print('='*60)

        # 1. Visualization. The whole-chain PSEs are per protein; the component PSE is
        # per component, since that is what it highlights.
        if not args.skip_visualization:
            if uniprot_id not in visualized:
                print("  Running visualization...")
                try:
                    from pymol_code import run_all
                    run_all(uniprot_id, args.results_dir, args.reference_dir,
                            pval_file=pval_path, test=args.test,
                            components=components_by_protein.get(uniprot_id))
                    print("  Visualization done.")
                except Exception as e:
                    print(f"  Warning: visualization failed: {e}")
                visualized.add(uniprot_id)
            if kind == 'component':
                try:
                    from pymol_code import pymol_component
                    pymol_component(uniprot_id, args.reference_dir, args.results_dir,
                                    pval_path, unit['residues'], unit['label'],
                                    gene_name=gene_name, test=args.test,
                                    centers=unit.get('centers'))
                except Exception as e:
                    print(f"  Warning: component visualization failed: {e}")

        # 2. Annotation (per protein -- several components can share one protein)
        features_path = os.path.join(
            annotations_dir, f'{uniprot_id}_features.tsv'
        )
        if uniprot_id in annotated:
            features_df = annotated[uniprot_id]
        else:
            print(f"  Annotating functional sites -> {features_path}")
            try:
                features_df = _annotate_and_save(
                    uniprot_id, features_path, cache_dir,
                    confidence_threshold=args.confidence_threshold
                )
            except Exception as e:
                print(f"  Warning: annotation failed: {e}")
                features_df = pd.DataFrame()
            annotated[uniprot_id] = features_df

        # 3. The region's residues. A component carries its own set; a neighborhood is
        # rebuilt from the structure at the radius its scan chose.
        if unit['residues'] is not None:
            nbhd = unit['residues']
            print(f"  Component has {len(nbhd)} residues")
        else:
            radius = unit['radius']
            print(f"  Computing neighborhood (center={center}, radius={radius}, "
                  f"pae={args.pae_cutoff})...")
            try:
                nbhd = _compute_nbhd(
                    uniprot_id, center, args.reference_dir,
                    radius, args.pae_cutoff
                )
                print(f"  Neighborhood has {len(nbhd)} residues")
            except Exception as e:
                print(f"  Warning: neighborhood computation failed: {e}")
                continue

        # 4. Load per-residue mutation data
        try:
            df_mut, mu_mode = _load_mut_data(pval_path, uniprot_id, args.test)
        except Exception as e:
            print(f"  Warning: could not read mutation data: {e}")
            continue

        # 5. Per-residue table for the region
        second_col = 'mu' if mu_mode else 'ac_control'
        stem = f'{gene_name}_{uniprot_id}_{unit["label"]}'
        suffix = 'component' if kind == 'component' else 'nbhd'
        res_path = os.path.join(nbhd_dir, f'{stem}_{suffix}.tsv')
        df_mut[df_mut['aa_pos'].isin(nbhd)][['aa_pos', 'ac_case', second_col]].to_csv(
            res_path, sep='\t', index=False
        )
        print(f"  Saved {kind} residues to {res_path}")

        # 6. Features table (skip if no features)
        if len(features_df) == 0:
            print(f"  No features available; skipping {kind}-features table.")
            continue

        result_df = _nbhd_features_table(nbhd, df_mut, features_df, mu_mode=mu_mode)
        feat_path = os.path.join(nbhd_dir, f'{stem}_{suffix}_features.tsv')
        result_df.to_csv(feat_path, sep='\t', index=False)
        print(f"  Saved {len(result_df)} overlapping features to {feat_path}")

    label = ('components' if args.components
             else ('signals' if args.all_signals else 'proteins'))
    print(f"\nDone. Processed {n} {label}.")
    return 0


def pse_to_png(input_pse, ray_trace_gain=0.15):
    from pymol import cmd

    output_png = input_pse.replace(".pse", ".png")
    print(f"[INFO] Loading PSE: {input_pse}")
    print(f"[INFO] Output PNG:  {output_png}")

    cmd.load(input_pse)

    # Rendering settings
    cmd.set("ambient", 0.5)
    cmd.set("ray_shadows", 0)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_gain", ray_trace_gain)
    cmd.set("ray_opaque_background", 1)
    cmd.bg_color("white")

    width  = 2400
    height = 1800
    dpi    = 300

    cmd.ray(width, height)
    cmd.png(output_png, width, height, dpi, 0, 0)

    print(f"[INFO] Saved PNG to: {output_png}")
    cmd.quit(0)


def pse_to_gif(input_pse, n_frames=36, width=640, height=480, fps=15, ray_trace_gain=0.15):
    from pymol import cmd
    import tempfile

    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] Pillow is required: pip install Pillow")
        return

    output_gif = input_pse.replace(".pse", ".gif")
    print(f"[INFO] Loading PSE: {input_pse}")
    print(f"[INFO] Output GIF:  {output_gif} ({n_frames} frames at {fps} fps)")

    cmd.load(input_pse)
    cmd.set("ambient", 0.5)
    cmd.set("ray_shadows", 0)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_gain", ray_trace_gain)
    cmd.set("ray_opaque_background", 1)
    cmd.bg_color("white")

    tmpdir = tempfile.mkdtemp()
    frame_paths = []
    step = 360.0 / n_frames

    try:
        for i in range(n_frames):
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            print(f"[INFO] Rendering frame {i + 1}/{n_frames}...")
            cmd.png(frame_path, width, height, dpi=72, ray=1, quiet=1)
            frame_paths.append(frame_path)
            cmd.turn("y", step)

        print("[INFO] Stitching frames into GIF...")
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        duration_ms = int(1000 / fps)
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=duration_ms,
            optimize=False,
        )
    finally:
        for p in frame_paths:
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(tmpdir):
            os.rmdir(tmpdir)

    print(f"[INFO] Saved GIF to: {output_gif}")
    cmd.quit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Protein structure interpretation tools',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_test_arg(p):
        """--test selects the null for a mutation-rate p-value file; ignored otherwise."""
        p.add_argument('--test', choices=['binomial', 'poisson'], default='binomial',
                       help='For mutation-rate results, which null drives the output: '
                            'binomial = Test B, the within-gene comparison and the test '
                            'of interest (default); poisson = Test A, the regional-'
                            'constraint guard, whose standalone significance is not '
                            'calibrated. Ignored for standard 3DNT results.')

    # -- annotate subcommand --
    p_annotate = subparsers.add_parser(
        'annotate',
        help='Transfer functional site annotations from protein family members',
        description='Transfers functional site annotations from protein family members and outputs '
                    'a table showing each unique feature with its positions.'
    )
    p_annotate.add_argument('uniprot_id', help='UniProt ID of protein to annotate')
    p_annotate.add_argument('output_file', help='Output TSV file path')
    p_annotate.add_argument('--cache-dir', default='cache', help='Cache directory (default: cache)')
    p_annotate.add_argument('--confidence-threshold', type=float, default=0.8,
                            help='Minimum alignment confidence to include (default: 0.8)')

    # -- nbhd-features subcommand --
    p_nbhd = subparsers.add_parser(
        'nbhd-features',
        help='Cross-reference a neighborhood with a functional features TSV',
        description='Identifies which functional features (from interpret.py annotate) '
                    'overlap with a structural neighborhood centered at a given residue.'
    )
    p_nbhd.add_argument('--features', required=True,
                        help='Features TSV from interpret.py annotate')
    p_nbhd.add_argument('--reference-dir', required=True,
                        help='Reference directory (same as for run.py)')
    p_nbhd.add_argument('--results-dir', required=True,
                        help='Results directory (for output files)')
    p_nbhd.add_argument('--pval-file', required=True,
                        help='Full path to p_values.h5 HDF5 file')
    p_nbhd.add_argument('--uniprot', required=True, help='UniProt ID')
    p_nbhd.add_argument('--aa-pos', required=True, type=int,
                        help='Neighborhood center residue position')
    p_nbhd.add_argument('--output', default=None,
                        help='Output TSV file; if omitted, prints to stdout')
    p_nbhd.add_argument('--radius', type=float, default=15.0,
                        help='Neighborhood radius in Angstroms (default: 15.0)')
    p_nbhd.add_argument('--pae-cutoff', type=float, default=15.0,
                        help='Maximum PAE value (default: 15.0)')
    add_test_arg(p_nbhd)

    # -- visualize subcommand --
    p_vis = subparsers.add_parser(
        'visualize',
        help='Generate PyMOL visualizations for a protein (wraps run.py --visualization)',
    )
    p_vis.add_argument('--uniprot', required=True, help='UniProt ID')
    p_vis.add_argument('--results-dir', required=True, help='Results directory')
    p_vis.add_argument('--reference-dir', required=True, help='Reference directory')
    p_vis.add_argument('--pval-file', required=True, help='Full path to p_values.h5 HDF5 file')
    add_test_arg(p_vis)

    # -- run_all subcommand --
    p_analyze = subparsers.add_parser(
        'run_all',
        help='Run the full interpretation pipeline for every significant protein in an FDR file',
        description='For each significant protein: visualization, annotation, top-neighborhood '
                    'residue table, and neighborhood × features table.',
    )
    p_analyze.add_argument('--fdr-file', default=None,
                           help='FDR results TSV (output of run.py --run-3dnt), or the '
                                '.conditional.tsv from a --conditional run. Required '
                                'unless --components is given')
    p_analyze.add_argument('--components', default=None,
                           help='<stem>.components.tsv from a --conditional run. Switches '
                                'the unit of analysis from one neighborhood to one '
                                'component: the whole set of overlapping neighborhoods '
                                'reporting a signal, which is what the conditional '
                                'analysis masked. Residues come from the file, so no '
                                'structure is needed to rebuild them')
    p_analyze.add_argument('--results-dir', required=True,
                           help='Results directory (must contain p_values.h5)')
    p_analyze.add_argument('--reference-dir', required=True,
                           help='Reference directory')
    p_analyze.add_argument('--significance-column', default=None,
                           help='Column to use for significance cutoff. Default depends '
                                'on the file: "fdr" for standard 3DNT results, "fdr_max" '
                                'for mutation-rate results (a neighborhood passes only '
                                'if it passes both tests)')
    p_analyze.add_argument('--all-signals', action='store_true', default=False,
                           help='Process every significant row rather than only the best '
                                'neighborhood per protein. Use with the .conditional.tsv '
                                'from a --conditional run, which lists one row per '
                                'independent signal')
    p_analyze.add_argument('--significance-cutoff', type=float, default=0.05,
                           help='Significance cutoff (default: 0.05)')
    p_analyze.add_argument('--annotations-dir', default=None,
                           help='Directory for annotation files (default: {results-dir}/annotations)')
    p_analyze.add_argument('--cache-dir', default=None,
                           help='Cache directory for UniProt API calls (default: {annotations-dir}/cache)')
    p_analyze.add_argument('--pval-file', required=True,
                           help='Full path to p_values.h5 HDF5 file')
    p_analyze.add_argument('--radius', type=float, default=15.0,
                           help='Fallback neighborhood radius in Angstroms (default: 15.0). '
                                'The radius the scan recorded for each hit is used when '
                                'the results file has one')
    p_analyze.add_argument('--pae-cutoff', type=float, default=15.0,
                           help='Maximum PAE value (default: 15.0)')
    p_analyze.add_argument('--confidence-threshold', type=float, default=0.8,
                           help='Alignment confidence threshold for annotation (default: 0.8)')
    p_analyze.add_argument('--skip-visualization', action='store_true', default=False,
                           help='Skip PyMOL visualization step')
    add_test_arg(p_analyze)

    # -- color subcommand --
    p_color = subparsers.add_parser(
        'color',
        help='Color a chain in a PyMOL session by per-residue ratio values',
        description='Colors a specific chain in an mmCIF structure by per-residue ratio values '
                    'and saves a PyMOL session file.'
    )
    p_color.add_argument('--tsv', default=None,
                         help='TSV file with uniprot_id, aa_pos, ratio columns')
    p_color.add_argument('--pval-file', default=None,
                         help='Full path to p_values.h5 HDF5 file (preferred over --tsv for smooth gradients)')
    p_color.add_argument('--input-cif', required=True, help='Input structure (.cif)')
    p_color.add_argument('--output-pse', required=True, help='Output PyMOL session (.pse)')
    p_color.add_argument('--chain', required=True, help='Target chain ID (e.g. A)')
    p_color.add_argument('--uniprot', required=True, help='Target UniProt ID')
    add_test_arg(p_color)

    # -- pse to png --
    p_pse_to_png = subparsers.add_parser(
        'pse_to_png',
        help='Convert a PyMOL session file (.pse) to a PNG image',
        description='Loads a PyMOL session file and saves a PNG image of the current view.'
    )
    p_pse_to_png.add_argument('input_pse', help='Input PyMOL session file (.pse)')
    p_pse_to_png.add_argument('--ray-trace-gain', type=float, default=0.15,
                               help='Outline thickness (default: 0.15; use smaller for large complexes)')

    p_pse_to_gif = subparsers.add_parser(
        'pse_to_gif',
        help='Convert a PyMOL session file (.pse) to a rotating GIF',
        description='Loads a PyMOL session file and saves a GIF of a full y-axis rotation.'
    )
    p_pse_to_gif.add_argument('input_pse', help='Input PyMOL session file (.pse)')
    p_pse_to_gif.add_argument('--n-frames', type=int, default=36,
                               help='Number of frames for one full rotation (default: 36)')
    p_pse_to_gif.add_argument('--width', type=int, default=640,
                               help='Frame width in pixels (default: 640)')
    p_pse_to_gif.add_argument('--height', type=int, default=480,
                               help='Frame height in pixels (default: 480)')
    p_pse_to_gif.add_argument('--fps', type=int, default=15,
                               help='Frames per second (default: 15)')
    p_pse_to_gif.add_argument('--ray-trace-gain', type=float, default=0.15,
                               help='Outline thickness (default: 0.15; use smaller for large complexes)')

    args = parser.parse_args()

    if args.command == 'run_all' and not (args.fdr_file or args.components):
        parser.error('run_all needs --fdr-file or --components')

    if args.command == 'annotate':
        return cmd_annotate(args)
    elif args.command == 'nbhd-features':
        return cmd_nbhd_features(args)
    elif args.command == 'visualize':
        return cmd_visualize(args)
    elif args.command == 'run_all':
        return cmd_analyze(args)
    elif args.command == 'color':
        return cmd_color(args)
    elif args.command == 'pse_to_png':
        return pse_to_png(args.input_pse, args.ray_trace_gain)
    elif args.command == 'pse_to_gif':
        return pse_to_gif(args.input_pse, args.n_frames, args.width, args.height, args.fps, args.ray_trace_gain)


if __name__ == '__main__':
    sys.exit(main())
