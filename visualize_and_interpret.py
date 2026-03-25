#!/usr/bin/env python3
"""
visualize_and_interpret.py - Protein structure visualization and interpretation tools

Subcommands:
  run_all    Full pipeline (visualize, annotate, neighborhoods) for all significant proteins
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
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
from Bio.Align import PairwiseAligner


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


def _nbhd_features_table(nbhd, df_mut, features_df):
    """
    Cross-reference `nbhd` (set of residue positions) with `features_df`
    (output of generate_simple_feature_table). `df_mut` must have columns
    aa_pos, ac_case, ac_control with per-residue mutation counts.

    For each feature that overlaps the neighborhood, computes:
      - prop_nbhd_in_feature: fraction of neighborhood residues in the feature
      - prop_feature_in_nbhd: fraction of feature residues in the neighborhood
      - prop_case_only_in_feature: fraction of case-only alleles (in the nbhd)
        that fall inside the feature
      - case_only_positions: comma-separated list of case-only-mutated positions
        inside the feature

    Returns a DataFrame sorted descending by prop_case_only_in_feature.
    """
    df_nbhd = df_mut[df_mut['aa_pos'].isin(nbhd)]
    df_case_only = df_nbhd[(df_nbhd['ac_case'] > 0) & (df_nbhd['ac_control'] == 0)]
    n_case_only_nbhd = int(df_case_only['ac_case'].sum())
    rows = []
    for _, row in features_df.iterrows():
        feature_pos = parse_pymol_positions(row['residues_pymol'])
        overlap = nbhd & feature_pos
        if not overlap:
            continue
        df_case_in_feature = df_case_only[df_case_only['aa_pos'].isin(overlap)]
        n_case_in_feature = int(df_case_in_feature['ac_case'].sum())
        case_positions = sorted(df_case_in_feature['aa_pos'].tolist())
        rows.append({
            'feature_name': row['feature_name'],
            'functional_site_category': row['functional_site_category'],
            'prop_nbhd_in_feature': round(len(overlap) / len(nbhd), 3),
            'prop_feature_in_nbhd': round(len(overlap) / row['num_positions'], 3),
            'prop_case_only_in_feature': round(n_case_in_feature / n_case_only_nbhd, 3) if n_case_only_nbhd > 0 else None,
            'case_only_positions': ','.join(str(p) for p in case_positions) if case_positions else '',
        })
    result_df = pd.DataFrame(rows)
    if len(result_df) > 0:
        result_df = result_df.sort_values('prop_case_only_in_feature', ascending=False)
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
    import h5py
    import hdf5plugin
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import read_original_mutation_data

    features_df = pd.read_csv(args.features, sep='\t')

    gene_name = _load_gene_map(args.reference_dir).get(args.uniprot, args.uniprot)
    center = int(args.aa_pos)

    nbhd = _compute_nbhd(args.uniprot, center, args.reference_dir, args.radius, args.pae_cutoff)
    print(f"Neighborhood centered at {gene_name} ({args.uniprot}) aa {center}: {len(nbhd)} residues")

    pval_file = os.path.join(args.results_dir, args.pval_file)
    with h5py.File(pval_file, 'r') as fid:
        df_mut = read_original_mutation_data(fid, args.uniprot)

    nbhd_dir = os.path.join(args.results_dir, 'neighborhoods')
    os.makedirs(nbhd_dir, exist_ok=True)

    # Save neighborhood residue file
    nbhd_path = os.path.join(nbhd_dir, f'{gene_name}_{args.uniprot}_{center}_nbhd.tsv')
    df_mut[df_mut['aa_pos'].isin(nbhd)][['aa_pos', 'ac_case', 'ac_control']].to_csv(
        nbhd_path, sep='\t', index=False
    )
    print(f"Saved neighborhood residues to {nbhd_path}")

    result_df = _nbhd_features_table(nbhd, df_mut, features_df)

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

def cmd_color(args):
    """
    CLI handler for `color`. Loads a CIF structure, greys out all chains, then
    colors the specified chain by per-residue case/control ratio using a
    green (low) to red (high) spectrum stored as B-factors. Ratios are read
    from p_values.h5 (preferred — smooth gradient over all residues) or a TSV.
    Saves a PyMOL session file (.pse).
    """
    from pymol import cmd as pymol_cmd

    if not args.tsv and not args.results_dir:
        raise ValueError("Either --tsv or --results-dir must be provided")

    pymol_cmd.reinitialize()
    pymol_cmd.load(args.input_cif, "complex")
    pymol_cmd.hide("everything", "all")
    pymol_cmd.show("cartoon", "all")
    pymol_cmd.color("grey70", "all")
    pymol_cmd.set("cartoon_transparency", 0.6, "all")
    pymol_cmd.bg_color("white")

    residue_ratios = {}

    if args.results_dir:
        import h5py
        import hdf5plugin
        pval_file = os.path.join(args.results_dir, args.pval_file)
        with h5py.File(pval_file, 'r') as fid:
            case_control = fid[f'{args.uniprot}_nbhd'][:]
        nbhd_case = case_control[:, 0]
        nbhd_control = case_control[:, 1]
        n_case_total = nbhd_case.sum()
        n_control_total = nbhd_control.sum()
        ratio = (nbhd_case + 2) / (nbhd_control + 2 * n_control_total / n_case_total)
        for pos_idx, r in enumerate(ratio):
            residue_ratios[pos_idx + 1] = float(r)
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

    max_ratio = max(residue_ratios.values()) or 1.0
    for resi, ratio in residue_ratios.items():
        pymol_cmd.alter(f"chain {args.chain} and resi {resi}", f"b={ratio / max_ratio}")

    pymol_cmd.rebuild()
    pymol_cmd.spectrum("b", "green_red", f"chain {args.chain}", byres=1)
    pymol_cmd.set("cartoon_transparency", 0.0, f"chain {args.chain}")
    pymol_cmd.save(args.output_pse)

    print(f"[OK] Saved colored session to {args.output_pse}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: visualize
# ---------------------------------------------------------------------------

def cmd_visualize(args):
    """CLI handler for `visualize`. Delegates to pymol_code.run_all for one protein."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pymol_code import run_all
    run_all(args.uniprot, args.results_dir, args.reference_dir)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------

def cmd_analyze(args):
    """
    Full pipeline for every significant protein in an FDR results file:
      1. Visualization (PyMOL session)
      2. Annotation (functional sites)
      3. Top-neighborhood residue table (_nbhd.tsv)
      4. Neighborhood × features table (_nbhd_features.tsv)
    """
    import h5py
    import hdf5plugin
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import read_original_mutation_data
    from pymol_code import run_all

    # Load FDR file and filter to significant proteins
    df_fdr = pd.read_csv(args.fdr_file, sep='\t')
    sig_col = args.significance_column
    if sig_col not in df_fdr.columns:
        raise ValueError(
            f"Column '{sig_col}' not found in {args.fdr_file}. "
            f"Available: {list(df_fdr.columns)}"
        )

    df_sig = df_fdr[df_fdr[sig_col] < args.significance_cutoff].copy()
    if len(df_sig) == 0:
        print(f"No proteins with {sig_col} < {args.significance_cutoff} in {args.fdr_file}")
        return 0

    # For each protein, pick the neighborhood with the lowest p_value
    top_per_protein = (
        df_sig.sort_values('p_value')
              .groupby('uniprot_id', as_index=False)
              .first()
    )

    gene_map = _load_gene_map(args.reference_dir)
    annotations_dir = args.annotations_dir or os.path.join(args.results_dir, 'annotations')
    cache_dir = args.cache_dir or os.path.join(annotations_dir, 'cache')
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    pval_path = os.path.join(args.results_dir, args.pval_file)
    nbhd_dir = os.path.join(args.results_dir, 'neighborhoods')
    os.makedirs(nbhd_dir, exist_ok=True)

    n = len(top_per_protein)
    for i, row in enumerate(top_per_protein.itertuples(), 1):
        uniprot_id = row.uniprot_id
        center = int(row.aa_pos)
        gene_name = gene_map.get(uniprot_id, uniprot_id)

        print(f"\n{'='*60}")
        print(f"[{i}/{n}] {gene_name} ({uniprot_id})  top neighborhood center: {center}")
        print(f"  {sig_col}={getattr(row, sig_col):.4g}  p={row.p_value:.4g}")
        print('='*60)

        # 1. Visualization
        if not args.skip_visualization:
            print("  Running visualization...")
            try:
                run_all(uniprot_id, args.results_dir, args.reference_dir)
                print("  Visualization done.")
            except Exception as e:
                print(f"  Warning: visualization failed: {e}")

        # 2. Annotation
        features_path = os.path.join(
            annotations_dir, f'{uniprot_id}_features.tsv'
        )
        print(f"  Annotating functional sites -> {features_path}")
        try:
            features_df = _annotate_and_save(
                uniprot_id, features_path, cache_dir,
                confidence_threshold=args.confidence_threshold
            )
        except Exception as e:
            print(f"  Warning: annotation failed: {e}")
            features_df = pd.DataFrame()

        # 3. Compute neighborhood
        print(f"  Computing neighborhood (center={center}, radius={args.radius}, pae={args.pae_cutoff})...")
        try:
            nbhd = _compute_nbhd(
                uniprot_id, center, args.reference_dir,
                args.radius, args.pae_cutoff
            )
            print(f"  Neighborhood has {len(nbhd)} residues")
        except Exception as e:
            print(f"  Warning: neighborhood computation failed: {e}")
            continue

        # 4. Load per-residue mutation data
        try:
            with h5py.File(pval_path, 'r') as fid:
                df_mut = read_original_mutation_data(fid, uniprot_id)
        except Exception as e:
            print(f"  Warning: could not read mutation data: {e}")
            continue

        # 5. Save _nbhd.tsv
        nbhd_path = os.path.join(nbhd_dir, f'{gene_name}_{uniprot_id}_{center}_nbhd.tsv')
        df_mut[df_mut['aa_pos'].isin(nbhd)][['aa_pos', 'ac_case', 'ac_control']].to_csv(
            nbhd_path, sep='\t', index=False
        )
        print(f"  Saved neighborhood residues to {nbhd_path}")

        # 6. _nbhd_features.tsv (skip if no features)
        if len(features_df) == 0:
            print("  No features available; skipping nbhd-features table.")
            continue

        result_df = _nbhd_features_table(nbhd, df_mut, features_df)
        nbhd_feat_path = os.path.join(nbhd_dir, f'{gene_name}_{uniprot_id}_{center}_nbhd_features.tsv')
        result_df.to_csv(nbhd_feat_path, sep='\t', index=False)
        print(f"  Saved {len(result_df)} overlapping features to {nbhd_feat_path}")

    print(f"\nDone. Processed {n} proteins.")
    return 0


def pse_to_png(input_pse):
    from pymol import cmd

    output_png = input_pse.replace(".pse", ".png")
    print(f"[INFO] Loading PSE: {input_pse}")
    print(f"[INFO] Output PNG:  {output_png}")

    cmd.load(input_pse)

    # Rendering settings
    cmd.set("ambient", 0.5)
    cmd.set("ray_shadows", 0)
    cmd.set("ray_trace_mode", 0.2)
    cmd.set("ray_trace_gain", 0.05)
    cmd.set("ray_opaque_background", 1)
    cmd.bg_color("white")

    width  = 2400
    height = 1800
    dpi    = 300

    cmd.ray(width, height)
    cmd.png(output_png, width, height, dpi, 0, 0)

    print(f"[INFO] Saved PNG to: {output_png}")
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
                        help='Results directory containing p_values.h5')
    p_nbhd.add_argument('--pval-file', default='p_values.h5',
                        help='HDF5 filename within --results-dir (default: p_values.h5)')
    p_nbhd.add_argument('--uniprot', required=True, help='UniProt ID')
    p_nbhd.add_argument('--aa-pos', required=True, type=int,
                        help='Neighborhood center residue position')
    p_nbhd.add_argument('--output', default=None,
                        help='Output TSV file; if omitted, prints to stdout')
    p_nbhd.add_argument('--radius', type=float, default=15.0,
                        help='Neighborhood radius in Angstroms (default: 15.0)')
    p_nbhd.add_argument('--pae-cutoff', type=float, default=15.0,
                        help='Maximum PAE value (default: 15.0)')

    # -- visualize subcommand --
    p_vis = subparsers.add_parser(
        'visualize',
        help='Generate PyMOL visualizations for a protein (wraps run.py --visualization)',
    )
    p_vis.add_argument('--uniprot', required=True, help='UniProt ID')
    p_vis.add_argument('--results-dir', required=True, help='Results directory')
    p_vis.add_argument('--reference-dir', required=True, help='Reference directory')

    # -- run_all subcommand --
    p_analyze = subparsers.add_parser(
        'run_all',
        help='Run the full interpretation pipeline for every significant protein in an FDR file',
        description='For each significant protein: visualization, annotation, top-neighborhood '
                    'residue table, and neighborhood × features table.',
    )
    p_analyze.add_argument('--fdr-file', required=True,
                           help='FDR results TSV (output of run.py --run-3dnt)')
    p_analyze.add_argument('--results-dir', required=True,
                           help='Results directory (must contain p_values.h5)')
    p_analyze.add_argument('--reference-dir', required=True,
                           help='Reference directory')
    p_analyze.add_argument('--significance-column', default='fdr',
                           help='Column to use for significance cutoff (default: fdr)')
    p_analyze.add_argument('--significance-cutoff', type=float, default=0.05,
                           help='Significance cutoff (default: 0.05)')
    p_analyze.add_argument('--annotations-dir', default=None,
                           help='Directory for annotation files (default: {results-dir}/annotations)')
    p_analyze.add_argument('--cache-dir', default=None,
                           help='Cache directory for UniProt API calls (default: {annotations-dir}/cache)')
    p_analyze.add_argument('--pval-file', default='p_values.h5',
                           help='HDF5 filename within --results-dir (default: p_values.h5)')
    p_analyze.add_argument('--radius', type=float, default=15.0,
                           help='Neighborhood radius in Angstroms (default: 15.0)')
    p_analyze.add_argument('--pae-cutoff', type=float, default=15.0,
                           help='Maximum PAE value (default: 15.0)')
    p_analyze.add_argument('--confidence-threshold', type=float, default=0.8,
                           help='Alignment confidence threshold for annotation (default: 0.8)')
    p_analyze.add_argument('--skip-visualization', action='store_true', default=False,
                           help='Skip PyMOL visualization step')

    # -- color subcommand --
    p_color = subparsers.add_parser(
        'color',
        help='Color a chain in a PyMOL session by per-residue ratio values',
        description='Colors a specific chain in an mmCIF structure by per-residue ratio values '
                    'and saves a PyMOL session file.'
    )
    p_color.add_argument('--tsv', default=None,
                         help='TSV file with uniprot_id, aa_pos, ratio columns')
    p_color.add_argument('--results-dir', default=None,
                         help='Results directory containing p_values.h5 (all-residue ratios; '
                              'preferred over --tsv for smooth gradients)')
    p_color.add_argument('--pval-file', default='p_values.h5',
                         help='HDF5 filename within --results-dir (default: p_values.h5)')
    p_color.add_argument('--input-cif', required=True, help='Input structure (.cif)')
    p_color.add_argument('--output-pse', required=True, help='Output PyMOL session (.pse)')
    p_color.add_argument('--chain', required=True, help='Target chain ID (e.g. A)')
    p_color.add_argument('--uniprot', required=True, help='Target UniProt ID')

    # -- pse to png --
    p_pse_to_png = subparsers.add_parser(
        'pse_to_png',
        help='Convert a PyMOL session file (.pse) to a PNG image',
        description='Loads a PyMOL session file and saves a PNG image of the current view.'
    )
    p_pse_to_png.add_argument('input_pse', help='Input PyMOL session file (.pse)')

    args = parser.parse_args()

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
        return pse_to_png(args.input_pse)


if __name__ == '__main__':
    sys.exit(main())
