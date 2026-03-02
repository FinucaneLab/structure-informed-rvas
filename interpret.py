#!/usr/bin/env python3
"""
interpret.py - Protein structure interpretation tools

Subcommands:
  annotate   Transfer functional site annotations from protein family members
  color      Color a chain in a PyMOL session by per-residue ratio values

Usage:
  python interpret.py annotate <uniprot_id> <output_tsv> [options]
  python interpret.py color --tsv <tsv> --input-cif <cif> --output-pse <pse> --chain <chain> --uniprot <id>
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
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(seq1, seq2)
    return alignments[0]


def alignment_to_strings(alignment):
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
    current_query_pos = 0
    for align_pos in range(len(aligned_seq)):
        if aligned_seq[align_pos] != '-':
            current_query_pos += 1
        if current_query_pos == query_pos:
            return align_pos
    return None


def calculate_local_identity(alignment, position, window=5):
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
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)
    align_pos = get_alignment_position(aligned_seq1, position)
    if align_pos is None:
        return 0.0
    if aligned_seq1[align_pos] == '-' or aligned_seq2[align_pos] == '-':
        return 0.0
    return 1.0 if aligned_seq1[align_pos] == aligned_seq2[align_pos] else 0.0


def calculate_confidence_score(alignment, position):
    local_identity = calculate_local_identity(alignment, position)
    gap_density = calculate_gap_density(alignment, position)
    position_score = calculate_position_score(alignment, position)
    return 0.4 * local_identity + 0.3 * (1 - gap_density) + 0.3 * position_score


def get_aligned_position(alignment, query_pos):
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)
    align_pos = get_alignment_position(aligned_seq1, query_pos)
    if align_pos is None:
        return None
    if aligned_seq2[align_pos] == '-':
        return None
    target_pos = sum(1 for i in range(align_pos + 1) if aligned_seq2[i] != '-')
    return target_pos


def transfer_all_annotations(query_seq, target_seq, target_annotations):
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
    if not family_text:
        return None
    match = re.search(r'Belongs to the (.+?)(super)?family', family_text)
    if match:
        return match.group(1).strip()
    return None


def search_family_members_with_annotations(family_name, is_superfamily=False):
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
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}")
    response.raise_for_status()
    return response.json().get('sequence', {}).get('value', '')


def get_protein_function(uniprot_id):
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}")
    response.raise_for_status()
    try:
        return response.json()['comments'][0]['texts'][0]['value']
    except (KeyError, IndexError):
        return None


def get_functional_sites_detailed(uniprot_id, cache_dir=None):
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
            'Region', 'Domain', 'Topological domain',
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
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"alignment_{query_id}_{target_id}.json")
    with open(cache_file, 'w') as f:
        json.dump({'seqA': aligned_seq1, 'seqB': aligned_seq2}, f)


def load_alignment_cache(query_id, target_id, cache_dir):
    cache_file = os.path.join(cache_dir, f"alignment_{query_id}_{target_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Annotation transfer logic
# ---------------------------------------------------------------------------

def filter_family_by_annotation_count(family_members, cache_dir, max_members=50):
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


def positions_to_pymol_notation(positions):
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
    print(f"Simple functional site annotation for {args.uniprot_id}")
    print(f"Output will be saved to: {args.output_file}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")

    os.makedirs(args.cache_dir, exist_ok=True)

    print("\n=== Fetching Family Members ===")
    family_members = get_protein_list(args.uniprot_id)

    if len(family_members) == 0:
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
# Subcommand: color
# ---------------------------------------------------------------------------

def cmd_color(args):
    from pymol import cmd as pymol_cmd

    pymol_cmd.reinitialize()
    pymol_cmd.load(args.input_cif, "complex")
    pymol_cmd.hide("everything", "all")
    pymol_cmd.show("cartoon", "all")
    pymol_cmd.color("grey70", "all")
    pymol_cmd.set("cartoon_transparency", 0.6, "all")
    pymol_cmd.bg_color("white")

    residue_ratios = {}
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

    # -- color subcommand --
    p_color = subparsers.add_parser(
        'color',
        help='Color a chain in a PyMOL session by per-residue ratio values',
        description='Colors a specific chain in an mmCIF structure by per-residue ratio values '
                    'and saves a PyMOL session file.'
    )
    p_color.add_argument('--tsv', required=True, help='TSV file with uniprot_id, aa_pos, ratio columns')
    p_color.add_argument('--input-cif', required=True, help='Input structure (.cif)')
    p_color.add_argument('--output-pse', required=True, help='Output PyMOL session (.pse)')
    p_color.add_argument('--chain', required=True, help='Target chain ID (e.g. A)')
    p_color.add_argument('--uniprot', required=True, help='Target UniProt ID')

    args = parser.parse_args()

    if args.command == 'annotate':
        return cmd_annotate(args)
    elif args.command == 'color':
        return cmd_color(args)


if __name__ == '__main__':
    sys.exit(main())
