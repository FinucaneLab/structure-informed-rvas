#!/usr/bin/env python3
"""
Simple Functional Site Annotation Tool - No LLM Classification

Transfers functional site annotations from protein family members and outputs
a table showing each unique feature with its positions.

Usage:
    python annotate_functional_sites_simple.py <uniprot_id> <output_tsv> [options]

Example:
    python annotate_functional_sites_simple.py P42263 P42263_features_simple.tsv --cache-dir ./cache
"""

import argparse
import os
import sys
import pandas as pd
from collections import defaultdict

from alignment_utils import (
    get_pairwise_alignment,
    alignment_to_strings,
    transfer_all_annotations
)
from uniprot_api import (
    get_protein_list,
    get_uniprot_sequence,
    get_protein_function,
    get_functional_sites_detailed,
    save_alignment_cache,
    load_alignment_cache
)


def filter_family_by_annotation_count(family_members, cache_dir, max_members=50):
    """
    Filter family members to the top N with the most functional site annotations.

    This is useful when there are many family members but no PDB structures,
    or when you want to focus on the most well-annotated proteins.

    Args:
        family_members: List of UniProt IDs
        cache_dir: Cache directory for functional sites
        max_members: Maximum number of members to return (default: 50)

    Returns:
        List of UniProt IDs with the most annotations
    """
    functional_cache = os.path.join(cache_dir, 'functional_sites')
    os.makedirs(functional_cache, exist_ok=True)

    print(f"\nFiltering {len(family_members)} family members to top {max_members} by annotation count...")

    # Count annotations for each family member
    annotation_counts = []
    for member_id in family_members:
        try:
            annotations = get_functional_sites_detailed(member_id, functional_cache)
            count = len(annotations)
            annotation_counts.append((member_id, count))
        except Exception as e:
            # If we can't get annotations, count as 0
            annotation_counts.append((member_id, 0))

    # Sort by count (descending) and take top N
    annotation_counts.sort(key=lambda x: x[1], reverse=True)
    top_members = [member_id for member_id, count in annotation_counts[:max_members]]

    print(f"Selected {len(top_members)} family members with most annotations")
    print(f"  Top member: {annotation_counts[0][0]} with {annotation_counts[0][1]} positions")
    if len(annotation_counts) > 1:
        print(f"  Median: {annotation_counts[len(annotation_counts)//2][1]} positions")
        print(f"  Bottom: {annotation_counts[-1][1]} positions")

    return top_members


def positions_to_pymol_notation(positions):
    """
    Convert list of positions to PyMOL residue notation.

    Args:
        positions: List of integer positions

    Returns:
        str: PyMOL notation like "150-155+160-165+170"
    """
    if not positions:
        return ""

    positions = sorted(set(positions))
    ranges = []
    start = positions[0]
    end = positions[0]

    for pos in positions[1:]:
        if pos == end + 1:
            end = pos
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = pos
            end = pos

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return "+".join(ranges)


def aggregate_annotations_across_family(query_id, family_members, cache_dir, confidence_threshold=0.7):
    """
    Transfer ALL functional site annotations from all family members.

    Args:
        query_id: UniProt ID of query protein
        family_members: List of UniProt IDs for family members
        cache_dir: Directory for caching alignments and annotations
        confidence_threshold: Minimum alignment confidence to include (default: 0.7)

    Returns:
        pd.DataFrame with columns: query_position, source_protein, feature_type,
                                   feature_description, alignment_confidence
    """
    print("\n=== Annotation Transfer ===")
    print(f"Query protein: {query_id}")
    print(f"Family members: {len(family_members)}")

    # Get query sequence
    query_seq = get_uniprot_sequence(query_id)
    print(f"Query sequence length: {len(query_seq)}")

    # Set up cache directories
    functional_cache = os.path.join(cache_dir, 'functional_sites')
    alignment_cache = os.path.join(cache_dir, 'alignments')
    os.makedirs(functional_cache, exist_ok=True)
    os.makedirs(alignment_cache, exist_ok=True)

    all_transferred = []

    # First, add annotations from the query protein itself (no alignment needed)
    print(f"\n[0/{len(family_members)}] Processing {query_id} (query itself)...")
    query_annotations = get_functional_sites_detailed(query_id, functional_cache)
    num_query_positions = len(query_annotations)
    print(f"  Functional sites: {num_query_positions} positions annotated")

    if num_query_positions > 0:
        # Add query's own annotations with perfect confidence (1.0)
        for pos_str, annotations in query_annotations.items():
            pos = int(pos_str)
            for annot in annotations:
                all_transferred.append({
                    'query_position': pos,
                    'target_position': pos,
                    'feature_type': annot['feature_type'],
                    'feature_description': annot['description'],
                    'alignment_confidence': 1.0,  # Perfect confidence - it's the same protein
                    'source_protein': query_id
                })
        print(f"  Added: {len([a for a in all_transferred if a['source_protein'] == query_id])} annotations from query itself")

    # Process each family member
    for i, target_id in enumerate(family_members, 1):
        print(f"\n[{i}/{len(family_members)}] Processing {target_id}...")

        # Skip if it's the query (already processed above)
        if target_id == query_id:
            print("  Skipping (already processed as query)")
            continue

        try:
            # Get target sequence
            target_seq = get_uniprot_sequence(target_id)
            print(f"  Target sequence length: {len(target_seq)}")

            # Get target annotations
            target_annotations = get_functional_sites_detailed(target_id, functional_cache)
            num_annotated_positions = len(target_annotations)
            print(f"  Functional sites: {num_annotated_positions} positions annotated")

            if num_annotated_positions == 0:
                print("  Skipping (no functional sites)")
                continue

            # Check for cached alignment
            cached_alignment = load_alignment_cache(query_id, target_id, alignment_cache)

            if cached_alignment:
                print("  Using cached alignment")
                # For cached alignments, we'll recompute since PairwiseAlignment
                # objects are not easily serializable
                alignment = get_pairwise_alignment(query_seq, target_seq)
                aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)
            else:
                # Compute alignment
                print("  Computing alignment...")
                alignment = get_pairwise_alignment(query_seq, target_seq)
                aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)

                # Cache alignment
                save_alignment_cache(query_id, target_id, aligned_seq1, aligned_seq2, alignment_cache)

            # Transfer annotations
            transferred = transfer_all_annotations(query_seq, target_seq, target_annotations)
            print(f"  Transferred: {len(transferred)} annotations")

            # Add source protein ID
            for item in transferred:
                item['source_protein'] = target_id

            # Group transferred annotations by feature for detailed logging
            feature_groups = defaultdict(list)
            for item in transferred:
                key = (item['feature_type'], item['feature_description'])
                feature_groups[key].append(item)

            # Log details for each feature
            print(f"  Feature details:")
            for (feat_type, feat_desc), items in feature_groups.items():
                # Get source positions for this feature
                source_positions = set()
                for pos_str, annotations in target_annotations.items():
                    for annot in annotations:
                        if annot['feature_type'] == feat_type and annot['description'] == feat_desc:
                            source_positions.add(int(pos_str))

                # Get high-confidence transfers (confidence >= 0.7)
                high_conf_items = [item for item in items if item['alignment_confidence'] >= confidence_threshold]

                # Get query positions
                query_positions = sorted(set(item['query_position'] for item in high_conf_items))
                target_positions = sorted(set(item['target_position'] for item in high_conf_items))

                print(f"    [{feat_type}] {feat_desc[:60]}...")
                print(f"      Source: {len(source_positions)} positions in {target_id}")
                print(f"      Transferred: {len(high_conf_items)} high-confidence (≥{confidence_threshold}) positions to query")
                if len(query_positions) > 0:
                    query_range = f"{min(query_positions)}-{max(query_positions)}" if len(query_positions) > 1 else str(query_positions[0])
                    target_range = f"{min(target_positions)}-{max(target_positions)}" if len(target_positions) > 1 else str(target_positions[0])
                    print(f"      Query positions: {query_range} (from target {target_range})")

            all_transferred.extend(transferred)

        except Exception as e:
            print(f"  Error processing {target_id}: {e}")
            continue

    # Convert to DataFrame
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
    Generate output table with one row per unique feature.

    Args:
        annotations_df: DataFrame from annotation transfer
        confidence_threshold: Minimum confidence to include annotation (default: 0.8)

    Returns:
        pd.DataFrame: Output table
    """
    print("\n=== Generating Feature Table ===")
    print(f"Confidence threshold: {confidence_threshold}")

    # Filter by confidence threshold
    total_before = len(annotations_df)
    annotations_df = annotations_df[annotations_df['alignment_confidence'] >= confidence_threshold].copy()
    total_after = len(annotations_df)
    print(f"Filtered annotations: {total_before} → {total_after} (removed {total_before - total_after} low-confidence)")

    if len(annotations_df) == 0:
        print("\nWarning: No annotations remain after confidence filtering!")
        return pd.DataFrame(columns=['feature_name', 'functional_site_category', 'is_major_functional_site',
                                    'residues_pymol', 'num_positions', 'num_supporting_proteins',
                                    'source_uniprot_ids', 'mean_confidence'])

    # Group by feature_type and feature_description
    feature_groups = annotations_df.groupby(['feature_type', 'feature_description'])

    features = []

    for (feat_type, feat_desc), group in feature_groups:
        # Get all positions for this feature
        positions = group['query_position'].unique().tolist()

        # Convert to PyMOL notation
        pymol_notation = positions_to_pymol_notation(positions)

        # Get supporting proteins (source UniProt IDs)
        source_proteins = sorted(group['source_protein'].unique())
        source_proteins_str = ', '.join(source_proteins)

        # Count supporting proteins
        num_proteins = len(source_proteins)

        # Mean confidence
        mean_confidence = group['alignment_confidence'].mean()

        # Determine if this is a "major" functional site based on type
        is_major = feat_type in ['Binding site', 'Domain', 'Active site', 'Site']

        features.append({
            'feature_name': feat_desc,
            'functional_site_category': feat_type,
            'is_major_functional_site': is_major,
            'residues_pymol': pymol_notation,
            'num_positions': len(positions),
            'num_supporting_proteins': num_proteins,
            'source_uniprot_ids': source_proteins_str,
            'mean_confidence': round(mean_confidence, 3)
        })

    # Create DataFrame
    features_df = pd.DataFrame(features)

    # Sort by functional_site_category, then by number of positions (descending)
    features_df = features_df.sort_values(['functional_site_category', 'num_positions'], ascending=[True, False])

    print(f"\nGenerated {len(features_df)} unique features")

    return features_df


def main():
    parser = argparse.ArgumentParser(
        description='Simple functional site annotation (no LLM classification)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('uniprot_id', help='UniProt ID of protein to annotate')
    parser.add_argument('output_file', help='Output TSV file path')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory (default: cache)')
    parser.add_argument('--confidence-threshold', type=float, default=0.8,
                       help='Minimum alignment confidence to include (default: 0.8)')

    args = parser.parse_args()

    # Validate inputs
    uniprot_id = args.uniprot_id
    output_file = args.output_file
    cache_dir = args.cache_dir

    print(f"Simple functional site annotation for {uniprot_id}")
    print(f"Output will be saved to: {output_file}")
    print(f"Cache directory: {cache_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Get protein family members
    print("\n=== Fetching Family Members ===")
    family_members = get_protein_list(uniprot_id)

    # If no family members (likely no PDB structures), try without PDB requirement
    if len(family_members) == 0:
        print("No family members with PDB found. Searching without PDB requirement...")
        from uniprot_api import get_protein_family, extract_family_name
        import requests

        family_text = get_protein_family(uniprot_id)
        family_name = extract_family_name(family_text)
        is_superfamily = 'superfamily' in family_text.lower()

        # Search without PDB requirement, limited to reviewed proteins first
        suffixes = ['superfamily', 'family'] if is_superfamily else ['family', 'superfamily']

        for suffix in suffixes:
            params = {
                'query': f'family:"{family_name} {suffix}" AND reviewed:true',
                'fields': 'accession',
                'size': 50,  # Limit to 50 to avoid overwhelming API
                'format': 'json'
            }

            response = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params)
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    family_members = [r['primaryAccession'] for r in results]
                    print(f"Found {len(family_members)} reviewed family members (max 50)")
                    break

        # If still no results, try unreviewed proteins
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

    # Transfer annotations
    annotations_df = aggregate_annotations_across_family(
        uniprot_id,
        family_members,
        cache_dir,
        confidence_threshold=0.7  # Use lower threshold for transfer, filter later
    )

    # Generate simple feature table
    features_df = generate_simple_feature_table(annotations_df, args.confidence_threshold)

    # Save output
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    features_df.to_csv(output_file, sep='\t', index=False)

    print(f"\n{'='*80}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*80}")

    # Print summary
    print("\nFeature summary by category:")
    type_counts = features_df.groupby('functional_site_category').size().sort_values(ascending=False)
    for feat_type, count in type_counts.items():
        print(f"  {feat_type}: {count} unique features")

    return 0


if __name__ == '__main__':
    sys.exit(main())
