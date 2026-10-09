"""
UniProt functional-site annotation, and transferring it between homologues.

A protein's own UniProt record often annotates only a handful of residues, so a
neighborhood can land on something well characterised and still look bare. This module
widens the evidence: it finds the query's protein family, pulls every member's annotated
sites, aligns each member to the query, and carries the annotations across, scoring each
transfer by how trustworthy the local alignment is.

Split out of visualize_and_interpret.py, which had grown to cover sequence alignment, a
REST client, annotation transfer, structure tables, PyMOL sessions and a CLI in one file.

Everything here talks to the network, and everything here caches: sequences and
functional sites per accession, alignments per (query, target) pair. A second run over
the same family costs nothing.
"""

import json
import os
import time
from collections import defaultdict

import pandas as pd
from Bio.Align import PairwiseAligner

try:
    import requests
except ImportError:
    # Only this module needs network access. Visualization, the neighborhood residue
    # tables and the feature cross-reference all work offline, so a missing `requests`
    # must not stop the rest of the tool being imported.
    class _RequestsUnavailable:
        def __getattr__(self, name):
            raise ImportError(
                'The annotation steps need the `requests` package (pip install requests). '
                'Visualization and neighborhood feature tables do not.'
            )

    requests = _RequestsUnavailable()


# Alignments are full-length and pairwise, so cost grows with family size. UniProt
# returns up to 500 members for a large family, which is hundreds of alignments per
# protein; keeping the best-annotated members holds the runtime down without losing
# much, since the trimmed members are the ones with little to transfer.
DEFAULT_MAX_FAMILY_MEMBERS = 50


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
# UniProt API
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
# Annotation pipeline
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



def resolve_family_members(uniprot_id, cache_dir=None,
                           max_members=DEFAULT_MAX_FAMILY_MEMBERS):
    """
    The family members whose annotations will be transferred to `uniprot_id`.

    Tries, in order: members with a PDB structure; reviewed members; any members. The
    widening steps matter because the PDB requirement is what usually comes back empty,
    and a family with no solved structures still has transferable annotations.

    Returns [] when the protein has no family record at all, which is normal -- the
    caller falls back to the query's own annotations.

    This consolidates two copies of the same cascade that had drifted apart in
    cmd_annotate and _annotate_and_save.
    """
    try:
        family_text = get_protein_family(uniprot_id)
    except Exception as e:
        print(f'  Warning: could not look up family for {uniprot_id}: {e}')
        return []
    if not family_text:
        print(f'  No family information for {uniprot_id}; using its own annotations only.')
        return []

    family_name = extract_family_name(family_text)
    if not family_name:
        print(f'  Could not parse a family name from: {family_text}')
        return []

    is_superfamily = 'superfamily' in family_text.lower()
    suffixes = ['superfamily', 'family'] if is_superfamily else ['family', 'superfamily']
    print(f'Family name: {family_name}' + ('  (superfamily)' if is_superfamily else ''))

    members = search_family_members_with_annotations(family_name, is_superfamily)
    if not members:
        for extra in ('AND reviewed:true', ''):
            for suffix in suffixes:
                query = f'family:"{family_name} {suffix}" {extra}'.strip()
                params = {'query': query, 'fields': 'accession', 'size': 500,
                          'format': 'json'}
                r = requests.get('https://rest.uniprot.org/uniprotkb/search', params=params)
                if r.status_code == 200 and r.json().get('results'):
                    members = [x['primaryAccession'] for x in r.json()['results']]
                    break
            if members:
                break

    print(f'Found {len(members)} family members')
    if max_members and len(members) > max_members and cache_dir:
        members = filter_family_by_annotation_count(members, cache_dir, max_members)
    return members
