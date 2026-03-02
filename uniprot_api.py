"""
UniProt API utilities for fetching protein information and functional annotations.
Ported from characterize_sites.ipynb with improvements.
"""

import requests
import re
import time
import os
import json


def get_protein_family(uniprot_id):
    """Get family information for a UniProt ID from SIMILARITY comment"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Look for SIMILARITY comment which contains family information
    try:
        comments = data.get('comments', [])
        for comment in comments:
            if comment.get('commentType') == 'SIMILARITY':
                family_text = comment['texts'][0]['value']
                return family_text
        return None
    except (KeyError, IndexError):
        return None


def extract_family_name(family_text):
    """Extract the main family name from the family description"""
    if not family_text:
        return None

    # Look for pattern "Belongs to the FAMILY_NAME family" or "superfamily"
    match = re.search(r'Belongs to the (.+?)(super)?family', family_text)
    if match:
        return match.group(1).strip()
    return None


def search_family_members_with_annotations(family_name, is_superfamily=False):
    """Search for family members with PDB structures"""
    url = "https://rest.uniprot.org/uniprotkb/search"

    # Ensure proper spacing for "family"/"superfamily" suffix
    if not family_name.endswith(' '):
        family_name = family_name + ' '

    # Try with superfamily first if indicated, then fall back to family
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
        protein_list = [protein['primaryAccession'] for protein in response.json().get('results', [])]

        if protein_list:
            return protein_list

    # If both fail, return empty list
    return []


def get_protein_list(uniprot_id):
    """Get list of family members for a UniProt ID"""
    family_text = get_protein_family(uniprot_id)
    if not family_text:
        raise ValueError(
            f"No family information found for {uniprot_id}. "
            f"This protein may not have a SIMILARITY annotation in UniProt. "
            f"Check https://www.uniprot.org/uniprotkb/{uniprot_id}"
        )

    family_name = extract_family_name(family_text)
    if not family_name:
        raise ValueError(f"Could not extract family name from: {family_text}")

    # Check if it's a superfamily
    is_superfamily = 'superfamily' in family_text.lower()

    print(f'Family name: {family_name}')
    if is_superfamily:
        print(f'Type: superfamily')

    protein_list = search_family_members_with_annotations(family_name, is_superfamily)
    print(f'Found {len(protein_list)} family members with PDB structures')

    return protein_list


def get_uniprot_sequence(uniprot_id):
    """Get protein sequence from UniProt"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
    params = {'fields': 'sequence'}

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    sequence = data.get('sequence', {}).get('value', '')
    return sequence


def get_protein_function(uniprot_id):
    """Get protein function description from UniProt"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
    params = {'fields': 'cc_function'}

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    try:
        function_text = data['comments'][0]['texts'][0]['value']
        return function_text
    except (KeyError, IndexError):
        return None


def get_functional_sites_detailed(uniprot_id, cache_dir=None):
    """
    Fetch detailed functional site information for a UniProt ID.

    Args:
        uniprot_id: UniProt accession ID
        cache_dir: Optional directory for caching results

    Returns:
        Dictionary mapping amino acid positions (as strings) to list of annotation dicts
    """
    # Check cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{uniprot_id}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

    functional_sites = {}

    try:
        # UniProt REST API endpoint
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Extract features
        features = data.get('features', [])

        # Filter for functional site types
        # Expanded to include structural/functional domains
        functional_types = {
            'Binding site',
            'Active site',
            'Site',
            'Metal binding',
            'Calcium binding',
            'DNA binding',
            'Nucleotide binding',
            'Modified residue',
            'Cross-link',
            'Transmembrane',      # Often contains functional info (e.g., voltage sensor)
            'Motif',              # Functional motifs (e.g., selectivity filter)
            'Region',             # Functional regions
            'Domain',             # Functional domains
            'Topological domain', # Can indicate functional locations
        }

        for feature in features:
            if feature.get('type') not in functional_types:
                continue

            location = feature.get('location', {})
            feature_type = feature.get('type')
            description = feature.get('description', '')

            # Add ligand info if available
            if 'ligand' in feature:
                ligand_name = feature['ligand'].get('name', '')
                if ligand_name:
                    description = f"{description} Ligand: {ligand_name}".strip()

            positions_to_add = []

            # Handle single position
            if 'position' in location:
                pos = location['position'].get('value')
                if pos:
                    positions_to_add.append(int(pos))

            # Handle range (avoid excessively large ranges)
            elif 'start' in location and 'end' in location:
                start_pos = location['start'].get('value')
                end_pos = location['end'].get('value')
                if start_pos and end_pos and (end_pos - start_pos < 250):
                    positions_to_add.extend(range(int(start_pos), int(end_pos) + 1))

            # Add functional site info for each position
            for pos in positions_to_add:
                site = {
                    'position': pos,
                    'feature_type': feature_type,
                    'description': description,
                }

                pos_str = str(pos)
                if pos_str not in functional_sites:
                    functional_sites[pos_str] = []
                functional_sites[pos_str].append(site)

        # Small delay to be respectful to the API
        time.sleep(0.1)

    except Exception as e:
        print(f"Warning: Error fetching data for {uniprot_id}: {e}")

    # Cache results
    if cache_dir:
        with open(cache_file, 'w') as f:
            json.dump(functional_sites, f, indent=2)

    return functional_sites


def save_alignment_cache(query_id, target_id, aligned_seq1, aligned_seq2, cache_dir):
    """Save alignment to cache"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"alignment_{query_id}_{target_id}.json")

    alignment_data = {
        'seqA': aligned_seq1,
        'seqB': aligned_seq2,
    }

    with open(cache_file, 'w') as f:
        json.dump(alignment_data, f)


def load_alignment_cache(query_id, target_id, cache_dir):
    """Load alignment from cache"""
    cache_file = os.path.join(cache_dir, f"alignment_{query_id}_{target_id}.json")

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)

    return None
