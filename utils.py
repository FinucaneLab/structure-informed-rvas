import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import gzip
import sklearn.metrics
import hdf5plugin
import os
import re
import json
import warnings
from logger_config import get_logger

logger = get_logger(__name__)

def get_pairwise_distances(pdb_file, *args):
    # optional arguments i and j:
    # i: start from aminoacid i (inclusive)
    # j: end with aminoacid j (inclusive)
    # if only one argument i is provided, it computes starting from i till the end of the chain
    parser = PDBParser(QUIET=True)
    if pdb_file.endswith('.gz'):
        with gzip.open(pdb_file, 'rt') as handle:
            structure = parser.get_structure("protein", handle)
    else:
        with open(pdb_file, 'r') as handle:
            structure = parser.get_structure("protein", handle)

    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atoms.append(residue['CA'].get_coord())
   
    if len(args) > 1:
        i = args[0]
        j = args[1]
        ca_atoms = np.array(ca_atoms[i-1:j])
    elif len(args) > 0:
        i = args[0]
        ca_atoms = np.array(ca_atoms[i-1:])
    else:
        ca_atoms = np.array(ca_atoms)
    pairwise_distances = sklearn.metrics.pairwise_distances(ca_atoms)
    return pairwise_distances


def get_distance_matrix_structure(pdb_file_pos_guide, pdb_dir, uniprot_id):
    logger.debug(f"Computing distance matrix for {uniprot_id}")
    info = pd.read_csv(pdb_file_pos_guide, sep="\t")
    pdb_files = info.loc[info.pdb_filename.str.contains(uniprot_id),'pdb_filename']
    ## Version 2: central on top of all
    if len(pdb_files)==0:
        logger.warning(f"Protein {uniprot_id} not found.")
        return None
    elif len(pdb_files)==1:
        # One pdb file in structure
        pathfile = os.path.join(pdb_dir, pdb_files.iloc[0])
        distance_matrix = get_pairwise_distances(pathfile)
    else:
        # Multiple pdb files in structure
        info = info.iloc[pdb_files.index].copy().reset_index()
        info['startAA'] = info.apply(lambda x: int(re.findall(r"\d+", x.pos_covered)[0]), axis=1)
        info['endAA'] = info.apply(lambda x: int(re.findall(r"\d+", x.pos_covered)[1]), axis=1)
        nAA = info.endAA.max()
        info['startAA_next'] = info.startAA.shift(periods=-1, fill_value=nAA)
        info['j'] = np.floor((info.endAA+info.startAA_next)/2).astype(int)
        info['i'] = info.j.shift(periods=1, fill_value=0)+1
    
        distance_matrix = np.full(shape=(nAA,nAA), fill_value=np.inf)
        cum_nAA=0
        # Calculate distance matrix for the entire pdb range
        for pdb in range(0,info.shape[0]):
            pathfile = os.path.join(pdb_dir, info.pdb_filename.values[pdb])
            i = info.startAA[pdb]
            j = info.endAA[pdb]
            distance_matrix[i-1:j,i-1:j] = get_pairwise_distances(pathfile)
        # Substitute overlapping part using "most central" rule
        for pdb in range(0,info.shape[0]):
            pathfile = os.path.join(pdb_dir, info.pdb_filename.values[pdb])
            i = info.i[pdb]
            j = info.j[pdb]
            i_in_pdb = i-(info.startAA[pdb]-1)
            j_in_pdb = j-(info.startAA[pdb]-1)
            nAA_in_pdb = j_in_pdb - i_in_pdb + 1
            distance_matrix[cum_nAA:cum_nAA+nAA_in_pdb, cum_nAA:cum_nAA+nAA_in_pdb] = get_pairwise_distances(pathfile, i_in_pdb, j_in_pdb)
            cum_nAA = cum_nAA + nAA_in_pdb 

    return distance_matrix

def get_paes(pae_file, *args):
    # optional arguments i and j:
    # i: start from aminoacid i (inclusive)
    # j: end with aminoacid j (inclusive)
    # if only one argument i is provided, it computes starting from i till the end of the chain
    if pae_file.endswith('.gz'):
        with gzip.open(pae_file, 'rt') as f: 
            data = json.load(f)
    else:
        with open(pae_file, 'r') as f:
            data = json.load(f)
    pae_matrix = np.array(data[0]['predicted_aligned_error'])
    #(pae_matrix < 15) * 1
   
    if len(args) > 1:
        i = args[0]
        j = args[1]
        pae_matrix = pae_matrix[i-1:j, i-1:j]
    elif len(args) > 0:
        i = args[0]
        pae_matrix = pae_matrix[i-1:, i-1:]

    return pae_matrix
    
def get_pae_matrix_structure(pae_file_pos_guide, pae_dir, uniprot_id):
    logger.debug(f"Computing pae matrix for {uniprot_id}")
    info = pd.read_csv(pae_file_pos_guide, sep="\t")
    #pae_files = info.loc[info.pae_filename.str.contains(uniprot_id),'pae_filename']
    # need to deal with null entries 
    pae_files = info.loc[info.pae_filename.notnull() & info.pae_filename.str.contains(uniprot_id), 'pae_filename']
    ## Version 2: central on top of all
    if len(pae_files)==0:
        pdb_files = info.loc[info.pdb_filename.str.contains(uniprot_id), 'pdb_filename']
        if len(pdb_files)==0:
            logger.warning(f"Protein {uniprot_id} not found.")
            return None
        else:
            known_no_pae = info.loc[info.uniprot_id == uniprot_id, 'pae_filename'].isna().any()
            if known_no_pae:
                logger.debug(f"PAE file not available for {uniprot_id} (known). Using distance-only adjacency.")
            else:
                logger.warning(f"PAE file not found for Protein {uniprot_id}. No PAE filtering will be used.")
            return None
    elif len(pae_files)==1:
        # One pae file for structure
        pathfile = os.path.join(pae_dir, pae_files.iloc[0])
        if not os.path.exists(pathfile):
            logger.warning(f"File {pathfile} does not exist. No PAE filtering will be used.")
            return None
        pae_matrix = get_paes(pathfile)
    else:
        # Multiple pae files for structure
        info = info.iloc[pae_files.index].copy().reset_index()
        info['startAA'] = info.apply(lambda x: int(re.findall(r"\d+", x.pos_covered)[0]), axis=1)
        info['endAA'] = info.apply(lambda x: int(re.findall(r"\d+", x.pos_covered)[1]), axis=1)
        nAA = info.endAA.max()
        info['startAA_next'] = info.startAA.shift(periods=-1, fill_value=nAA)
        info['j'] = np.floor((info.endAA+info.startAA_next)/2).astype(int)
        info['i'] = info.j.shift(periods=1, fill_value=0)+1
    
        pae_matrix = np.full(shape=(nAA,nAA), fill_value=np.inf)
        cum_nAA=0
        # Calculate pae matrix for the entire pae range
        for pae in range(0,info.shape[0]):
            pathfile = os.path.join(pae_dir, info.pae_filename.values[pae])
            i = info.startAA[pae]
            j = info.endAA[pae]
            pae_matrix[i-1:j,i-1:j] = get_paes(pathfile)
        # Substitute overlapping part using "most central" rule
        for pae in range(0,info.shape[0]):
            pathfile = os.path.join(pae_dir, info.pae_filename.values[pae])
            i = info.i[pae]
            j = info.j[pae]
            i_in_pae = i-(info.startAA[pae]-1)
            j_in_pae = j-(info.startAA[pae]-1)
            nAA_in_pae = j_in_pae - i_in_pae + 1
            pae_matrix[cum_nAA:cum_nAA+nAA_in_pae, cum_nAA:cum_nAA+nAA_in_pae] = get_paes(pathfile, i_in_pae, j_in_pae)
            cum_nAA = cum_nAA + nAA_in_pae

    # force PAE matrix to be symmetrical
    ## take the minimum where it is not symmetrical
    if pae_matrix is not None:
        pae_matrix = np.minimum(pae_matrix, pae_matrix.T)
    return pae_matrix

def get_adjacency_matrix(pdb_pae_file_pos_guide, pdb_dir, pae_dir, uniprot_id, radius, pae_cutoff):
    logger.debug(f"Computing adjacency matrix for {uniprot_id} with radius={radius}, pae_cutoff={pae_cutoff}")
    distance_matrix = get_distance_matrix_structure(pdb_pae_file_pos_guide, pdb_dir, uniprot_id)
    if distance_matrix is None:
        logger.warning(f"No distance matrix found for {uniprot_id}, returning None")
        return None
    dist_thresh = (distance_matrix < radius) * 1
    if pae_cutoff == 0:
        pae_thresh = np.ones_like(dist_thresh)
    else:
        pae_matrix = get_pae_matrix_structure(pdb_pae_file_pos_guide, pae_dir, uniprot_id)
        if pae_matrix is None:
            pae_thresh = np.ones_like(dist_thresh)
        else:
            pae_thresh = (pae_matrix < pae_cutoff) * 1
    adj_mat = dist_thresh & pae_thresh
    logger.debug(f"Adjacency matrix computed: {np.sum(adj_mat)} edges from {adj_mat.shape[0]} positions")
    return adj_mat
def valid_for_fisher(contingency_table):
    valid_columns = np.all(np.sum(contingency_table, axis=0) > 0)
    valid_rows = np.all(np.sum(contingency_table, axis=1) > 0)
    all_non_neg = np.all(contingency_table >= 0)
    if valid_columns and valid_rows and all_non_neg:
        return True
    else:
        return False

def write_dataset(fid, name, data, clevel=5):
    if name in fid:
        del fid[name]
    fid.create_dataset(
        name,
        data = data,
        compression = hdf5plugin.Zstd(clevel=clevel)
    )


MU_TEST_GROUPS = ('poisson', 'binomial')


def resolve_pval_node(fid, test='binomial'):
    """
    Resolve where one file keeps its per-protein datasets, and say which kind it is.

    The standard 3DNT writes `{uniprot_id}`, `_nbhd`, `_radius` and `_original` at the
    root. The mutation-rate 3DNT runs two tests against the same observed data and writes
    one such set per test, under the 'poisson' and 'binomial' groups. Readers that index
    the root therefore raise KeyError on a mutation-rate file, so every consumer should
    come through here rather than reaching into `fid` directly.

    Returns (node, mu_mode). `test` selects the group and is ignored for standard files.
    """
    if fid.attrs.get('test_type') != 'mutation_rate':
        return fid, False
    if test not in MU_TEST_GROUPS:
        raise ValueError(f"test must be one of {MU_TEST_GROUPS}, got {test!r}")
    if test not in fid:
        raise KeyError(
            f"'{test}' group missing from a mutation-rate p-value file "
            f"(groups present: {sorted(k for k in fid.keys())})"
        )
    return fid[test], True


def read_p_values(fid, uniprot_id, mu_mode=False):
    """
    Reads the p values for one uniprot_id from an HDF5 results file,
    with the exception of the null values.

    When mu_mode is True the second column of the _nbhd dataset holds the expected
    de novo count under that test's null rather than a control allele count, and the
    reported effect size is a regularized observed/expected ratio.
    """
    pvalue_data = fid[uniprot_id][:]
    case_control = fid[f'{uniprot_id}_nbhd'][:]

    nbhd_case = case_control[:, 0]
    second = case_control[:, 1]

    radius_key = f'{uniprot_id}_radius'
    radius_vals = fid[radius_key][:, 0].astype(float) if radius_key in fid else np.full(len(nbhd_case), np.nan)

    base = {'uniprot_id': uniprot_id,
            'aa_pos': np.arange(1, pvalue_data.shape[0]+1),
            'p_value': pvalue_data[:, 0]}

    if mu_mode:
        base.update({'obs_exp': (nbhd_case + 1) / (second + 1),
                     'nbhd_case': nbhd_case,
                     'nbhd_expected': second,
                     'radius': radius_vals})
        return pd.DataFrame(base)

    # Calculate ratio on-the-fly
    nbhd_control = second
    n_case_total = nbhd_case.sum()
    n_control_total = nbhd_control.sum()
    ratio = (nbhd_case + 2) / (nbhd_control + 2 * n_control_total / n_case_total)

    base.update({'ratio': ratio,
                 'nbhd_case': nbhd_case,
                 'nbhd_control': nbhd_control,
                 'radius': radius_vals})
    return pd.DataFrame(base)

def read_original_mutation_data(fid, uniprot_id, mu_mode=False):
    """
    Reads the original per-residue mutation counts for one uniprot_id from an HDF5 results file.

    In mutation-rate mode the second column of `_original` is the residue's mutational
    opportunity, not a control allele count, and the second column of `_nbhd` is the
    expected de novo count under that test's null. The returned frame then carries `mu`
    and `obs_exp` instead of `ac_control` and `ratio`.

    The columns are deliberately NOT aliased onto the case/control names. Mutation rates
    are not allele counts and their scale depends on the rate model: per-residue gnomAD mu
    sits around 2e-08, while a Roulette-derived table has a median near 0.3. So a caller
    testing `ac_control == 0` silently finds no case-only residues under either model, and
    one testing `ac_control >= 1` silently finds no controls under gnomAD but a scattering
    of them under Roulette. Both produce a plausible-looking answer from a column that
    means something else entirely, and the answer changes with the rate file.
    """
    pvalue_data = fid[uniprot_id][:]
    original_data = fid[f'{uniprot_id}_original'][:]
    case_control = fid[f'{uniprot_id}_nbhd'][:]

    nbhd_case = case_control[:, 0]
    second = case_control[:, 1]

    base = {'uniprot_id': uniprot_id,
            'aa_pos': np.arange(1, pvalue_data.shape[0]+1),
            'p_value': pvalue_data[:, 0]}

    if mu_mode:
        # Same regularized effect size read_p_values reports, so the colouring and the
        # results table agree.
        base.update({'obs_exp': (nbhd_case + 1) / (second + 1),
                     'ac_case': original_data[:, 0],
                     'mu': original_data[:, 1]})
        return pd.DataFrame(base)

    # Calculate ratio on-the-fly using neighborhood data
    nbhd_control = second
    n_case_total = nbhd_case.sum()
    n_control_total = nbhd_control.sum()
    ratio = (nbhd_case + 2) / (nbhd_control + 2 * n_control_total / n_case_total)

    base.update({'ratio': ratio,
                 'ac_case': original_data[:, 0],
                 'ac_control': original_data[:, 1]})
    return pd.DataFrame(base)

def get_nbhd_info(df_rvas, uniprot_id, aa_pos, reference_dir, radius, pae_cutoff):
    """
    For a given UniProt ID (str) and neighborhood center (int), returns the following: 
    1. a list of residue positions in the neighborhood (list of integers)
    2. a dataframe case variant residue positions in the neighborhood
        - variant genetic coordinates; amino acid position, reference, and alternate; and allele count
    3. a dataframe control variant residue positions in the neighborhood
        - variant genetic coordinates; amino acid position, reference, and alternate; and allele count
    """
    pdb_pae_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'

    # get all neighborhood residues
    adj_mat = get_adjacency_matrix(pdb_pae_file_pos_guide, pdb_dir, pae_dir, uniprot_id, radius, pae_cutoff)
    nbhd = np.where(adj_mat[int(aa_pos)-1,:]==1)[0]+1

    # restrict to our uniprot id
    df_rvas_u = df_rvas[df_rvas.uniprot_id==uniprot_id]
    # get all variants on neighborhood residues
    df_rvas_nbhd = df_rvas_u[df_rvas_u.aa_pos.isin(nbhd)]
    # all case variants on neighborhood residues
    df_rvas_case = df_rvas_nbhd[df_rvas_nbhd.ac_case>0]
    # all control variants on neighborhood residues
    df_rvas_cntrl = df_rvas_nbhd[df_rvas_nbhd.ac_control>0]

    cases = df_rvas_case[['Variant ID', 'aa_pos','ac_case', 'aa_ref', 'aa_alt']].reset_index(drop=True)
    cntrls = df_rvas_cntrl[['Variant ID', 'aa_pos', 'ac_control', 'aa_ref', 'aa_alt']].reset_index(drop=True)

    # alternative option - function takes in list of aa_pos (aa_list) and calcs nbhd for each of them
    # could also take in no list of pos and just return neighborhoods of whole protein
    # aa_array = np.array(aa_list) - 1  # convert to 0-based indices
    # # Get a boolean mask for rows in adjacency_matrix
    # submatrix = adjacency_matrix[aa_array, :]  # shape: (len(aa_list), n_res)
    # # Convert each row's "1"s to residue numbers
    # residues = [list(np.where(row == 1)[0] + 1) for row in submatrix]
    # return pd.DataFrame({
    #     'uniprot_id': uniprot_id,
    #     'aa_pos': aa_list,
    #     'residues': residues
    
    return nbhd, cases, cntrls


# ---------------------------------------------------------------------------
# residue masking for conditional analysis
#
# INDEXING CONTRACT. Every per-residue array in this codebase is positional: row i
# holds residue aa_pos = i + 1. That convention is relied on by get_ac_per_residue,
# mu_vector_for_protein, sum_per_residue, the adjacency matrices, and -- critically --
# by read_p_values, which does not store aa_pos at all but rebuilds it as
# arange(1, n_rows + 1), and by empirical_fdr._load_all_pvalues, which filters
# positions with `x + 1 in aa_pos_keep for x in range(len(df))`.
#
# So masking a residue must NEVER drop a row. A masked residue keeps its slot and is
# zeroed out; the p-value table written to HDF5 always has exactly n_res rows in
# residue order. check_keep_mask() is the guard that a mask built against one
# structure is not silently applied to an array of a different length.
#
# Masks use "keep" polarity: keep[i] is True when residue i + 1 still participates.
# ---------------------------------------------------------------------------

def check_keep_mask(keep, n_res, uniprot_id=''):
    """Validate a residue keep-mask against n_res. Returns a bool array, or None."""
    if keep is None:
        return None
    keep = np.asarray(keep).ravel().astype(bool)
    if keep.shape[0] != n_res:
        raise ValueError(
            f'{uniprot_id}: residue keep-mask has length {keep.shape[0]} but the '
            f'structure has {n_res} residues. The mask is positional -- entry i is '
            f'residue i+1 -- so a length mismatch would silently shift every residue.'
        )
    return keep


def positions_to_pymol_notation(positions):
    """
    Encode a collection of 1-based integer positions as a compact PyMOL selection
    string using run-length range syntax, e.g. [1,2,3,5,6] -> "1-3+5-6".
    """
    if positions is None:
        return ""
    positions = sorted(set(int(p) for p in positions))
    if not positions:
        return ""
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


def parse_pymol_positions(pymol_str):
    """Parse PyMOL residue notation like '10+20-25+30' back to a set of integers."""
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


def residues_in_nbhd_idx(adjacency_matrix, center_aa_pos):
    """
    Zero-based row indices of the residues in the neighborhood centered on the
    one-based residue `center_aa_pos`. Add 1 to get aa_pos values back.
    """
    n_res = adjacency_matrix.shape[0]
    i = int(center_aa_pos) - 1
    if not 0 <= i < n_res:
        raise ValueError(
            f'neighborhood center aa_pos {center_aa_pos} is outside the structure '
            f'(1..{n_res})'
        )
    return np.flatnonzero(adjacency_matrix[i] == 1)


def conditioning_keep_mask(pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id,
                           centers, pae_cutoff, residues=None, n_res=None):
    """
    Build the keep-mask that conditions out one or more called regions.

    Two forms, because the two callers have different things in hand:

      `centers`  an iterable of (aa_pos, radius) pairs. Every residue within `radius` of
                 any center is dropped. This is what --remove-nbhd passes, and what the
                 old single-neighborhood conditioning used. Adjacency matrices are built
                 once per distinct radius.

      `residues` an explicit iterable of 1-based residue positions to drop. Conditional
                 analysis passes this: a component is the union of many overlapping
                 neighborhoods, so its residue set is already known and re-deriving it
                 from centers and radii would just rebuild the same union. Needs `n_res`
                 unless `centers` is also given.

    Both may be supplied, in which case the masks are unioned.

    Returns (keep, n_res, removed_by_center). removed_by_center maps each center's aa_pos
    to the number of residues it took out, with the key 'residues' used for the explicit
    set; counts are of residues not already masked by an earlier entry, so they sum to
    the total removed.
    """
    centers = list(centers or [])
    if not centers and residues is None:
        raise ValueError('conditioning_keep_mask needs centers or residues')
    if not centers and n_res is None:
        raise ValueError('conditioning_keep_mask needs n_res when only residues are given')

    adj_cache = {}
    keep = None
    removed_by_center = {}
    for aa_pos, radius in centers:
        if radius not in adj_cache:
            adj = get_adjacency_matrix(pdb_file_pos_guide, pdb_dir, pae_dir,
                                       uniprot_id, radius, pae_cutoff)
            if adj is None:
                raise FileNotFoundError(f'No structure available for {uniprot_id}')
            adj_cache[radius] = adj
        adj = adj_cache[radius]
        if keep is None:
            keep = np.ones(adj.shape[0], dtype=bool)
        elif adj.shape[0] != keep.shape[0]:
            raise ValueError(
                f'{uniprot_id}: adjacency matrices disagree on residue count '
                f'({adj.shape[0]} vs {keep.shape[0]})'
            )
        idx = residues_in_nbhd_idx(adj, aa_pos)
        removed_by_center[int(aa_pos)] = int(keep[idx].sum())
        keep[idx] = False

    if residues is not None:
        if keep is None:
            keep = np.ones(int(n_res), dtype=bool)
        idx = np.asarray(sorted(set(int(r) for r in residues)), dtype=int) - 1
        if len(idx) and (idx.min() < 0 or idx.max() >= keep.shape[0]):
            raise ValueError(
                f'{uniprot_id}: conditioning residues fall outside the structure '
                f'(1..{keep.shape[0]})'
            )
        removed_by_center['residues'] = int(keep[idx].sum())
        keep[idx] = False

    return keep, keep.shape[0], removed_by_center


_STRUCT_LEN_CACHE = {}


def structure_length(pdb_file_pos_guide, uniprot_id):
    """
    Number of residues in a protein's structure, read from the position guide.

    This is the same n_res the adjacency matrix has -- get_distance_matrix_structure
    sizes itself by the maximum end of pos_covered -- so a mask built against it will
    pass check_keep_mask. Reading the guide is cached because conditioning calls this
    once per gene per round.
    """
    guide = _STRUCT_LEN_CACHE.get(pdb_file_pos_guide)
    if guide is None:
        df = pd.read_csv(pdb_file_pos_guide, sep='\t')
        ends = df.pos_covered.str.extract(r'(\d+)\s*\]')[0].astype(float)
        guide = df.assign(end=ends).groupby('uniprot_id')['end'].max().to_dict()
        _STRUCT_LEN_CACHE[pdb_file_pos_guide] = guide
    n_res = guide.get(uniprot_id)
    if n_res is None or not np.isfinite(n_res):
        raise ValueError(f'{uniprot_id}: no structure length in {pdb_file_pos_guide}')
    return int(n_res)


def build_conditioning_mask(pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, entry,
                            pae_cutoff):
    """
    Keep-mask for one gene's conditioning entry, whichever form it takes.

    `entry` is either a list of (aa_pos, radius) pairs -- what --remove-nbhd supplies --
    or an iterable of 1-based residue positions, which is what conditional analysis
    supplies now that it masks whole components. Dispatching here keeps the two scan
    backends from each having to know the difference.
    """
    entry = list(entry)
    if not entry:
        return None, None, {}
    is_centers = isinstance(entry[0], (tuple, list))
    if is_centers:
        return conditioning_keep_mask(pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id,
                                      entry, pae_cutoff)
    n_res = structure_length(pdb_file_pos_guide, uniprot_id)
    return conditioning_keep_mask(pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id,
                                  None, pae_cutoff, residues=entry, n_res=n_res)


def neighborhood_retention(adjacency_matrix, keep):
    """
    Fraction of each neighborhood's residues that survive `keep`, as an (n_res,) float
    array aligned to the same positional convention. Neighborhoods of size zero -- a
    residue with no adjacency at all, which the PAE filter can produce -- get 0.0.
    """
    sizes = adjacency_matrix.sum(axis=1).astype(float).ravel()
    retained = (adjacency_matrix @ keep.astype(float)).ravel()
    out = np.zeros_like(sizes)
    np.divide(retained, sizes, out=out, where=sizes > 0)
    return out


def best_radius_for_hit(df_fdr, uniprot_id, aa_pos, default=15.0,
                        radius_cols=('radius',)):
    """
    Look up the radius a scan actually chose for one (protein, position).

    Under --neighborhood-radius multiple-small/multiple-big each position picks its own
    best radius, which the results record; a single-radius run records that radius. The
    first column in `radius_cols` that is present and non-null wins, else `default`.
    """
    if df_fdr is None:
        return float(default)
    rows = df_fdr[(df_fdr['uniprot_id'] == uniprot_id) & (df_fdr['aa_pos'] == aa_pos)]
    if len(rows) == 0:
        return float(default)
    row = rows.iloc[0]
    for col in radius_cols:
        if col in df_fdr.columns and not pd.isna(row[col]):
            return float(row[col])
    return float(default)
