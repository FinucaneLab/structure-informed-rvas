import gzip
import json
import os
import re

import hdf5plugin
import numpy as np
import pandas as pd
import sklearn.metrics
from Bio.PDB import PDBParser
from scipy.stats import chi2 as chi2_dist

from logger_config import get_logger

logger = get_logger(__name__)

# 1-df chi2 screen (p~0.1): exact tests run only above it, lenient so no p<0.05 is missed.
DEFAULT_SCREEN_CHI2 = 2.706

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


def get_chi2_pvals_case_control(n_case_nbhd_mat, n_control_nbhd_mat, n_case, n_ctrl):
    """Chi-square p-value lookup table over every (case_ac, ctrl_ac) pair in a neighborhood.

    Vectorized 1-df chi-square approximation to the Fisher's exact p-value for every
    possible 2x2 table with margins (n_case, n_ctrl); callers should only run the exact
    Fisher test afterwards for the (a, b) pairs this flags as borderline - avoids calling
    scipy.stats.fisher_exact per cell/permutation, which dominates runtime at scale.
    Used by annotation_test_permutation.

    Returns:
        pvals: (max_a+1, max_b+1) chi-square p-values.
        potential_significant: same-shape mask, True where the exact test should be used.
    """
    max_n_case_nbhd = int(np.max(n_case_nbhd_mat))
    max_n_control_nbhd = int(np.max(n_control_nbhd_mat))

    case_nbhd_range = np.arange(max_n_case_nbhd + 1)
    ctrl_nbhd_range = np.arange(max_n_control_nbhd + 1)
    n_case_grid, n_ctrl_grid = np.meshgrid(
        case_nbhd_range, ctrl_nbhd_range, indexing="ij"
    )

    a, b = n_case_grid, n_ctrl_grid
    c, d = n_case - a, n_ctrl - b

    col1_sum, col2_sum = a + c, b + d
    row1_sum, row2_sum = a + b, c + d
    valid_mask = (
        (col1_sum > 0)
        & (col2_sum > 0)
        & (row1_sum > 0)
        & (row2_sum > 0)
        & (a >= 0)
        & (b >= 0)
        & (c >= 0)
        & (d >= 0)
    )

    n_total = n_case + n_ctrl
    expected_a = row1_sum * col1_sum / n_total
    expected_b = row1_sum * col2_sum / n_total
    expected_c = row2_sum * col1_sum / n_total
    expected_d = row2_sum * col2_sum / n_total

    valid_array = valid_mask.astype(float)
    chi2_stat = valid_array * (
        ((a - expected_a) ** 2 / np.maximum(1e-10, expected_a))
        + ((b - expected_b) ** 2 / np.maximum(1e-10, expected_b))
        + ((c - expected_c) ** 2 / np.maximum(1e-10, expected_c))
        + ((d - expected_d) ** 2 / np.maximum(1e-10, expected_d))
    )

    pvals = np.where(valid_mask, chi2_dist.sf(chi2_stat, df=1), 1.0)
    potential_significant = (chi2_stat > DEFAULT_SCREEN_CHI2) & valid_mask
    return pvals, potential_significant


def valid_for_fisher(contingency_table):
    valid_columns = np.all(np.sum(contingency_table, axis=0) > 0)
    valid_rows = np.all(np.sum(contingency_table, axis=1) > 0)
    all_non_neg = np.all(contingency_table >= 0)
    return bool(valid_columns and valid_rows and all_non_neg)


def write_dataset(fid, name, data, clevel=5):
    if name in fid:
        del fid[name]
    fid.create_dataset(
        name,
        data = data,
        compression = hdf5plugin.Zstd(clevel=clevel)
    )


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


def read_original_mutation_data(fid, uniprot_id):
    """
    Reads the original per-residue mutation counts for one uniprot_id from an HDF5 results file.
    """
    pvalue_data = fid[uniprot_id][:]
    original_data = fid[f'{uniprot_id}_original'][:]
    case_control = fid[f'{uniprot_id}_nbhd'][:]
    
    # Calculate ratio on-the-fly using neighborhood data
    nbhd_case = case_control[:, 0]
    nbhd_control = case_control[:, 1]
    n_case_total = nbhd_case.sum()
    n_control_total = nbhd_control.sum()
    ratio = (nbhd_case + 2) / (nbhd_control + 2 * n_control_total / n_case_total)
    
    df = pd.DataFrame({'uniprot_id': uniprot_id,
                       'aa_pos': np.arange(1, pvalue_data.shape[0]+1),
                       'p_value': pvalue_data[:, 0],
                       'ratio': ratio,
                       'ac_case': original_data[:, 0],
                       'ac_control': original_data[:, 1]})
    return df


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


def get_ac_per_residue(df, colname, n_res):
    case_ac_per_residue = np.zeros((n_res, 1))
    ac_by_pos = df.groupby('aa_pos')[colname].sum().reset_index()
    case_ac_per_residue[ac_by_pos.aa_pos - 1, 0] = ac_by_pos[colname]
    return case_ac_per_residue


def get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim, seed=None):
    # seed is passed straight to default_rng: a previous version called np.random.seed(),
    # which seeds the legacy global RNG and has no effect on default_rng(), so the seed
    # was silently ignored and runs were not reproducible.
    n_alleles = int(  case_ac_per_residue.sum()  )
    gen = np.random.default_rng(seed)
    null_ac_per_residue = gen.multivariate_hypergeometric(total_ac_per_residue.astype(int), n_alleles, n_sim).T
    return null_ac_per_residue


def get_case_control_ac_matrix(df, n_res, n_sim, seed=None):
    case_ac_per_residue = get_ac_per_residue(df, 'ac_case', n_res)
    control_ac_per_residue = get_ac_per_residue(df, 'ac_control', n_res)
    total_ac_per_residue = (case_ac_per_residue + control_ac_per_residue).flatten()
    null_case_ac_per_residue = get_random_ac_per_residue(case_ac_per_residue, total_ac_per_residue, n_sim, seed)
    null_control_ac_per_residue = total_ac_per_residue[:, np.newaxis] - null_case_ac_per_residue
    case_ac_matrix = np.hstack([case_ac_per_residue, null_case_ac_per_residue])
    control_ac_matrix = np.hstack([control_ac_per_residue, null_control_ac_per_residue])
    return case_ac_matrix, control_ac_matrix


def get_stratified_case_ac_matrix(
    df: pd.DataFrame,
    stratum_col: str,
    n_res: int,
    n_sim: int,
    seed: int | list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified permutation null (case labels redrawn within stratum), drawn only on residues with alleles."""
    if stratum_col not in df.columns:
        raise KeyError(f"Missing stratum column '{stratum_col}'")
    if df[stratum_col].isna().any():
        raise ValueError(f"Missing values found in stratum column '{stratum_col}'")
    pos = df["aa_pos"].to_numpy(dtype=np.int64) - 1
    bad = (pos < 0) | (pos >= n_res)
    if bad.any():
        raise ValueError(f"Amino-acid positions outside structure bounds: {(pos[bad] + 1)[:10].tolist()}")

    labels = df[stratum_col].astype(str).to_numpy()
    strata = np.sort(np.unique(labels))
    if len(strata) == 0:
        raise ValueError("No strata found")
    codes = np.searchsorted(strata, labels)
    case_by_stratum = np.zeros((n_res, len(strata)), dtype=np.int64)
    total_by_stratum = np.zeros((n_res, len(strata)), dtype=np.int64)
    ac_case = df["ac_case"].to_numpy(dtype=np.int64)
    np.add.at(case_by_stratum, (pos, codes), ac_case)
    np.add.at(total_by_stratum, (pos, codes), ac_case + df["ac_control"].to_numpy(dtype=np.int64))

    rng = np.random.default_rng(seed)
    null_case = np.zeros((n_res, n_sim), dtype=np.int64)
    for j in range(len(strata)):
        idx = np.flatnonzero(total_by_stratum[:, j])
        if len(idx):
            null_case[idx] += draw_stratified_null_cases(
                total_by_stratum[idx, j], case_by_stratum[:, j].sum(), n_sim, rng
            )
    case_ac_matrix = np.hstack([case_by_stratum.sum(1)[:, None], null_case])
    return case_ac_matrix, case_by_stratum, total_by_stratum


def draw_stratified_null_cases(total_per_residue, n_case, n_sims, rng):
    total_per_residue = np.asarray(total_per_residue, dtype=np.int64)
    n_total = int(total_per_residue.sum())
    n_case = int(n_case)

    if n_case < 0 or n_case > n_total:
        raise ValueError("Invalid case total for multivariate hypergeometric draw")
    if n_sims <= 0:
        return np.empty((len(total_per_residue), 0), dtype=np.int32)
    if n_case == 0:
        return np.zeros((len(total_per_residue), n_sims), dtype=np.int32)
    if n_case == n_total:
        return np.repeat(
            total_per_residue[:, None].astype(np.int32),
            n_sims,
            axis=1,
        )

    draws = rng.multivariate_hypergeometric(
        total_per_residue,
        n_case,
        size=n_sims,
    )
    return draws.T.astype(np.int32, copy=False)
