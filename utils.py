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
    info = pd.read_csv(pdb_file_pos_guide, sep="\t")
    pdb_files = info.loc[info.pdb_filename.str.contains(uniprot_id),'pdb_filename']
    ## Version 2: central on top of all
    if len(pdb_files)==0:
        raise Exception(f"Protein {uniprot_id} not found.")
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
    info = pd.read_csv(pae_file_pos_guide, sep="\t")
    #pae_files = info.loc[info.pae_filename.str.contains(uniprot_id),'pae_filename']
    # need to deal with null entries 
    pae_files = info.loc[info.pae_filename.notnull() & info.pae_filename.str.contains(uniprot_id), 'pae_filename']
    ## Version 2: central on top of all
    if len(pae_files)==0:
        pdb_files = info.loc[info.pdb_filename.str.contains(uniprot_id), 'pdb_filename']
        if len(pdb_files)==0:
            raise Exception(f"Protein {uniprot_id} not found.")
        else:
            warnings.warn(f"PAE file not found for Protein {uniprot_id}. No PAE filtering will be used.")
            pae_matrix = None
    elif len(pae_files)==1:
        # One pae file for structure
        pathfile = os.path.join(pae_dir, pae_files.iloc[0])
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

# def get_adjacency_matrix_pdb(pdb_file, radius):
#     pairwise_distances = get_pairwise_distances(pdb_file)
#     return (pairwise_distances < radius) * 1

# def get_adjacency_matrix(pdb_file_pos_guide, pdb_dir, uniprot_id, radius):
#     distance_matrix = get_distance_matrix_structure(pdb_file_pos_guide, pdb_dir, uniprot_id)
#     return (distance_matrix < radius) * 1

def get_adjacency_matrix(pdb_pae_file_pos_guide, pdb_dir, pae_dir, uniprot_id, radius, pae_cutoff):
    logger.debug(f"Computing adjacency matrix for {uniprot_id} with radius={radius}, pae_cutoff={pae_cutoff}")
    distance_matrix = get_distance_matrix_structure(pdb_pae_file_pos_guide, pdb_dir, uniprot_id)
    dist_thresh = (distance_matrix < radius) * 1
    if pae_cutoff == 0:
        pae_thresh = np.ones_like(dist_thresh)
    else:
        pae_matrix = get_pae_matrix_structure(pdb_pae_file_pos_guide, pae_dir, uniprot_id)
        if pae_matrix is None:
            pae_thresh = np.ones_like(dist_thresh)
            logger.warning(f"No PAE matrix found for {uniprot_id}, using distance-only adjacency")
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


def read_p_values(fid, uniprot_id):
    """
    Reads the p values for one uniprot_id from an HDF5 results file,
    with the exception of the null values.
    """
    pvalue_data = fid[uniprot_id][:]
    case_control = fid[f'{uniprot_id}_nbhd'][:]
    
    # Calculate ratio on-the-fly
    nbhd_case = case_control[:, 0]
    nbhd_control = case_control[:, 1]
    n_case_total = nbhd_case.sum()
    n_control_total = nbhd_control.sum()
    ratio = (nbhd_case + 2) / (nbhd_control + 2 * n_control_total / n_case_total)
    
    df = pd.DataFrame({'uniprot_id': uniprot_id,
                       'aa_pos': np.arange(1, pvalue_data.shape[0]+1),
                       'p_value': pvalue_data[:, 0],
                       'ratio': ratio,
                       'nbhd_case': nbhd_case,
                       'nbhd_control': nbhd_control})
    return df

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
