import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import gzip
import sklearn.metrics
import hdf5plugin
import os
import re

def get_pairwise_distances(pdb_file, *args):
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


def get_distance_matrix_structure(info_file, pdb_path, uniprot_id):
    info = pd.read_csv(info_file, sep="\t")
    pdb_files = info.loc[info.filename.str.contains(uniprot_id),'filename']

    ## Version 2: central on top of all
    if len(pdb_files)==0:
        raise Exception("Protein not found.")
    elif len(pdb_files)==1:
        # One pdb file in structure
        pathfile = os.path.join(pdb_path, pdb_files[0])
        i = int(re.findall(r"\d+", info.loc[pdb_files.index[0],'pos_covered'])[0])
        j = int(re.findall(r"\d+", info.loc[pdb_files.index[0],'pos_covered'])[1])
        distance_matrix = get_pairwise_distances(pathfile, i, j)
    else:
        # Multiple pdb files in structure
        info = info.iloc[pdb_files.index].copy().reset_index()
        info['startAA'] = info.apply(lambda x: int(re.findall(r"\d+", x.pos_covered)[0]), axis=1)
        info['endAA'] = info.apply(lambda x: int(re.findall(r"\d+", x.pos_covered)[1]), axis=1)
        nAA = info.endAA.values[-1]
        info['startAA_next'] = info.startAA.shift(periods=-1, fill_value=nAA+1)
        info['j'] = np.floor((info.endAA+info.startAA_next-1)/2).astype(int) + 1
        info['i'] = info.j.shift(periods=1, fill_value=1)
    
        distance_matrix2 = np.full(shape=(nAA,nAA), fill_value=np.inf)
        cum_nAA=0
        # Calculate distance matrix for the entire pdb range
        for pdb in range(0,info.shape[0]):
            pathfile = os.path.join(pdb_path, info.filename.values[pdb])
            i = info.startAA[pdb]
            j = info.endAA[pdb]-1
            if (pdb==info.shape[0]-1):
                j=j+1
            distance_matrix2[i-1:j,i-1:j] = get_pairwise_distances(pathfile)
        # Substitute overlapping part using "most central" rule
        for pdb in range(0,info.shape[0]):
            pathfile = os.path.join(pdb_path, info.filename.values[pdb])
            i = info.i[pdb]
            j = info.j[pdb]-1
            i_in_pdb = i-(info.startAA[pdb]-1)
            j_in_pdb = j-(info.startAA[pdb]-1)
            nAA_in_pdb = j-i+1
            distance_matrix2[cum_nAA:cum_nAA+nAA_in_pdb, cum_nAA:cum_nAA+nAA_in_pdb] = get_pairwise_distances(pathfile, i_in_pdb, j_in_pdb)
            cum_nAA = cum_nAA+nAA_in_pdb

    return distance_matrix

def get_adjacency_matrix_pdb(pdb_file, radius):
    pairwise_distances = get_pairwise_distances(pdb_file)
    return (pairwise_distances < radius) * 1

def get_adjacency_matrix(distance_matrix, radius):
    return (distance_matrix < radius) * 1
    
def valid_for_fisher(contingency_table):
    valid_columns = np.all(np.sum(contingency_table, axis=0) > 0)
    valid_rows = np.all(np.sum(contingency_table, axis=1) > 0)
    all_non_neg = np.all(contingency_table >= 0)
    if valid_columns and valid_rows and all_non_neg:
        return True
    else:
        return False

def write_dataset(fid, name, data, clevel=5):
    dset = fid.create_dataset(
        name,
        data = data,
        compression = hdf5plugin.Zstd(clevel=clevel)
    )


def read_p_values(fid, uniprot_id):
    """
    Reads the p values for one uniprot_id from an HDF5 results file,
    with the exception of the null values.
    """
    pvalue_ratio = fid[uniprot_id][:]
    case_control = fid[f'{uniprot_id}_nbhd'][:]
    df = pd.DataFrame({'uniprot_id': uniprot_id,
                       'aa_pos': np.arange(1, pvalue_ratio.shape[0]+1),
                       'p_value': pvalue_ratio[:, 0],
                       'ratio': pvalue_ratio[:, 1],
                       'nbhd_case': case_control[:, 0],
                       'nbhd_control': case_control[:, 1]})
    return df
