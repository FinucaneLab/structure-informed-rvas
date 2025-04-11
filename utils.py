import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import gzip
import sklearn.metrics
import hdf5plugin

def get_pairwise_distances(pdb_file):
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
    ca_atoms = np.array(ca_atoms)
    pairwise_distances = sklearn.metrics.pairwise_distances(ca_atoms)
    return pairwise_distances

def get_adjacency_matrix(pdb_file, radius):
    pairwise_distances = get_pairwise_distances(pdb_file)
    return (pairwise_distances < radius) * 1

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