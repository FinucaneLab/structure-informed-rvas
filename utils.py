import numpy as np
from Bio.PDB import PDBParser

def get_pairwise_distances(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atoms.append(residue['CA'].get_coord())
    ca_atoms = np.array(ca_atoms)
    pairwise_distances = np.sqrt(np.sum((ca_atoms[:, np.newaxis] - ca_atoms) ** 2, axis=-1))
    return pairwise_distances

def get_adjacency_matrix(pairwise_distances, radius):
    return (pairwise_distances < radius) * 1

def valid_for_fisher(contingency_table):
    valid_columns = np.all(np.sum(contingency_table, axis=0) > 0)
    valid_rows = np.all(np.sum(contingency_table, axis=1) > 0)
    all_non_neg = np.all(contingency_table >= 0)
    if valid_columns and valid_rows and all_non_neg:
        return True
    else:
        return False

def get_pairwise_distance_matrix(pdb_file):
    '''
    return all pairwise distances
    '''
    
def fdr_uncorrelated():
    '''
    compute fdr from a list of p-values. used to correct across genes.
    '''

def empirical_fdr():
    '''
    compute fdr from null permutations
    '''