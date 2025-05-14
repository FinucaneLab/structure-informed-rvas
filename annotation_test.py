import os
import warnings
from datetime import datetime
import gzip
import functools
import numpy as np
import pandas as pd
from utils import valid_for_fisher, get_adjacency_matrix
from scipy import stats
import statsmodels.stats.multitest as multitest



def perform_fischer_exact(inCas, outCas, inCon, outCon, uniprot_id) :
    contingency_table = np.array([ [inCas, outCas], [inCon, outCon] ])
    if valid_for_fisher(contingency_table):
        print(f'{uniprot_id}: Ran Fischer\'s exact test.')
        o, p = stats.fisher_exact(contingency_table)
    else:
        print(f'{uniprot_id}: Not valid for Fischer\'s exact test.')
        o = np.nan
        p = np.nan
    return (uniprot_id, inCas, outCas, inCon, outCon, o, p)


def perform_fdr_corretion(p):
    p = np.array(p)
    mask = np.isfinite(p)
    p_reject1, p_fdr1 = multitest.fdrcorrection(p[mask], alpha=0.05)
    p_fdr = np.full(p.shape, np.nan)
    p_fdr[mask] = p_fdr1
    p_reject = np.full(p.shape, False)
    p_reject[mask] = p_reject1

    return p_fdr, p_reject

def expand_annot_neighborhood(df_annot, pdb_file_pos_guide, pdb_dir, results_dir, uniprot_id, radius):
    df_annot = df_annot[df_annot.uniprot_id == uniprot_id]
    resAnnot = np.sort(df_annot.aa_pos.unique())
    if len(resAnnot)==0:
        return np.array([])
    if os.path.isfile(os.path.join(results_dir,f'{uniprot_id}.adj_mat.npy')):
        adjacency_matrix = np.load(os.path.join(results_dir,f'{uniprot_id}.adj_mat.npy'))
    else:
        adjacency_matrix = get_adjacency_matrix(pdb_file_pos_guide, pdb_dir, uniprot_id, radius)

    ## sanity check for annotation aa_pos: are all aa_pos within adjacency matrix range?
    resAnnot_checked = resAnnot[resAnnot <= adjacency_matrix.shape[0]]
    if len(resAnnot_checked)<len(resAnnot):
        warnings.warn(f"Warning: For '{uniprot_id}' annotation aa_pos not entirely within adj_matrix range. subsetting.", UserWarning)
    
    adjacency_matrix = adjacency_matrix[:, resAnnot_checked-1] # restrict columns to annotation residues (correct for zero-based indexing)
    is_neighbor = adjacency_matrix.max(axis=1)
    return np.where(is_neighbor>0)[0]
    

def loop_proteins(uniprot_id, df_rvas, pdb_file_pos_guide, pdb_dir, results_dir, df_annot, df_filter, radius):

    ## some sanity checks
    info = pd.read_csv(pdb_file_pos_guide, sep="\t")
    pdb_files = info.loc[info.pdb_filename.str.contains(uniprot_id),'pdb_filename']
    if len(pdb_files)==0:
        warnings.warn(f"Warning: PDB file for '{uniprot_id}' not found. skipping.", UserWarning)
        return (uniprot_id, np.array([]), np.nan, np.nan)
    for file in pdb_files:
        if not os.path.isfile(os.path.join(pdb_dir,file)):
            warnings.warn(f"Warning: PDB file(s) for '{uniprot_id}' not found. skipping.")
            return (uniprot_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        
    expanded_annot_residues = expand_annot_neighborhood(df_annot, pdb_file_pos_guide, pdb_dir, results_dir, uniprot_id, radius)
    n_res_annot = expanded_annot_residues.shape[0]

    if n_res_annot==0:
        ## If no annotation found for uniprot_id
        print(f'{uniprot_id}: No annotated variants found.')
        return (uniprot_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    df_rvas_curr = df_rvas[df_rvas.uniprot_id == uniprot_id].copy()
    df_rvas_curr['hasAnnot'] = 0
    df_rvas_curr.loc[df_rvas_curr.aa_pos.isin(expanded_annot_residues), 'hasAnnot'] = 1

    ## Filter rvas data frame
    n_res_annot_filtered = np.nan
    if df_filter is not None:
        df_rvas_curr = df_rvas_curr.merge(df_filter, on=['uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt'], how='inner')
        n_res_annot_filtered =  df_rvas_curr.shape[0]
        
    if n_res_annot_filtered==0:
        ## If no annotated residues remain after filtering
        print(f'{uniprot_id}: No variants remain after filtering.')
        return (uniprot_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    ### Perform Fischer's exact test
    inCas = df_rvas_curr.loc[df_rvas_curr.hasAnnot.astype(bool), 'ac_case'].sum()
    inCon = df_rvas_curr.loc[df_rvas_curr.hasAnnot.astype(bool), 'ac_control'].sum()
    outCas = df_rvas_curr.loc[~df_rvas_curr.hasAnnot.astype(bool), 'ac_case'].sum()
    outCon = df_rvas_curr.loc[~df_rvas_curr.hasAnnot.astype(bool), 'ac_control'].sum()
    
    return perform_fischer_exact(inCas, outCas, inCon, outCon, uniprot_id)
    

def annotation_test(
        df_rvas,
        annotation_file,
        reference_dir,
        neighborhood_radius,
        results_dir,
        filter_file=None, #e.g. list of high alpha missense
    ):

    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    
    try:
        print(df_rvas)
        uniprot_id_list = df_rvas.uniprot_id.unique()
        print(f"Found {len(uniprot_id_list)} unique UniProt IDs")
    except AttributeError:
        print("Error: df_rvas is not defined or doesn't have a 'uniprot_id' attribute")
    except Exception as e:
        print(f"Error extracting unique UniProt IDs from df_rvas: {e}")

    # read annotation file
    try:
        df_annot = pd.read_csv(annotation_file, sep="\t")
    except FileNotFoundError:
        print(f"Warning: File not found: {annotation_file}")
    except pd.errors.EmptyDataError:
        print(f"Warning: Empty file: {annotation_file}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file {annotation_file}: {e}")

    # read filter file (or set to None)
    if filter_file is not None:
        try:
            df_filter = pd.read_csv(filter_file, sep="\t")
        except FileNotFoundError:
            print(f"Warning: File not found: {filter_file}")
        except Exception as e:
            print(f"Error reading file {filter_file}: {e}")
    else:
        df_filter = None
        
    uniprot_id_list = df_rvas.uniprot_id.unique()
    fet = list(map(functools.partial(loop_proteins, 
                                         df_rvas=df_rvas,
                                         pdb_file_pos_guide=pdb_file_pos_guide, 
                                         pdb_dir=pdb_dir,
                                         results_dir=results_dir,
                                         df_annot = df_annot,
                                         df_filter = df_filter,
                                         radius=neighborhood_radius), 
                       uniprot_id_list))
    # this list will contain an entry per protein, which will be a tuple (size 7) constisting of:
    # - the uniprot_id
    # - the contingency table (4 entries)
    # - the odds ratio
    # - the pvalue of the Fischer's exact test

    pvals = [item[6] for item in fet]
    p_fdr, fdr_reject = perform_fdr_corretion(pvals)
    
    df_fet = pd.DataFrame(fet, columns=['uniprot_id', 'in_case', 'out_case', 'in_control', 'out_control', 'or', 'p'])
    df_fet['p_fdr'] = p_fdr
    df_fet['fdr_reject'] = fdr_reject
    df_fet = df_fet.sort_values(by='p_fdr')

    timestamp_format = "%M%d%m"
    timestamp = datetime.now().strftime(timestamp_format)
    df_fet.to_csv(os.path.join(results_dir, f'annotation_test_results_{timestamp}.fdr.tsv'), sep='\t', index=False, na_rep='NaN')

    return df_fet
    
    '''
    perform annotation test. annotation file and filter file have columns uniprot_id,
    aa_pos, aa_ref, aa_alt, which specify the members of the annotation/filter. 
    reference_directory has pdb_files. 

    this function loops over proteins. for each protein, it takes the annotation, uses the 
    pdb files to extend by the neighborhood radius, then filters using the filter file. then 
    performs fisher's exact to compare the resulting set of variants to the background of the 
    whole protein.

    df_rvas: pandas dataframe with columns uniprot_id, aa_pos, aa_ref, aa_alt, ac_case, and ac_control
    '''