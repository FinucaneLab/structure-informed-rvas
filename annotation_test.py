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
from logger_config import get_logger

logger = get_logger(__name__)


def perform_fischer_exact(inCas, outCas, inCon, outCon, annotation_id, uniprot_id) :
    contingency_table = np.array([ [inCas, outCas], [inCon, outCon] ])
    if valid_for_fisher(contingency_table):
        logger.debug(f'{annotation_id}: Ran Fischer\'s exact test.')
        o, p = stats.fisher_exact(contingency_table)
    else:
        logger.warning(f'{annotation_id}: Not valid for Fischer\'s exact test.')
        o = np.nan
        p = np.nan

    return (uniprot_id, annotation_id, inCas, outCas, inCon, outCon, o, p)


def perform_fdr_corretion(p):
    p = np.array(p)
    mask = np.isfinite(p)
    logger.info(f'Performing FDR correction on {(mask*1).sum()} proteins with valid Fischer test.')
    p_reject1, p_fdr1 = multitest.fdrcorrection(p[mask], alpha=0.05)
    p_fdr = np.full(p.shape, np.nan)
    p_fdr[mask] = p_fdr1
    p_reject = np.full(p.shape, False)
    p_reject[mask] = p_reject1

    return p_fdr, p_reject

def expand_annot_neighborhood(df_annot, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, annotation_id, radius, pae_cutoff):
    df_subset = df_annot[df_annot['annotation_id'] == annotation_id]
    resAnnot = np.sort(df_annot.aa_pos.unique())
    # print(df_annot)
    uniprot_id = df_subset.uniprot_id.unique()[0]

    if len(resAnnot)==0:
        return np.array([])
    if os.path.isfile(os.path.join(results_dir,f'{uniprot_id}.adj_mat.npy')):
        adjacency_matrix = np.load(os.path.join(results_dir,f'{uniprot_id}.adj_mat.npy'))
    else:
        adjacency_matrix = get_adjacency_matrix(pdb_file_pos_guide, pdb_dir, pae_dir, uniprot_id, radius, pae_cutoff)
    
    if adjacency_matrix is None:
        return np.array([])

    ## sanity check for annotation aa_pos: are all aa_pos within adjacency matrix range?
    resAnnot_checked = resAnnot[resAnnot <= adjacency_matrix.shape[0]]
    # print(f"resAnnot: {resAnnot}, resAnnot_checked: {resAnnot_checked}", 'type:', type(resAnnot_checked[0]))
    if len(resAnnot_checked)<len(resAnnot):
        warnings.warn(f"Warning: For '{uniprot_id}' annotation aa_pos not entirely within adj_matrix range. subsetting.", UserWarning)
    
    adjacency_matrix = adjacency_matrix[:, resAnnot_checked-1] # restrict columns to annotation residues (correct for zero-based indexing)
    is_neighbor = adjacency_matrix.max(axis=1)
    return np.where(is_neighbor>0)[0]    

def loop_proteins(annotation_id, df_rvas, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, df_annot, df_filter, radius, pae_cutoff):

#    ## some sanity checks
#    info = pd.read_csv(pdb_file_pos_guide, sep="\t")
#    pdb_files = info.loc[info.pdb_filename.str.contains(uniprot_id),'pdb_filename']
#    if len(pdb_files)==0:
#        warnings.warn(f"Warning: PDB file for '{uniprot_id}' not found. skipping.", UserWarning)
#        return (uniprot_id, np.array([]), np.nan, np.nan)
#    for file in pdb_files:
#        if not os.path.isfile(os.path.join(pdb_dir,file)):
#            warnings.warn(f"Warning: PDB file(s) for '{uniprot_id}' not found. skipping.")
#            return (uniprot_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    # print('annotation_id:', annotation_id)
    df_annot = df_annot[df_annot.annotation_id == annotation_id]
    uniprot_id = df_annot.uniprot_id.unique()[0]
    # print('protein uniprot_id:', uniprot_id)
    # print('df_rvas:', df_rvas)  
    
    if radius>0:
        expanded_annot_residues = expand_annot_neighborhood(df_annot, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, annotation_id, radius, pae_cutoff)
        n_annot = expanded_annot_residues.shape[0]
        ## if expanding, specific aminoacid changes are no longer relevant - drop
        df_annot = pd.DataFrame({'annotation_id': [annotation_id]*n_annot, 'aa_pos': expanded_annot_residues})
    
    df_rvas_curr = df_rvas[df_rvas.uniprot_id == uniprot_id].copy()
    #df_rvas_curr['hasAnnot'] = 0
    #df_rvas_curr.loc[df_rvas_curr.aa_pos.isin(expanded_annot_residues), 'hasAnnot'] = 1
    df_rvas_curr = df_rvas_curr.merge(df_annot, how='left', indicator='hasAnnot')
    ## after merging 'hasAnnot' is either "left_only" (no annotation) or "both" (has annotation)
    df_rvas_curr.hasAnnot = list(map(lambda x: 1 if x=="both" else 0, df_rvas_curr.hasAnnot)) 

    n_res_annot = (df_rvas_curr.hasAnnot*1).sum()
    if n_res_annot==0:
        ## If no annotation found for uniprot_id
        logger.warning(f'{annotation_id}: No annotated variants found.')
        return (uniprot_id, annotation_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    ## Filter rvas data frame
    n_res_annot_filtered = np.nan
    if df_filter is not None:
        df_filter = df_filter[df_filter.uniprot_id == uniprot_id]
        df_rvas_curr = df_rvas_curr.merge(df_filter, on=['uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt'], how='inner')
        n_res_annot_filtered =  df_rvas_curr.shape[0]
        
    if n_res_annot_filtered==0:
        ## If no annotated residues remain after filtering
        logger.warning(f'{annotation_id}: No variants remain after filtering.')
        return (uniprot_id, annotation_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    # print('df_rvas_curr:', df_rvas_curr)
    os.makedirs(f'{results_dir}/neighborhoods_aa_pos', exist_ok=True)
    df_rvas_curr.to_csv(f'{results_dir}/neighborhoods_aa_pos/df_rvas_curr_{annotation_id}.tsv', sep='\t', index=False)

    ### Perform Fischer's exact test
    inCas = df_rvas_curr.loc[df_rvas_curr.hasAnnot.astype(bool), 'ac_case'].sum()
    inCon = df_rvas_curr.loc[df_rvas_curr.hasAnnot.astype(bool), 'ac_control'].sum()
    outCas = df_rvas_curr.loc[~df_rvas_curr.hasAnnot.astype(bool), 'ac_case'].sum()
    outCon = df_rvas_curr.loc[~df_rvas_curr.hasAnnot.astype(bool), 'ac_control'].sum()
    
    return perform_fischer_exact(inCas, outCas, inCon, outCon, annotation_id, uniprot_id)


def check_pdb_files_exist(uniprot_id, pdb_dir, pdb_file_list):
    pdb_files = pdb_file_list.loc[pdb_file_list.pdb_filename.str.contains(uniprot_id),'pdb_filename']
    if len(pdb_files)==0:
        warnings.warn(f"Warning: PDB file for '{uniprot_id}' not found. skipping.", UserWarning)
        return None
    if all(os.path.isfile(os.path.join(pdb_dir,file)) for file in pdb_files):
        return uniprot_id
    else:
        warnings.warn(f"Warning: PDB file(s) for '{uniprot_id}' not found. skipping.")
        return None


def annotation_test(
        df_rvas,
        annotation_file,
        reference_dir,
        neighborhood_radius,
        pae_cutoff,
        results_dir,
        filter_file=None, #e.g. list of high alpha missense
    ):

    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'

    # print('pae dir exists:', os.path.exists(pae_dir))
    if pae_cutoff != 0 and not os.path.exists(pae_dir):
        raise Exception('Error: pae_dir does not exist, but pae_cutoff is not 0.')

    try:
        logger.debug(f"df_rvas content: {df_rvas}")
        uniprot_id_list = df_rvas.uniprot_id.unique()
        logger.info(f"Found {len(uniprot_id_list)} unique UniProt IDs")
    except AttributeError:
        logger.error("df_rvas is not defined or doesn't have a 'uniprot_id' attribute")
    except Exception as e:
        logger.error(f"Error extracting unique UniProt IDs from df_rvas: {e}")

    # read annotation file
    try:
        df_annot = pd.read_csv(annotation_file, sep="\t")
    except FileNotFoundError:
        logger.warning(f"File not found: {annotation_file}")
    except pd.errors.EmptyDataError:
        logger.warning(f"Empty file: {annotation_file}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading file {annotation_file}: {e}")

    # read filter file (or set to None)
    if filter_file is not None:
        try:
            df_filter = pd.read_csv(filter_file, sep="\t")
        except FileNotFoundError:
            logger.warning(f"File not found: {filter_file}")
        except Exception as e:
            logger.error(f"Error reading file {filter_file}: {e}")
    else:
        df_filter = None
        
    uniprot_id_list = df_rvas.uniprot_id.unique()
    uniprot_id_list_annot = df_annot.uniprot_id.unique()

    uniprot_id_list = uniprot_id_list = list(set(uniprot_id_list) & set(uniprot_id_list_annot))
    
    annotation_id_list = df_annot.annotation_id.unique()
    # print('annotation_id_list:', annotation_id_list)    

    ## check if pdb files exist for uniprot_ids
    # logger.debug('Checking PDB files for all proteins...')
    #info = pd.read_csv(pdb_file_pos_guide, sep="\t")
    #uniprot_id_list = list(map(functools.partial(check_pdb_files_exist,
    #                                             pdb_dir=pdb_dir,
    #                                             pdb_file_list=info),
    #                    uniprot_id_list))

    ## loop for all valid proteins
    fet = list(map(functools.partial(loop_proteins, 
                                         df_rvas=df_rvas,
                                         pdb_file_pos_guide=pdb_file_pos_guide, 
                                         pdb_dir=pdb_dir,
                                         pae_dir=pae_dir,
                                         results_dir=results_dir,
                                         df_annot = df_annot,
                                         df_filter = df_filter,
                                         radius=neighborhood_radius,
                                         pae_cutoff=pae_cutoff), 
                       annotation_id_list))
                                         
    # this list will contain an entry per protein, which will be a tuple (size 7) constisting of:
    # - the uniprot_id
    # - the contingency table (4 entries)
    # - the odds ratio
    # - the pvalue of the Fischer's exact test
    # print('fet:', fet)

    # pvals = [item[7] for item in fet]
    # p_fdr, fdr_reject = perform_fdr_corretion(pvals)
    
    df_fet = pd.DataFrame(fet, columns=['uniprot_id', 'annotation_id', 'in_case', 'out_case', 'in_control', 'out_control', 'or', 'p'])
    pvals = df_fet['p'].to_list()
    p_fdr, fdr_reject = perform_fdr_corretion(pvals)
    df_fet['p_fdr'] = p_fdr
    df_fet['fdr_reject'] = fdr_reject
    df_fet = df_fet.sort_values(by='p_fdr')

    timestamp_format = "%M%d%m"
    timestamp = datetime.now().strftime(timestamp_format)
#    df_fet.to_csv(os.path.join(results_dir, f'annotation_test_results_genebass-p4-foldx2-ar_{timestamp}.fdr.tsv'), sep='\t', index=False, na_rep='NaN')
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
