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
    if len(resAnnot_checked) == 0:
        warnings.warn(
            f"{annotation_id} ({uniprot_id}): no annotation residues within adjacency matrix range "
            f"(protein length={adjacency_matrix.shape[0]}, residues={resAnnot})",
            UserWarning
        )
        return np.array([])
    # print(f"resAnnot: {resAnnot}, resAnnot_checked: {resAnnot_checked}", 'type:', type(resAnnot_checked[0]))
    if len(resAnnot_checked)<len(resAnnot):
        warnings.warn(f"Warning: For '{uniprot_id}' annotation aa_pos not entirely within adj_matrix range. subsetting.", UserWarning)
    
    adjacency_matrix = adjacency_matrix[:, resAnnot_checked-1] # restrict columns to annotation residues (correct for zero-based indexing)
    is_neighbor = adjacency_matrix.max(axis=1)
    #return np.where(is_neighbor>0)[0]
    return np.where(is_neighbor>0)[0] + 1 # re-add one to go back to aa-pos one-based indexing!! 

def loop_annotations(annotation_id, df_rvas, pdb_file_pos_guide, pdb_dir, pae_dir, results_dir, df_annot, df_filter, radius, pae_cutoff):
    
    # print('annotation_id:', annotation_id)
    df_annot = df_annot[df_annot.annotation_id == annotation_id]
    uniprot_id = df_annot.uniprot_id.unique()[0]
    # print('protein uniprot_id:', uniprot_id)
    
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
        if 'aa_pos' in df_filter.columns:
            if 'aa_ref' in df_filter.columns:
                if 'aa_alt' in df_filter.columns:
                    df_rvas_curr = df_rvas_curr.merge(df_filter, on=['uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt'], how='inner')
                else:
                    df_rvas_curr = df_rvas_curr.merge(df_filter, on=['uniprot_id', 'aa_pos', 'aa_ref'], how='inner')
            else:
                df_rvas_curr = df_rvas_curr.merge(df_filter, on=['uniprot_id', 'aa_pos'], how='inner')
        else: 
            df_rvas_curr = df_rvas_curr.merge(df_filter, on=['uniprot_id'], how='inner')
        
        #df_rvas_curr = df_rvas_curr.merge(df_filter, on=['uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt'], how='inner')
        n_res_annot_filtered =  df_rvas_curr.shape[0]
        
    if n_res_annot_filtered==0:
        ## If no annotated residues remain after filtering
        logger.warning(f'{annotation_id}: No variants remain after filtering.')
        return (uniprot_id, annotation_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    # print('df_rvas_curr:', df_rvas_curr)
    #os.makedirs(f'{results_dir}/neighborhoods_aa_pos', exist_ok=True)
    #df_rvas_curr.to_csv(f'{results_dir}/neighborhoods_aa_pos/df_rvas_curr_{annotation_id}.tsv', sep='\t', index=False)

    ### Perform Fischer's exact test
    inCas = df_rvas_curr.loc[df_rvas_curr.hasAnnot.astype(bool), 'ac_case'].sum()
    inCon = df_rvas_curr.loc[df_rvas_curr.hasAnnot.astype(bool), 'ac_control'].sum()
    outCas = df_rvas_curr.loc[~df_rvas_curr.hasAnnot.astype(bool), 'ac_case'].sum()
    outCon = df_rvas_curr.loc[~df_rvas_curr.hasAnnot.astype(bool), 'ac_control'].sum()
    
    return perform_fischer_exact(inCas, outCas, inCon, outCon, annotation_id, uniprot_id)

def _preprocess_scan_data(df_rvas, ignore_ac):
    """Preprocess scan data based on ignore_ac flag."""
    if not ignore_ac:
        return df_rvas
    
    logger.debug("Applying ignore_ac preprocessing")
    df_processed = df_rvas.copy()
    df_processed['ac_case'] = (df_processed['ac_case'] > 0).astype(int)
    df_processed['ac_control'] = (df_processed['ac_control'] > 0).astype(int)
    df_processed['to_drop'] = df_processed['ac_case'] + df_processed['ac_control'] > 1
    df_processed = df_processed[~df_processed.to_drop].copy()
    df_processed.drop('to_drop', axis=1, inplace=True)
    
    return df_processed
    
def _filter_proteins_by_allele_count(df_rvas, df_fdr_filter, min_alleles=5):
    """Filter proteins to include only those with sufficient case and control alleles."""
    grouped = df_rvas.groupby('uniprot_id')[['ac_case', 'ac_control']].sum()
    ac_high_enough = grouped[(grouped['ac_case'] > min_alleles) & (grouped['ac_control'] > min_alleles)]
    uniprot_id_list = ac_high_enough.index.tolist()
    
    if df_fdr_filter is not None:
        uniprot_id_list = np.intersect1d(uniprot_id_list, np.unique(df_fdr_filter.uniprot_id))
    
    logger.info(f"Selected {len(uniprot_id_list)} proteins for analysis (min {min_alleles} alleles each)")
    return uniprot_id_list

def annotation_test(
        df_rvas,
        annotation_file,
        annotation_id,
        reference_dir,
        neighborhood_radius,
        pae_cutoff,
        results_dir,
        df_filter, #e.g. list of high alpha missense
        ignore_ac,
    ):

    pdb_file_pos_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'
    
    logger.info("Starting annotation test analysis")
    logger.info(f"Input dataset contains {len(df_rvas)} variants across {df_rvas['uniprot_id'].nunique()} proteins")

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
        
    # Preprocess data
    df_processed = _preprocess_scan_data(df_rvas, ignore_ac)
    
    # Filter proteins by allele count
    uniprot_id_list = _filter_proteins_by_allele_count(df_processed, df_filter)

    uniprot_id_list_annot = df_annot.uniprot_id.unique()

    uniprot_id_list = list(set(uniprot_id_list) & set(uniprot_id_list_annot))
    
    # filter annotation df to just those on valid uniprot ids
    df_annot = df_annot[df_annot.uniprot_id.isin(uniprot_id_list)]
    
    # annotation id
    annot_list = annotation_id.split(',')
    #df_annot['annotation_id'] = df_annot[annot_list].astype(str).agg('_'.join, axis=1)
    # replace na with blanks - have to make sure annotation file is clean
    df_annot['annotation_id'] = (
        df_annot[annot_list]
        .fillna('')
        .apply(lambda row: '_'.join(map(str, row)), axis=1)
    )
    
    annotation_id_list = df_annot.annotation_id.unique()
    logger.info(f"Running annotation test on {len(annotation_id_list)} annotations")

    ## loop for all valid annotations (which may be the same as all proteins, dependent on annotation type)
    fet = list(map(functools.partial(loop_annotations, 
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
                                         
    # this list will contain an entry per annotation, which will be a tuple (size 8) constisting of:
    # - the uniprot_id
    # - the annotation_id
    # - the contingency table (4 entries)
    # - the odds ratio
    # - the pvalue of the Fischer's exact test
    
    df_fet = pd.DataFrame(fet, columns=['uniprot_id', 'annotation_id', 'in_case', 'out_case', 'in_control', 'out_control', 'or', 'p'])
    pvals = df_fet['p'].to_list()
    p_fdr, fdr_reject = perform_fdr_corretion(pvals)
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
    - annotation file may only have uniprot_id and aa_pos

    this function loops over annotation_ids (often the same as proteins). for each annotation_id, it takes the annotation, uses the 
    pdb files to extend by the neighborhood radius, then filters using the filter file. then 
    performs fisher's exact to compare the resulting set of variants to the background of the 
    whole protein.

    df_rvas: pandas dataframe with columns uniprot_id, aa_pos, aa_ref, aa_alt, ac_case, and ac_control
    '''
