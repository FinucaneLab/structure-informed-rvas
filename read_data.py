import pandas as pd
import numpy as np
import h5py
import hdf5plugin
from logger_config import get_logger

logger = get_logger(__name__)

def load_ref_for_chrom(path, chrom, pos_filter):
    with h5py.File(path,'r') as f:
        if f'{chrom}_ref_alt' not in f:
            chrom_names = list(f.keys())
            chrom_names = set(chrom.split('_')[0] for chrom in chrom_names)
            chrom_names = ', '.join(sorted(chrom_names))
            logger.error(f'Chromosome {chrom} not found in reference. Chromosomes in reference: {chrom_names}.')
            return None
            
        ref_alt = f[f'{chrom}_ref_alt'][:]
        pdb_filename = f[f'{chrom}_filename'][:]
        uniprot_id = f[f'{chrom}_uniprot_id'][:]
        positions = f[f'{chrom}_pos'][:]

    pos_filter = set(pos_filter)
    subset = np.where( pd.Series(positions[:,0].flatten()).isin(pos_filter) )[0]
    
    df = pd.DataFrame({ 'ref': ref_alt[subset,0].flatten(),
                        'alt': ref_alt[subset,1].flatten(),
                        'aa_ref': ref_alt[subset,2].flatten(),
                        'aa_alt': ref_alt[subset,3].flatten(),
                        'pdb_filename': pdb_filename[subset].flatten(),
                        'uniprot_id': uniprot_id[subset].flatten(),
                        'pos': positions[subset,0].flatten(),
                        'aa_pos': positions[subset,1].flatten(),
                        'aa_pos_file': positions[subset,2].flatten(),
                     })

    for col in ['aa_ref', 'aa_alt', 'ref', 'alt', 'pdb_filename', 'uniprot_id']:
        df[col] = df[col].str.decode('ascii')

    df['index'] = chrom + '-' + df['pos'].astype(str) + '-' + df['ref'] + '-' + df['alt']
    df.set_index('index', inplace=True)

    return df


def load_mu_for_chrom(mu_path, chrom, pos_filter):
    """
    Load per-variant mutation rates for one chromosome from a rate reference file
    (e.g. roulette_missenses_filtered.parquet), restricted to the positions present
    in the RVAS data.

    Returns a dataframe indexed by chr-pos-ref-alt with column mu (and mu_adjusted
    when the rate file carries an `adjusted` flag), or None if the chromosome is
    absent from the rate file.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
    except ImportError:
        raise ImportError(
            'Reading --mu-file requires pyarrow. Install it with `pip install pyarrow`.'
        )

    schema_names = pq.ParquetFile(mu_path).schema_arrow.names
    if 'roulette_rate' in schema_names:
        rate_col = 'roulette_rate'
    else:
        candidates = [c for c in schema_names if 'rate' in c.lower() or c.lower() in ('mu', 'u')]
        if len(candidates) != 1:
            raise ValueError(
                f'Could not identify a mutation rate column in {mu_path}. '
                f'Columns present: {schema_names}'
            )
        rate_col = candidates[0]

    columns = ['chrom', 'pos', 'ref', 'alt', rate_col]
    if 'adjusted' in schema_names:
        columns.append('adjusted')

    table = pq.read_table(mu_path, columns=columns, filters=[('chrom', '=', chrom)])
    if table.num_rows == 0:
        logger.warning(f'Chromosome {chrom} not found in mutation rate file {mu_path}.')
        return None

    pos_values = pa.array(sorted({int(p) for p in pos_filter}), type=table['pos'].type)
    table = table.filter(pc.is_in(table['pos'], value_set=pos_values))

    df = table.to_pandas()
    rename = {rate_col: 'mu'}
    if 'adjusted' in df.columns:
        rename['adjusted'] = 'mu_adjusted'
    df = df.rename(columns=rename)
    df['index'] = chrom + '-' + df['pos'].astype(str) + '-' + df['ref'] + '-' + df['alt']
    keep = ['index', 'mu'] + (['mu_adjusted'] if 'mu_adjusted' in df.columns else [])
    df = df[keep].drop_duplicates(subset='index').set_index('index')
    return df


def map_to_protein(
    rvas_path,
    variant_id_col,
    ac_case_col,
    ac_control_col,
    reference_directory,
    which_proteins = 'all',
    genome_build = None,
    delimiter=None,
    mu_col=None,
    mu_file=None,
):
    '''
    rvas_path is a path to a .tsv.gz file with columns chr, pos, ref, alt, ac_case, ac_control.
    which_proteins is either the name of a protein or file with a list of proteins. we could 
    also make a mapper from gene name to uniprot id and allow this to be a gene name or file with
    multiple gene names.

    the output of this function is a dataframe where the columns are uniprot_id, aa_pos, aa_ref,
    aa_alt, pdb_file, file_index, ac_case, and ac_control

    this function should drop any proteins with insufficient data, and only output data for the
    requested proteins.
    '''

    # read data and ensure it has chr, pos and Variant ID columns
    pandas_engine = 'python' if delimiter is None else None
    compression = 'gzip' if rvas_path.endswith('.bgz') else 'infer' # infer doesn't recognize .bgz extension
    rvas_data = pd.read_csv(rvas_path, sep=delimiter, engine=pandas_engine, compression=compression)
    mutation_rate_mode = (mu_col is not None) or (mu_file is not None)
    rvas_data = rvas_data.rename(columns = {
        variant_id_col: 'Variant ID',
        ac_case_col: 'ac_case',
        ac_control_col: 'ac_control',
        mu_col: 'mu',
    })

    if 'Variant ID' in rvas_data:
        rvas_data['Variant ID'] = [x.replace(':', '-') for x in rvas_data['Variant ID']]
        if not all(rvas_data['Variant ID'].str.split('-').str.len() == 4):
            raise Exception('Variant ID should be formatted as chr-pos-ref-alt.')
        if 'chr' not in rvas_data:
            rvas_data['chr'] = rvas_data['Variant ID'].str.split('-').str[0]
        if 'pos' not in rvas_data:
            rvas_data['pos'] = rvas_data['Variant ID'].str.split('-').str[1].astype(int)
    elif ('locus' in rvas_data) and ('alleles' in rvas_data):
        rvas_data['chr'] = rvas_data['locus'].str.split(':').str[0]
        rvas_data['pos'] = rvas_data['locus'].str.split(':').str[1].astype(int)
        if '"' in rvas_data.alleles.iloc[0]:
            rvas_data['ref'] = rvas_data['alleles'].str.split('"').str[1]
            rvas_data['alt'] = rvas_data['alleles'].str.split('"').str[3]
        elif '\'' in rvas_data.alleles.iloc[0]:
            rvas_data['ref'] = rvas_data['alleles'].str.split('\'').str[1]
            rvas_data['alt'] = rvas_data['alleles'].str.split('\'').str[3]
        else:
            rvas_data['ref'] = rvas_data['alleles'].str.split(',').str[0].str[1:]
            rvas_data['alt'] = rvas_data['alleles'].str.split(',').str[1].str[:-1]
        rvas_data['Variant ID'] = rvas_data['chr'] + '-' + rvas_data['pos'].astype(str) + '-' + rvas_data['ref'] + '-' + rvas_data['alt']
    else:
        rvas_data['Variant ID'] = rvas_data['chr'] + '-' + rvas_data['pos'].astype(str) + '-' + rvas_data['ref'] + '-' + rvas_data['alt']

    # detect case/control columns
    def identify_column(name):
        possible_cols = [col for col in rvas_data if 'ac' in col.lower() and name in col.lower() and rvas_data[col].dtype == int]
        if len(possible_cols) == 0:
            possible_cols = [col for col in rvas_data if name in col.lower() and rvas_data[col].dtype == int]
        if len(possible_cols) != 1:
            raise Exception(f'Could not uniquely identify {name} column. Please include a column named ac_{name} in RVAS data.')
        return possible_cols[0]

    if 'ac_case' not in rvas_data:
        rvas_data.rename( {identify_column('case'): 'ac_case'}, axis=1, inplace=True)
    if not mutation_rate_mode and 'ac_control' not in rvas_data:
        rvas_data.rename( {identify_column('control'): 'ac_control'}, axis=1, inplace=True)

    if mu_col is not None:
        if 'mu' not in rvas_data:
            raise Exception(f'Mutation rate column "{mu_col}" not found in {rvas_path}.')
        rvas_data['mu'] = pd.to_numeric(rvas_data['mu'], errors='coerce')
        n_bad = int(rvas_data['mu'].isna().sum())
        if n_bad:
            logger.warning(f'Dropping {n_bad} variants with a non-numeric mutation rate.')
            rvas_data = rvas_data[rvas_data['mu'].notna()]
    # join to reference variants and identify relevant proteins
    result = []
    n_mapped_total = 0
    n_mu_missing_total = 0
    ref_path = f'{reference_directory}/all_missense_variants_gr38.h5'
    for chrom, rvas_data_by_chr in rvas_data.groupby('chr'):
        ref = load_ref_for_chrom(ref_path, chrom, rvas_data_by_chr['pos'])
        if ref is None:
            continue
        joined = rvas_data_by_chr.join(ref, on='Variant ID', how='inner', rsuffix='_ref')

        if mu_file is not None:
            df_mu = load_mu_for_chrom(mu_file, chrom, rvas_data_by_chr['pos'])
            if df_mu is None:
                continue
            n_before = len(joined)
            joined = joined.join(df_mu, on='Variant ID', how='left')
            n_missing = int(joined['mu'].isna().sum())
            if n_missing:
                logger.info(
                    f'{chrom}: {n_missing} of {n_before} mapped variants '
                    f'({100 * n_missing / max(n_before, 1):.1f}%) have no mutation rate; dropping.'
                )
                joined = joined[joined['mu'].notna()]
            n_mapped_total += n_before
            n_mu_missing_total += n_missing

        cols = ['Variant ID', 'uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt',
                'pdb_filename', 'aa_pos_file', 'ac_case']
        if mutation_rate_mode:
            cols.append('mu')
            if 'mu_adjusted' in joined.columns:
                cols.append('mu_adjusted')
        else:
            cols.append('ac_control')
        joined = joined[cols]
        result.append(joined)

    if mu_file is not None and n_mapped_total > 0:
        logger.info(
            f'Mutation rate join: {n_mapped_total - n_mu_missing_total} of {n_mapped_total} '
            f'mapped variants ({100 * (1 - n_mu_missing_total / n_mapped_total):.1f}%) '
            f'have a rate.'
        )

    if len(result) > 0 and result[0].shape[0] == 0:
        logger.warning('Could not identify proteins.')
        if rvas_data.shape[0] > 0 and rvas_data_by_chr.shape[0] > 0:
            logger.warning(f'Does variant id have the same format? Rvas_data: {rvas_data["Variant ID"].iloc[0]}, reference data: {rvas_data_by_chr.index[0]}')

    result = pd.concat(result)
    return result
