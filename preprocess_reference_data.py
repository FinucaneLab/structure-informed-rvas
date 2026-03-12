"""
Internal data preparation scripts for generating reference files.

These scripts were used to create the reference data distributed with this repo
and are not required for running the 3D neighborhood test. Users should download
the pre-built reference data as described in the README.

- prepare_hdf5(): converts a TSV of missense variants to the HDF5 reference file
- make_common_filter(): reads gnomAD v4 allele frequencies from Google Cloud Storage
  and writes the common_variants_uniprot.tsv filter file
"""

import pandas as pd
import h5py
import hdf5plugin
import csv
from read_data import load_ref_for_chrom
import os

def write_dataset(fid, name, data):
    dset = fid.create_dataset(
        name,
        data = data,
        compression = hdf5plugin.Zstd(clevel=22)
    )

def prepare_hdf5():
    fid = h5py.File('all_missense_variants_gr38.h5','w')
    variants = pd.read_csv('all_missense_variants_gr38.tsv.gz', usecols=['chrom', 'pos', 'ref', 'alt', 'aa_pos', 'aa_ref', 'aa_alt', 'uniprot_id', 'filename', 'file_pos'], sep='\t')
    for chrom, df in variants.groupby('chrom'):
        write_dataset(fid, f'{chrom}_ref_alt', df[['ref', 'alt', 'aa_ref', 'aa_alt']].astype('S') )
        write_dataset(fid, f'{chrom}_filename', df[['filename']].astype('S'))
        write_dataset(fid, f'{chrom}_uniprot_id', df[['uniprot_id']].astype('S'))
        write_dataset(fid, f'{chrom}_pos', df[['pos', 'aa_pos', 'file_pos']])
    fid.close()

def make_common_filter(reference_directory):
    af = 'gs://missense-scoring/ancestries_gnomad_v4.tsv.bgz/ancestries_gnomad_v4.tsv.bgz'
    pops = ['afr', 'amr', 'asj', 'eas', 'fin', 'mid', 'nfe', 'sas', 'remaining']
    temp_out = 'common_variants_grch38.tsv'
    first_chunk = True

    for chunk in pd.read_csv(
        af,
        sep='\t',
        compression='gzip',
        chunksize=500_000
    ):
        # compute frequencies
        for pop in pops:
            chunk[f'freq_{pop}'] = chunk[f'AC_{pop}'] / chunk[f'AN_{pop}']

        chunk['max_af'] = chunk[[f'freq_{pop}' for pop in pops]].max(axis=1)

        common = chunk.loc[chunk['max_af'] > 0.05, ['locus', 'alleles']]

        # append to disk
        common.to_csv(
            temp_out,
            sep='\t',
            index=False,
            quoting=csv.QUOTE_NONE,
            mode='w' if first_chunk else 'a',
            header=first_chunk
        )

        first_chunk = False

    common = pd.read_csv(temp_out, sep='\t')
    common['chr'] = common['locus'].str.split(':').str[0]
    common['pos'] = common['locus'].str.split(':').str[1].astype(int)
    common['ref'] = common['alleles'].str.split('"').str[1]
    common['alt'] = common['alleles'].str.split('"').str[3]
    common['Variant ID'] = common['chr'] + '-' + common['pos'].astype(str) + '-' + common['ref'] + '-' + common['alt']
    result = []
    ref_path = f'{reference_directory}/all_missense_variants_gr38.h5'
    for chrom, rvas_data_by_chr in common.groupby('chr'):
        ref = load_ref_for_chrom(ref_path, chrom, rvas_data_by_chr['pos'])
        if ref is None:
            continue
        joined = rvas_data_by_chr.join(ref, on='Variant ID', how='inner', rsuffix='_ref')
        joined = joined[['Variant ID', 'uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt']]
        result.append(joined)
    result = pd.concat(result)
    result.to_csv(f'{reference_directory}/common_variants_uniprot.tsv', sep='\t')
    os.remove(temp_out)

if __name__ == "__main__":
    # prepare_hdf5()
    reference_directory = '../sir-reference-data/'
    make_common_filter(reference_directory)