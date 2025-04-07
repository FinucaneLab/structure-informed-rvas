"""
This file prepares the HDF5 file for read_data.py:load_ref_for_chrom

The input of this file is a csv file with genetic variants, their uniprot_id, pdb filename, and position of the variant in the pdb file.
"""

import pandas as pd
import h5py
import hdf5plugin

def write_dataset(fid, name, data):
    dset = fid.create_dataset(
        name,
        data = data,
        compression = hdf5plugin.Zstd(clevel=22)
    )

fid = h5py.File('all_missense_variants_gr38.h5','w')
variants = pd.read_csv('all_missense_variants_gr38.tsv.gz', usecols=['chrom', 'pos', 'ref', 'alt', 'aa_pos', 'aa_ref', 'aa_alt', 'uniprot_id', 'filename', 'file_pos'], sep='\t')
for chrom, df in variants.groupby('chrom'):
    write_dataset(fid, f'{chrom}_ref_alt', df[['ref', 'alt', 'aa_ref', 'aa_alt']].astype('S') )
    write_dataset(fid, f'{chrom}_filename', df[['filename']].astype('S'))
    write_dataset(fid, f'{chrom}_uniprot_id', df[['uniprot_id']].astype('S'))
    write_dataset(fid, f'{chrom}_pos', df[['pos', 'aa_pos', 'file_pos']])
fid.close()