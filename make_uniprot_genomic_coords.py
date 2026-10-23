"""
Build uniprot_genomic_coords.tsv: one genomic location per protein.

The Manhattan plots in the manuscript figures place each protein at a single point
on the genome. Nothing else in the pipeline needs this, which is why it is a separate
small script rather than part of preprocess_reference_data.py.

The locations are derived from the reference enumeration of possible missense SNVs
(all_missense_variants_gr38.h5), which already carries a genomic position and a
UniProt accession for every variant. A protein's span is the min and max genomic
position over its missense variants, so gene_start/gene_end approximate the coding
span -- not the transcript, which would include UTRs and leading/trailing non-coding
exons. gene_pos, the column the figures actually use, is the midpoint of that span:
strand-agnostic, and for a whole-genome x-axis the choice between midpoint, start and
median shifts a point by less than the width of its marker.

Output columns: uniprot_id, chrom, gene_pos, gene_start, gene_end, n_variants

Usage:
  python make_uniprot_genomic_coords.py --reference-dir <REFERENCE_DIR>
"""
import argparse
import os

import h5py
import hdf5plugin  # noqa: F401  (registers the Zstd filter used by the reference file)
import numpy as np
import pandas as pd


def discover_chroms(fid):
    """
    Chromosomes the file actually has both fields for.

    Guarded as precompute_mu_per_residue.py does: the prefix of every key is not
    necessarily a chromosome, and spans_for_chrom indexes {chrom}_uniprot_id and
    {chrom}_pos unconditionally -- so one auxiliary dataset, or a chromosome stored
    with only some of its fields, would abort the run with a KeyError after the
    earlier chromosomes had already been processed.
    """
    return sorted({k.split('_')[0] for k in fid.keys()
                   if f'{k.split("_")[0]}_pos' in fid
                   and f'{k.split("_")[0]}_uniprot_id' in fid})


def spans_for_chrom(fid, chrom):
    """Min/max genomic position and variant count per uniprot_id on one chromosome."""
    # Group on the raw bytes and decode only the unique accessions afterwards:
    # decoding every row first costs seconds per chromosome for no benefit.
    uniprot_id = fid[f'{chrom}_uniprot_id'][:, 0]
    pos = fid[f'{chrom}_pos'][:, 0]

    df = pd.DataFrame({'uniprot_id': uniprot_id, 'pos': pos})
    g = df.groupby('uniprot_id')['pos'].agg(['min', 'max', 'size']).reset_index()
    g.columns = ['uniprot_id', 'gene_start', 'gene_end', 'n_variants']
    g['uniprot_id'] = g['uniprot_id'].str.decode('ascii')
    g['chrom'] = chrom
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-dir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    h5_path = os.path.join(args.reference_dir, 'all_missense_variants_gr38.h5')
    out = args.out or os.path.join(args.reference_dir, 'uniprot_genomic_coords.tsv')

    with h5py.File(h5_path, 'r') as fid:
        keys = list(fid.keys())
        chroms = discover_chroms(fid)
        parts = []
        for chrom in chroms:
            part = spans_for_chrom(fid, chrom)
            print(f'{chrom}: {len(part)} proteins, {int(part.n_variants.sum())} variants')
            parts.append(part)

    if not parts:
        # pd.concat([]) raises "No objects to concatenate", which says nothing about
        # the real problem: the file has no {chrom}_uniprot_id / {chrom}_pos pair.
        raise ValueError(
            f'No chromosomes found in {h5_path}. Expected datasets named '
            f'{{chrom}}_uniprot_id and {{chrom}}_pos; found: {sorted(keys)[:10]}'
        )

    df = pd.concat(parts, ignore_index=True)

    # A handful of accessions appear on more than one chromosome -- pseudoautosomal
    # genes, and a few whose reference transcript maps to an alt contig as well. The
    # figures merge on uniprot_id alone, so more than one row per protein would
    # silently duplicate that protein's neighborhoods; keep the chromosome carrying
    # most of its variants.
    dup = df.uniprot_id.duplicated(keep=False)
    if dup.any():
        n_dup = df.loc[dup, 'uniprot_id'].nunique()
        print(f'\n{n_dup} proteins appear on more than one chromosome; '
              'keeping the one with the most variants')
        df = (df.sort_values('n_variants', ascending=False)
                .drop_duplicates(subset='uniprot_id', keep='first'))

    df['gene_pos'] = ((df.gene_start + df.gene_end) // 2).astype(np.int64)
    df = df[['uniprot_id', 'chrom', 'gene_pos', 'gene_start', 'gene_end', 'n_variants']]
    df = df.sort_values(['chrom', 'gene_pos']).reset_index(drop=True)

    df.to_csv(out, sep='\t', index=False)
    print(f'\nWrote {len(df)} proteins to {out}')


if __name__ == '__main__':
    main()
