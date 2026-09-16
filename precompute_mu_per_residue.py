"""
Build the per-residue mutation rate table used by the mutation-rate 3DNT.

Why this exists: the mutational opportunity of a residue is the sum of the rates of
ALL possible missense SNVs at that residue. A cohort de novo file cannot supply that,
because it only lists variants someone actually observed. Taking the universe from
such a file makes M_g a small, observation-selected subset of the gene's true
opportunity -- measured at ~1% of possible missense variants for the ASD counts file,
which biased observed/expected by up to 19.7x across mutation-rate deciles.

So the universe comes from the reference enumeration of all possible missense SNVs
(all_missense_variants_gr38.h5) inner-joined to the rate file, aggregated per
(uniprot_id, aa_pos). Observed counts are joined onto this at analysis time.

Output columns: uniprot_id, aa_pos, mu, n_with_mu, n_possible
  mu          summed mutation rate over possible missense SNVs at the residue
  n_with_mu   how many of them carry a rate
  n_possible  how many exist in the reference enumeration (denominator for mu_coverage)

Usage:
  python precompute_mu_per_residue.py --reference-dir DIR [--mu-file PATH] [--out PATH]
"""
import argparse
import os
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def _encode(pos, ref, alt):
    """pos/ref/alt -> one int64 join key, much cheaper than string concatenation."""
    r = pd.Series(ref).map(BASES).fillna(-1).astype(np.int64)
    a = pd.Series(alt).map(BASES).fillna(-1).astype(np.int64)
    return pos.astype(np.int64) * 16 + r.values * 4 + a.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-dir', required=True)
    ap.add_argument('--mu-file', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    ref_dir = args.reference_dir
    mu_file = args.mu_file or os.path.join(ref_dir, 'roulette_missenses_filtered.parquet')
    out = args.out or os.path.join(ref_dir, 'mu_per_residue.parquet')
    h5_path = os.path.join(ref_dir, 'all_missense_variants_gr38.h5')

    rate_col = 'roulette_rate'
    names = pq.ParquetFile(mu_file).schema_arrow.names
    if rate_col not in names:
        cands = [c for c in names if 'rate' in c.lower() or c.lower() in ('mu', 'u')]
        if len(cands) != 1:
            raise ValueError(f'Could not identify a rate column in {mu_file}: {names}')
        rate_col = cands[0]

    with h5py.File(h5_path, 'r') as f:
        chroms = sorted({k.split('_')[0] for k in f.keys() if f'{k.split("_")[0]}_pos' in f})

    parts = []
    for chrom in chroms:
        with h5py.File(h5_path, 'r') as f:
            if f'{chrom}_pos' not in f:
                continue
            ra = f[f'{chrom}_ref_alt'][:]
            pos = f[f'{chrom}_pos'][:]
            uid = pd.Series(f[f'{chrom}_uniprot_id'][:].flatten()).str.decode('ascii').values
            ref = pd.Series(ra[:, 0].flatten()).str.decode('ascii').values
            alt = pd.Series(ra[:, 1].flatten()).str.decode('ascii').values

        poss = pd.DataFrame({'uniprot_id': uid,
                             'aa_pos': pos[:, 1].astype(np.int32),
                             'key': _encode(pd.Series(pos[:, 0]), ref, alt)})

        tbl = pq.read_table(mu_file, columns=['chrom', 'pos', 'ref', 'alt', rate_col],
                            filters=[('chrom', '=', chrom)])
        if tbl.num_rows == 0:
            print(f'  {chrom}: no rates in {os.path.basename(mu_file)}; '
                  f'{len(poss)} possible missense variants get no rate', flush=True)
            rates = pd.DataFrame({'key': np.array([], dtype=np.int64), 'mu': []})
        else:
            r = tbl.to_pandas()
            rates = pd.DataFrame({'key': _encode(r['pos'], r['ref'].values, r['alt'].values),
                                  'mu': r[rate_col].astype(np.float64).values})
            rates = rates.drop_duplicates('key')

        merged = poss.merge(rates, on='key', how='left')
        merged['has_mu'] = merged.mu.notna()
        agg = merged.groupby(['uniprot_id', 'aa_pos'], as_index=False).agg(
            mu=('mu', 'sum'), n_with_mu=('has_mu', 'sum'), n_possible=('has_mu', 'size'))
        parts.append(agg)
        cov = merged.has_mu.mean()
        print(f'  {chrom}: {len(poss)} possible, coverage {cov:.3f}, '
              f'{len(agg)} residues', flush=True)

    df = pd.concat(parts, ignore_index=True)
    # A residue can appear on more than one chromosome entry only via odd mappings;
    # collapse defensively so the table has one row per (gene, residue).
    df = df.groupby(['uniprot_id', 'aa_pos'], as_index=False).agg(
        mu=('mu', 'sum'), n_with_mu=('n_with_mu', 'sum'), n_possible=('n_possible', 'sum'))
    df.to_parquet(out, index=False)
    print(f'\nwrote {out}')
    print(f'  {len(df)} (gene, residue) rows across {df.uniprot_id.nunique()} genes')
    print(f'  total mu = {df.mu.sum():.6g}')
    print(f'  overall variant-level rate coverage = '
          f'{df.n_with_mu.sum() / df.n_possible.sum():.4f}')


if __name__ == '__main__':
    main()
