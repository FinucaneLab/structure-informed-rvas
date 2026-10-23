"""
Correctness tests for reading mutation-rate results into the visualization tools.

The failure mode these guard against is silent. A mutation-rate p-value file reuses the
dataset names of a standard 3DNT file but fills them with different quantities: the
second column of `_original` is mutational opportunity rather than a control allele
count, and the second column of `_nbhd` is an expected de novo count rather than an
observed control count. Mutation rates are not allele counts, and code written for
case/control does not crash on them -- it finds no case-only residues, a near-constant
ratio, and (at the gnomAD scale used here, ~2e-08) no controls either, producing a
plausible-looking answer from the wrong column.

So these tests assert on the column *names* as much as the values: in mutation-rate mode
`ac_control` and `ratio` must be absent, so a caller written for the other mode fails
loudly instead of quietly.

Run: python tests/test_visualize_mu.py
"""
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (write_dataset, resolve_pval_node, read_p_values,
                   read_original_mutation_data)
from visualize_and_interpret import (_nbhd_features_table, _result_columns,
                                     MU_RESULT_COLUMNS, STD_RESULT_COLUMNS,
                                     normalize_result_columns,
                                     positions_to_pymol_notation)
from pymol_code import _obs_exp_to_b

FAILURES = []

UNIPROT = 'P00001'
N_RES = 20


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def _pvals_frame(second_col_name, second_col, original_second):
    """A minimal per-residue result frame in the layout both writers use."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'p_value': np.linspace(0.001, 0.9, N_RES),
        'nbhd_case': np.arange(N_RES) % 4,
        second_col_name: second_col,
        'radius': np.full(N_RES, 15.0),
        'original_case': (np.arange(N_RES) % 3 == 0).astype(int),
        'original_second': original_second,
    })
    df['null_pval_0'] = rng.random(N_RES)
    return df


def _write_file(path, mu_mode):
    """Write a p-value file of either flavour, with the real dataset layout."""
    if mu_mode:
        # mu at the gnomAD per-residue scale -- the magnitude is the whole reason
        # case/control code fails silently on these files.
        mu = np.full(N_RES, 3.7e-8)
        df = _pvals_frame('nbhd_expected', np.linspace(0.5, 4.0, N_RES), mu)
        with h5py.File(path, 'w') as fid:
            fid.attrs['test_type'] = 'mutation_rate'
            for group in ('poisson', 'binomial'):
                node = fid.require_group(group)
                write_dataset(node, UNIPROT, df[['p_value']])
                write_dataset(node, f'{UNIPROT}_null_pval', df[['null_pval_0']])
                write_dataset(node, f'{UNIPROT}_nbhd', df[['nbhd_case', 'nbhd_expected']])
                write_dataset(node, f'{UNIPROT}_radius', df[['radius']])
                write_dataset(node, f'{UNIPROT}_original', df[['original_case', 'original_second']])
    else:
        df = _pvals_frame('nbhd_control', np.arange(N_RES) % 5, np.arange(N_RES) % 2)
        with h5py.File(path, 'w') as fid:
            write_dataset(fid, UNIPROT, df[['p_value']])
            write_dataset(fid, f'{UNIPROT}_null_pval', df[['null_pval_0']])
            write_dataset(fid, f'{UNIPROT}_nbhd', df[['nbhd_case', 'nbhd_control']])
            write_dataset(fid, f'{UNIPROT}_radius', df[['radius']])
            write_dataset(fid, f'{UNIPROT}_original', df[['original_case', 'original_second']])
    return df


# ---------------------------------------------------------------------------
def test_node_resolution(tmpdir):
    """resolve_pval_node must find the per-test groups, and the root for standard files."""
    print('\n1. Resolving where the datasets live')

    mu_path = os.path.join(tmpdir, 'mu.h5')
    std_path = os.path.join(tmpdir, 'std.h5')
    _write_file(mu_path, mu_mode=True)
    _write_file(std_path, mu_mode=False)

    with h5py.File(mu_path, 'r') as fid:
        node, mu_mode = resolve_pval_node(fid, 'binomial')
        check('mutation-rate file reports mu_mode', mu_mode)
        check('mutation-rate file resolves into the requested group',
              UNIPROT in node and node.name.endswith('binomial'), node.name)
        check('the other test is reachable too',
              UNIPROT in resolve_pval_node(fid, 'poisson')[0])
        try:
            resolve_pval_node(fid, 'nonsense')
            check('an unknown test name is rejected', False)
        except ValueError:
            check('an unknown test name is rejected', True)

    with h5py.File(std_path, 'r') as fid:
        node, mu_mode = resolve_pval_node(fid, 'binomial')
        check('standard file reports mu_mode False', not mu_mode)
        check('standard file resolves to the root', UNIPROT in node)


# ---------------------------------------------------------------------------
def test_mutation_data_columns(tmpdir):
    """The mu-mode frame must not carry case/control column names."""
    print('\n2. Per-residue mutation data columns')

    mu_path = os.path.join(tmpdir, 'mu.h5')
    written = _write_file(mu_path, mu_mode=True)

    with h5py.File(mu_path, 'r') as fid:
        node, mu_mode = resolve_pval_node(fid, 'binomial')
        df = read_original_mutation_data(node, UNIPROT, mu_mode=mu_mode)

    check('ac_control is absent in mutation-rate mode', 'ac_control' not in df.columns,
          f'columns: {list(df.columns)}')
    check('ratio is absent in mutation-rate mode', 'ratio' not in df.columns)
    check('mu is present and carries the opportunity column', 'mu' in df.columns and
          np.allclose(df['mu'].values, written['original_second'].values))
    check('ac_case carries the observed de novo counts',
          np.array_equal(df['ac_case'].values, written['original_case'].values))
    check('aa_pos is 1-based and dense',
          np.array_equal(df['aa_pos'].values, np.arange(1, N_RES + 1)))

    # obs_exp must agree with what read_p_values reports, or the colouring and the
    # results table would disagree about the same neighborhood.
    with h5py.File(mu_path, 'r') as fid:
        node, _ = resolve_pval_node(fid, 'binomial')
        df_p = read_p_values(node, UNIPROT, mu_mode=True)
    check('obs_exp matches read_p_values',
          np.allclose(df['obs_exp'].values, df_p['obs_exp'].values))

    # And the standard path must be untouched.
    std_path = os.path.join(tmpdir, 'std.h5')
    _write_file(std_path, mu_mode=False)
    with h5py.File(std_path, 'r') as fid:
        node, mu_mode = resolve_pval_node(fid, 'binomial')
        df_std = read_original_mutation_data(node, UNIPROT, mu_mode=mu_mode)
    check('standard mode still reports ac_control and ratio',
          'ac_control' in df_std.columns and 'ratio' in df_std.columns)


# ---------------------------------------------------------------------------
def test_feature_table_mu_mode():
    """
    The neighborhood x feature table must not define hits by `ac_control == 0`.

    With mu in that column no residue is ever case-only, so the old rule returns an
    empty hit set and every proportion comes out null -- a ranking built from nothing.
    """
    print('\n3. Neighborhood x feature cross-reference')

    nbhd = set(range(1, 11))
    df_mut = pd.DataFrame({
        'aa_pos': np.arange(1, N_RES + 1),
        'ac_case': [2, 0, 0, 1, 0, 0, 0, 0, 0, 0] + [0] * 10,
        'mu': [1e-8] * 5 + [3e-8] * 5 + [1e-8] * 10,
    })
    features_df = pd.DataFrame([
        {'feature_name': 'binding site', 'functional_site_category': 'Binding site',
         'residues_pymol': positions_to_pymol_notation([1, 2, 3, 4]), 'num_positions': 4},
        {'feature_name': 'far domain', 'functional_site_category': 'Domain',
         'residues_pymol': positions_to_pymol_notation([15, 16]), 'num_positions': 2},
        {'feature_name': 'hotspot half', 'functional_site_category': 'Region',
         'residues_pymol': positions_to_pymol_notation([6, 7, 8, 9, 10]), 'num_positions': 5},
    ])

    out = _nbhd_features_table(nbhd, df_mut, features_df, mu_mode=True)

    check('features outside the neighborhood are dropped',
          'far domain' not in set(out['feature_name']), f'{list(out["feature_name"])}')
    check('mu-mode columns replace the case-only ones',
          {'prop_denovo_in_feature', 'prop_mu_in_feature', 'denovo_positions'}
          <= set(out.columns) and 'prop_case_only_in_feature' not in out.columns,
          f'columns: {list(out.columns)}')

    binding = out[out['feature_name'] == 'binding site'].iloc[0]
    # All 3 de novos in the neighborhood sit at positions 1 and 4, both in this feature.
    check('prop_denovo_in_feature counts alleles, not residues',
          binding['prop_denovo_in_feature'] == 1.0, str(binding['prop_denovo_in_feature']))
    check('denovo positions are listed', binding['denovo_positions'] == '1,4',
          binding['denovo_positions'])
    # Opportunity in the neighborhood: 5 * 1e-8 + 5 * 3e-8 = 2e-7; feature holds 4e-8.
    check('prop_mu_in_feature is the rate share, not the residue share',
          abs(binding['prop_mu_in_feature'] - 0.2) < 1e-9,
          str(binding['prop_mu_in_feature']))
    check('prop_nbhd_in_feature stays a residue share',
          abs(binding['prop_nbhd_in_feature'] - 0.4) < 1e-9)

    hotspot = out[out['feature_name'] == 'hotspot half'].iloc[0]
    # Half the neighborhood's residues but 75% of its opportunity, and no de novos --
    # exactly the case prop_mu_in_feature exists to make visible.
    check('a high-rate feature with no de novos is visible as such',
          hotspot['prop_denovo_in_feature'] == 0.0
          and abs(hotspot['prop_mu_in_feature'] - 0.75) < 1e-9,
          f"denovo {hotspot['prop_denovo_in_feature']}, mu {hotspot['prop_mu_in_feature']}")
    check('sorted by de novo share, descending',
          list(out['prop_denovo_in_feature']) == sorted(out['prop_denovo_in_feature'],
                                                        reverse=True))


# ---------------------------------------------------------------------------
def test_feature_table_std_mode_unchanged():
    """The case/control path must behave exactly as before."""
    print('\n4. Standard-mode feature table is unchanged')

    nbhd = set(range(1, 11))
    df_mut = pd.DataFrame({
        'aa_pos': np.arange(1, N_RES + 1),
        'ac_case':    [2, 0, 0, 1, 0, 3, 0, 0, 0, 0] + [0] * 10,
        'ac_control': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] + [0] * 10,
    })
    features_df = pd.DataFrame([
        {'feature_name': 'binding site', 'functional_site_category': 'Binding site',
         'residues_pymol': positions_to_pymol_notation([1, 2, 3, 4]), 'num_positions': 4},
        {'feature_name': 'mixed', 'functional_site_category': 'Region',
         'residues_pymol': positions_to_pymol_notation([6, 7]), 'num_positions': 2},
    ])

    out = _nbhd_features_table(nbhd, df_mut, features_df, mu_mode=False)
    check('case-only columns are used', 'prop_case_only_in_feature' in out.columns
          and 'prop_mu_in_feature' not in out.columns)
    binding = out[out['feature_name'] == 'binding site'].iloc[0]
    # Case-only alleles in the neighborhood total 3 (positions 1 and 4); position 6 has
    # a control so it does not count.
    check('residues carrying a control are excluded from case-only',
          binding['prop_case_only_in_feature'] == 1.0,
          str(binding['prop_case_only_in_feature']))
    mixed = out[out['feature_name'] == 'mixed'].iloc[0]
    check('a feature holding only case+control alleles scores zero',
          mixed['prop_case_only_in_feature'] == 0.0)


# ---------------------------------------------------------------------------
def test_result_columns_detection():
    """The two results layouts must be told apart by content, not by a flag."""
    print('\n5. Results-table column detection')

    mu_cols = ['uniprot_id', 'aa_pos', 'p_poisson', 'fdr_poisson', 'fwer_poisson',
               'nbhd_case', 'exp_poisson', 'obs_exp_poisson', 'radius_poisson',
               'gene_name', 'p_binomial', 'fdr_binomial', 'fwer_binomial',
               'exp_binomial', 'obs_exp_binomial', 'radius_binomial',
               'fdr', 'fwer', 'sig_fwer_both', 'sig_fdr_both',
               'n_denovo_gene', 'mu_gene', 'mu_coverage']
    std_cols = ['uniprot_id', 'aa_pos', 'p_value', 'fdr', 'fwer', 'nbhd_case',
                'nbhd_control', 'ratio', 'radius', 'gene_name']

    df_mu = pd.DataFrame(columns=mu_cols)
    df_std = pd.DataFrame(columns=std_cols)

    check('mutation-rate table detected', _result_columns(df_mu) is MU_RESULT_COLUMNS)
    check('standard table detected', _result_columns(df_std) is STD_RESULT_COLUMNS)

    # Every column the schema names must actually exist in the matching table, or
    # cmd_analyze fails at the point of use rather than here.
    for columns, df, name in ((MU_RESULT_COLUMNS, df_mu, 'mutation-rate'),
                              (STD_RESULT_COLUMNS, df_std, 'standard')):
        needed = {columns['significance_column']}
        needed |= set(columns['p_column'].values())
        ok = needed <= set(df.columns)
        check(f'{name} layout names only columns that exist', ok,
              f'missing {sorted(needed - set(df.columns))}')
        # Radius columns are optional (a single-radius run still records `radius`), but
        # the first choice for each test must be real where the table has any.
        check(f'{name} radius columns exist',
              all(c in df.columns for cols in columns['radius_columns'].values()
                  for c in cols))


# ---------------------------------------------------------------------------
def test_obs_exp_colour_anchor():
    """obs/exp must map to a diverging scale anchored at the null value."""
    print('\n6. Colour mapping for observed/expected')

    check('obs/exp = 1 lands exactly at the midpoint',
          abs(_obs_exp_to_b(1.0) - 0.5) < 1e-12, str(_obs_exp_to_b(1.0)))
    check('two-fold enrichment and depletion are symmetric about the midpoint',
          abs((_obs_exp_to_b(2.0) - 0.5) + (_obs_exp_to_b(0.5) - 0.5)) < 1e-12)
    check('enrichment is above the midpoint', _obs_exp_to_b(3.0) > 0.5)
    check('depletion is below the midpoint', _obs_exp_to_b(0.3) < 0.5)
    check('monotone in obs/exp',
          np.all(np.diff(_obs_exp_to_b([0.1, 0.5, 1.0, 2.0, 8.0])) > 0))

    # One extreme neighborhood must not flatten the rest of the protein, which is what
    # normalizing by the maximum does.
    vals = _obs_exp_to_b([1.0, 2.0, 1000.0])
    check('clipped at the fold cap so outliers do not compress the scale',
          vals.max() <= 1.0 and abs(vals[1] - 0.5) > 0.2, f'{vals}')
    check('stays inside [0, 1] for extreme and zero input',
          np.all((_obs_exp_to_b([0.0, 1e-9, 1e9]) >= 0.0)
                 & (_obs_exp_to_b([0.0, 1e-9, 1e9]) <= 1.0)))


if __name__ == '__main__':
    tmpdir = tempfile.mkdtemp()
    try:
        test_node_resolution(tmpdir)
        test_mutation_data_columns(tmpdir)
        test_feature_table_mu_mode()
        test_feature_table_std_mode_unchanged()
        test_result_columns_detection()
        test_obs_exp_colour_anchor()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print('\n' + ('ALL TESTS PASSED' if not FAILURES else f'FAILURES: {FAILURES}'))
    sys.exit(1 if FAILURES else 0)
