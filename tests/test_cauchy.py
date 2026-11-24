"""
Cauchy combination test (CCT) over gene-level p-values from several runs.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cauchy import cct, combine_gene_pvalues  # noqa: E402


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('p', [1e-6, 0.001, 0.05, 0.3, 0.5, 0.9, 0.999])
def test_identical_inputs_return_that_input(p):
    """CCT of k copies of p is p. The sharpest check on the arctan branch."""
    for k in (1, 2, 5, 14):
        assert cct([p] * k) == pytest.approx(p, rel=1e-12)


def test_dominated_by_the_smallest():
    """The Cauchy tail is heavy: one strong input carries the combination even when
    every other input is uninformative."""
    combined = cct([1e-6] + [0.9] * 13)
    assert combined < 1e-4


def test_monotone_in_each_input():
    base = [0.2, 0.4, 0.6]
    better = [0.02, 0.4, 0.6]
    assert cct(better) < cct(base)


def test_lies_in_the_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.random(rng.integers(1, 15))
        v = cct(p)
        assert 0.0 < v <= 1.0


def test_roughly_uniform_under_the_null():
    """Independent uniform inputs: CCT is calibrated, so the output should be about
    uniform. A loose check -- the point is to catch a sign or branch error, not to
    verify the tail approximation."""
    rng = np.random.default_rng(1)
    vals = np.array([cct(rng.random(5)) for _ in range(4000)])
    assert abs((vals < 0.05).mean() - 0.05) < 0.015
    assert abs((vals < 0.5).mean() - 0.5) < 0.04


def test_weights_are_honoured():
    p = [0.001, 0.9]
    assert cct(p, weights=[1, 1]) == pytest.approx(cct(p))
    assert cct(p, weights=[10, 1]) < cct(p, weights=[1, 10])


def test_rejects_zero():
    """gene_p_raw can be 0; gene_p cannot. Passing the wrong column must not quietly
    produce a combined p of 0."""
    with pytest.raises(ValueError, match='must lie in'):
        cct([0.0, 0.5])


def test_rejects_out_of_range_and_nan():
    with pytest.raises(ValueError, match='must lie in'):
        cct([1.5, 0.5])
    with pytest.raises(ValueError, match='non-finite'):
        cct([np.nan, 0.5])


def test_a_single_saturated_input_collapses_the_combination():
    """Why the clamp exists: tan((0.5-p)*pi) at p = 1 is ~ -1.6e16, which swamps the
    mean no matter what the other inputs say."""
    assert cct([1e-5] * 13 + [1.0]) == pytest.approx(1.0)
    assert cct([1e-5] * 13 + [1.0 - 1e-4]) < 1e-3


# ---------------------------------------------------------------------------
# combining files
# ---------------------------------------------------------------------------

def _gene_file(path, rows, n_sims=10000):
    pd.DataFrame([
        {'uniprot_id': uid, 'gene_name': name, 'gene_p': p, 'n_pos': 10,
         'obs_min_p': p / 10, 'best_aa_pos': pos, 'best_radius': 8.0,
         'n_null_le_obs': int(p * (n_sims + 1)) - 1, 'n_sims': n_sims,
         'gene_p_raw': p}
        for uid, name, p, pos in rows
    ]).to_csv(path, sep='\t', index=False)


@pytest.fixture
def inputs(tmp_path):
    _gene_file(tmp_path / 'a.gene.tsv',
               [('P1', 'G1', 0.001, 100), ('P2', 'G2', 0.40, 10)])
    _gene_file(tmp_path / 'b.gene.tsv',
               [('P1', 'G1', 0.004, 250), ('P2', 'G2', 0.90, 20)])
    _gene_file(tmp_path / 'c.gene.tsv',
               [('P1', 'G1', 0.002, 100), ('P2', 'G2', 0.60, 30)])
    return tmp_path, ['a.gene.tsv', 'b.gene.tsv', 'c.gene.tsv']


def test_combines_and_labels(inputs):
    tmp, files = inputs
    out = combine_gene_pvalues(files, results_dir=str(tmp))
    assert list(out.uniprot_id) == ['P1', 'P2']          # sorted by cct
    assert set(out.best_input) == {'a'}                  # 'a' has the min for both
    assert out.n_inputs.unique().tolist() == [3]
    p1 = out[out.uniprot_id == 'P1'].iloc[0]
    assert p1.cct == pytest.approx(cct([0.001, 0.004, 0.002]))
    assert p1.min_p == pytest.approx(0.001)


def test_best_nbhd_comes_from_best_input(inputs):
    """The reported neighborhood must be the one the winning input found, not a
    neighborhood from some other input or an aggregate."""
    tmp, files = inputs
    out = combine_gene_pvalues(files, results_dir=str(tmp))
    p2 = out[out.uniprot_id == 'P2'].iloc[0]
    assert p2.best_input == 'a'       # 0.40 beats 0.90 and 0.60
    assert p2.best_nbhd == 10         # a's best_aa_pos for P2, not b's 20 or c's 30


def test_n_sig_counts_inputs_below_the_threshold(inputs):
    tmp, files = inputs
    out = combine_gene_pvalues(files, results_dir=str(tmp), sig=0.003)
    assert out[out.uniprot_id == 'P1'].n_sig.iloc[0] == 2     # 0.001, 0.002
    assert out[out.uniprot_id == 'P2'].n_sig.iloc[0] == 0


def test_explicit_labels(inputs):
    tmp, files = inputs
    out = combine_gene_pvalues([f'filter{i}={f}' for i, f in enumerate(files)],
                               results_dir=str(tmp))
    assert set(out.best_input) == {'filter0'}


def test_duplicate_labels_rejected(inputs):
    tmp, files = inputs
    with pytest.raises(ValueError, match='Duplicate input label'):
        combine_gene_pvalues([files[0], files[0]], results_dir=str(tmp))


def test_refuses_an_fwer_file(tmp_path):
    """The whole point of the gene_p / fwer split, enforced rather than documented."""
    pd.DataFrame({'uniprot_id': ['P1'], 'fwer': [0.01]}).to_csv(
        tmp_path / 'f.tsv', sep='\t', index=False)
    with pytest.raises(ValueError, match='cannot be Cauchy-combined'):
        combine_gene_pvalues(['f.tsv', 'f.tsv'], results_dir=str(tmp_path))


def test_saturated_input_is_clamped_per_row(tmp_path):
    """Three strong inputs and one saturated one. Unclamped, the p = 1 input alone
    drives the combination to exactly 1 and the gene is lost; clamped, it costs the
    gene about what one uninformative input should cost it."""
    for name, p in [('a', 0.0001), ('b', 0.0001), ('c', 0.0001), ('d', 1.0)]:
        _gene_file(tmp_path / f'{name}.gene.tsv', [('P1', 'G1', p, 5)], n_sims=10000)
    files = ['a.gene.tsv', 'b.gene.tsv', 'c.gene.tsv', 'd.gene.tsv']
    out = combine_gene_pvalues(files, results_dir=str(tmp_path))
    assert out.n_clamped.iloc[0] == 1
    assert out.cct.iloc[0] == pytest.approx(cct([0.0001] * 3 + [1 - 1 / 10001]))
    assert out.cct.iloc[0] < 0.001
    # What it would have been without the clamp.
    assert cct([0.0001] * 3 + [1.0]) == pytest.approx(1.0)


def test_clamp_uses_each_input_s_own_n_sims(tmp_path):
    _gene_file(tmp_path / 'a.gene.tsv', [('P1', 'G1', 0.0001, 5)], n_sims=10000)
    _gene_file(tmp_path / 'b.gene.tsv', [('P1', 'G1', 1.0, 5)], n_sims=999)
    out = combine_gene_pvalues(['a.gene.tsv', 'b.gene.tsv'], results_dir=str(tmp_path))
    assert out.cct.iloc[0] == pytest.approx(cct([0.0001, 1 - 1 / 1000]))


def test_fixed_clamp_overrides(tmp_path):
    _gene_file(tmp_path / 'a.gene.tsv', [('P1', 'G1', 0.0001, 5)])
    _gene_file(tmp_path / 'b.gene.tsv', [('P1', 'G1', 1.0, 5)])
    out = combine_gene_pvalues(['a.gene.tsv', 'b.gene.tsv'],
                               results_dir=str(tmp_path), clamp=0.99)
    assert out.cct.iloc[0] == pytest.approx(cct([0.0001, 0.99]))


def test_across_gene_correction_applied_once(inputs):
    tmp, files = inputs
    out = combine_gene_pvalues(files, results_dir=str(tmp))
    m = len(out)
    assert (out.n_genes_tested == m).all()
    assert out.cct_bonf.iloc[0] == pytest.approx(min(out.cct.iloc[0] * m, 1.0))
    # BH is monotone and never below the raw cct.
    assert (np.diff(out.cct_bh) >= -1e-12).all()
    assert (out.cct_bh >= out.cct - 1e-12).all()


def test_gene_list_changes_only_the_across_gene_correction(inputs):
    """gene_p does not depend on the gene set, so restricting it must move cct_bonf
    and nothing else."""
    tmp, files = inputs
    everything = combine_gene_pvalues(files, results_dir=str(tmp))
    subset = combine_gene_pvalues(files, results_dir=str(tmp), gene_list=['P1'])

    a = everything[everything.uniprot_id == 'P1'].iloc[0]
    b = subset[subset.uniprot_id == 'P1'].iloc[0]
    assert a.cct == b.cct
    assert a.best_nbhd == b.best_nbhd
    assert a.cct_bonf == pytest.approx(b.cct_bonf * 2)   # m goes 2 -> 1


def test_gene_missing_from_one_input(tmp_path, caplog):
    _gene_file(tmp_path / 'a.gene.tsv',
               [('P1', 'G1', 0.01, 5), ('P2', 'G2', 0.02, 7)])
    _gene_file(tmp_path / 'b.gene.tsv', [('P1', 'G1', 0.03, 5)])
    out = combine_gene_pvalues(['a.gene.tsv', 'b.gene.tsv'], results_dir=str(tmp_path))
    assert out.set_index('uniprot_id').n_inputs.to_dict() == {'P1': 2, 'P2': 1}
    assert out[out.uniprot_id == 'P2'].cct.iloc[0] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# regressions from the code review
# ---------------------------------------------------------------------------

def test_colliding_basenames_fall_back_to_the_path(tmp_path):
    """Scans run with the default --fdr-file all write all_proteins.gene.tsv, so the
    directory is the only thing telling them apart. A glob cannot carry label=path,
    so the fallback has to be automatic."""
    for d in ('am_ge0.9', 'blosum_le0'):
        (tmp_path / d).mkdir()
        _gene_file(tmp_path / d / 'all_proteins.gene.tsv', [('P1', 'G1', 0.01, 5)])
    out = combine_gene_pvalues(['am_ge0.9/all_proteins.gene.tsv',
                                'blosum_le0/all_proteins.gene.tsv'],
                               results_dir=str(tmp_path))
    assert out.best_input.iloc[0] in {'am_ge0.9/all_proteins',
                                      'blosum_le0/all_proteins'}


def test_distinct_basenames_stay_short(inputs):
    """The fallback must not fire when it is not needed."""
    tmp, files = inputs
    out = combine_gene_pvalues(files, results_dir=str(tmp))
    assert out.best_input.iloc[0] == 'a'


def test_explicit_label_survives_a_collision(tmp_path):
    for d in ('x', 'y'):
        (tmp_path / d).mkdir()
        _gene_file(tmp_path / d / 'all_proteins.gene.tsv', [('P1', 'G1', 0.01, 5)])
    out = combine_gene_pvalues(['first=x/all_proteins.gene.tsv',
                                'second=y/all_proteins.gene.tsv'],
                               results_dir=str(tmp_path))
    assert out.best_input.iloc[0] in {'first', 'second'}


def test_the_same_file_twice_is_still_rejected(tmp_path):
    _gene_file(tmp_path / 'a.gene.tsv', [('P1', 'G1', 0.01, 5)])
    with pytest.raises(ValueError, match='Duplicate input label'):
        combine_gene_pvalues(['a.gene.tsv', 'a.gene.tsv'], results_dir=str(tmp_path))


def test_empty_inputs_raise_a_clear_error(tmp_path):
    """gene_level_pvalues legitimately writes a header-only file when no gene had a
    finite observed p; combining those must not die with KeyError: 'cct'."""
    for name in ('a', 'b'):
        pd.DataFrame(columns=['uniprot_id', 'gene_name', 'gene_p', 'n_sims',
                              'best_aa_pos', 'best_radius']).to_csv(
            tmp_path / f'{name}.gene.tsv', sep='\t', index=False)
    with pytest.raises(ValueError, match='No genes to combine'):
        combine_gene_pvalues(['a.gene.tsv', 'b.gene.tsv'], results_dir=str(tmp_path))


def test_disjoint_inputs_warn(tmp_path, caplog):
    """Every gene has n_inputs == 1, so counts.nunique() is 1 and the nunique guard
    would stay silent on the case that most needs the warning."""
    _gene_file(tmp_path / 'a.gene.tsv', [('P1', 'G1', 0.01, 5)])
    _gene_file(tmp_path / 'b.gene.tsv', [('P2', 'G2', 0.02, 7)])
    with caplog.at_level('WARNING'):
        out = combine_gene_pvalues(['a.gene.tsv', 'b.gene.tsv'],
                                   results_dir=str(tmp_path))
    assert 'missing from at least one input' in caplog.text
    assert set(out.n_inputs) == {1}


def test_gene_name_taken_from_any_input_that_has_one(tmp_path):
    """The winning input may be a hand-built file with no gene_name; the gene should
    still come back named."""
    _gene_file(tmp_path / 'named.gene.tsv', [('P1', 'G1', 0.50, 5)])
    pd.DataFrame({'uniprot_id': ['P1'], 'gene_p': [0.01], 'n_sims': [10000],
                  'best_aa_pos': [9], 'best_radius': [8.0]}).to_csv(
        tmp_path / 'bare.gene.tsv', sep='\t', index=False)
    out = combine_gene_pvalues(['named.gene.tsv', 'bare.gene.tsv'],
                               results_dir=str(tmp_path))
    assert out.best_input.iloc[0] == 'bare'      # the file with no gene_name won
    assert out.gene_name.iloc[0] == 'G1'         # named anyway
