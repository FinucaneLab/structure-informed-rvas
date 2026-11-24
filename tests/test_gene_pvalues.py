"""
Gene-level p-values: the within-gene FWER written by --gene-pvals.
"""
import os
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from empirical_fdr import compute_fdr          # noqa: E402
from gene_pvalues import gene_level_pvalues    # noqa: E402
from utils import write_dataset                # noqa: E402


def _obs(uniprot_id, pvals, radii=None):
    """Observed frame shaped like _load_all_pvalues' output for one protein."""
    n = len(pvals)
    return pd.DataFrame({
        'uniprot_id': uniprot_id,
        'aa_pos': np.arange(1, n + 1),
        'p_value': np.asarray(pvals, float),
        'radius': np.full(n, 8.0) if radii is None else np.asarray(radii, float),
    })


def _brute(obs, null):
    """The statistic, written out the slow way."""
    obs_min = min(obs)
    n_sims = len(null[0])
    k = sum(1 for s in range(n_sims)
            if min(null[j][s] for j in range(len(null))) <= obs_min)
    return (k + 1) / (n_sims + 1)


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    obs = rng.random(7)
    null = rng.random((7, 50))
    out = gene_level_pvalues(_obs('P1', obs), {'P1': null})
    assert len(out) == 1
    assert out.gene_p.iloc[0] == pytest.approx(_brute(obs, null))
    assert out.n_sims.iloc[0] == 50
    assert out.n_pos.iloc[0] == 7


def test_ties_are_counted():
    """The exact CMH is discrete: the null minimum frequently EQUALS the observed,
    and those simulations must count against the gene."""
    obs = [0.25, 0.5, 0.75]
    null = np.full((3, 10), 0.9)
    null[0, :4] = 0.25        # four simulations tie the observed minimum exactly
    out = gene_level_pvalues(_obs('P1', obs), {'P1': null})
    assert out.n_null_le_obs.iloc[0] == 4
    assert out.gene_p.iloc[0] == pytest.approx(5 / 11)

    # Counting with `<` instead would give 1/11 -- the floor -- which is the bug.
    assert out.gene_p.iloc[0] != pytest.approx(1 / 11)


def test_never_zero():
    """A gene no simulation beats still gets 1/(n+1), not 0: a 0 would send any
    downstream Cauchy combination to 0."""
    out = gene_level_pvalues(_obs('P1', [1e-9]), {'P1': np.full((1, 99), 0.5)})
    assert out.n_null_le_obs.iloc[0] == 0
    assert out.gene_p.iloc[0] == pytest.approx(1 / 100)
    assert out.gene_p_raw.iloc[0] == 0.0


def test_saturates_at_one():
    out = gene_level_pvalues(_obs('P1', [0.9]), {'P1': np.full((1, 9), 0.1)})
    assert out.gene_p.iloc[0] == 1.0


def test_best_neighborhood_is_the_argmin():
    obs = [0.4, 0.01, 0.3]
    radii = [6.0, 10.0, 8.0]
    out = gene_level_pvalues(_obs('P1', obs, radii), {'P1': np.full((3, 5), 0.5)})
    assert out.best_aa_pos.iloc[0] == 2
    assert out.best_radius.iloc[0] == 10.0
    assert out.obs_min_p.iloc[0] == pytest.approx(0.01)


def test_row_order_does_not_matter():
    """_load_all_pvalues sorts globally by p_value, so a gene's observed rows are not
    in position order and need not line up with its null rows."""
    rng = np.random.default_rng(1)
    obs = rng.random(6)
    null = rng.random((6, 40))
    straight = gene_level_pvalues(_obs('P1', obs), {'P1': null})
    shuffled = gene_level_pvalues(
        _obs('P1', obs).sort_values('p_value'), {'P1': null})
    assert straight.gene_p.iloc[0] == shuffled.gene_p.iloc[0]
    assert straight.best_aa_pos.iloc[0] == shuffled.best_aa_pos.iloc[0]


def test_independent_of_the_other_genes_in_the_run():
    """This is what lets one gene-level file be reused under several gene sets: only
    the final across-gene correction depends on the set."""
    rng = np.random.default_rng(2)
    o1, n1 = rng.random(5), rng.random((5, 30))
    o2, n2 = rng.random(4), rng.random((4, 30))

    alone = gene_level_pvalues(_obs('P1', o1), {'P1': n1})
    together = gene_level_pvalues(
        pd.concat([_obs('P1', o1), _obs('P2', o2)]), {'P1': n1, 'P2': n2})
    assert (together[together.uniprot_id == 'P1'].gene_p.iloc[0]
            == alone.gene_p.iloc[0])


def test_nan_observed_positions_are_skipped():
    out = gene_level_pvalues(_obs('P1', [np.nan, 0.2, np.nan]),
                             {'P1': np.full((3, 10), 0.5)})
    assert out.n_pos.iloc[0] == 1
    assert out.obs_min_p.iloc[0] == pytest.approx(0.2)
    assert out.best_aa_pos.iloc[0] == 2


def test_gene_with_no_finite_p_is_dropped():
    out = gene_level_pvalues(_obs('P1', [np.nan, np.nan]),
                             {'P1': np.full((2, 10), 0.5)})
    assert len(out) == 0


# ---------------------------------------------------------------------------
# end to end through compute_fdr
# ---------------------------------------------------------------------------

def _write_pval_file(path, genes):
    """genes: uniprot_id -> (observed 1-d array, null 2-d array)."""
    with h5py.File(path, 'w') as fid:
        for uid, (obs, null) in genes.items():
            n = len(obs)
            write_dataset(fid, uid, np.asarray(obs, float).reshape(-1, 1))
            write_dataset(fid, f'{uid}_null_pval', np.asarray(null, float))
            write_dataset(fid, f'{uid}_nbhd', np.column_stack(
                [np.arange(n, dtype=float), np.arange(n, dtype=float) + 1]))
            write_dataset(fid, f'{uid}_radius', np.full((n, 1), 8.0))


@pytest.fixture
def scan_dir(tmp_path):
    rng = np.random.default_rng(3)
    genes = {'P1': (rng.random(6), rng.random((6, 40))),
             'P2': (rng.random(5), rng.random((5, 40)))}
    results = tmp_path / 'results'
    results.mkdir()
    _write_pval_file(results / 'x.pvals.h5', genes)

    reference = tmp_path / 'ref'
    reference.mkdir()
    pd.DataFrame({'uniprot_id': ['P1', 'P2'],
                  'gene_name': ['GENE1', 'GENE2'],
                  'sequence': ['A' * 6, 'A' * 5]}).to_csv(
        reference / 'protein_sequence_guide.tsv', sep='\t', index=False)
    return results, reference, genes


def test_compute_fdr_writes_the_gene_file(scan_dir):
    results, reference, genes = scan_dir
    compute_fdr(str(results), 0.05, None, str(reference), 'x.pvals.h5',
                gene_file='x.gene.tsv')

    out = pd.read_csv(results / 'x.gene.tsv', sep='\t')
    assert list(out.columns[:3]) == ['uniprot_id', 'gene_name', 'gene_p']
    assert set(out.uniprot_id) == {'P1', 'P2'}
    assert set(out.gene_name) == {'GENE1', 'GENE2'}
    for uid, (obs, null) in genes.items():
        row = out[out.uniprot_id == uid].iloc[0]
        assert row.gene_p == pytest.approx(_brute(obs, null))
        assert row.best_aa_pos == int(np.argmin(obs)) + 1


def test_no_gene_file_without_the_flag(scan_dir):
    results, reference, _ = scan_dir
    compute_fdr(str(results), 0.05, None, str(reference), 'x.pvals.h5')
    assert not (results / 'x.gene.tsv').exists()


def test_position_filter_applies_to_both_sides(scan_dir):
    """--df-filter (the am99 restriction) has to mask the observed and the null
    together, or the gene is scored against centres it was not tested at."""
    results, reference, genes = scan_dir
    keep = [1, 2, 3]
    df_filter = pd.DataFrame({'uniprot_id': ['P1'] * 3, 'aa_pos': keep})
    compute_fdr(str(results), 0.05, df_filter, str(reference), 'x.pvals.h5',
                gene_file='filtered.gene.tsv')

    out = pd.read_csv(results / 'filtered.gene.tsv', sep='\t')
    assert set(out.uniprot_id) == {'P1'}
    obs, null = genes['P1']
    idx = [p - 1 for p in keep]
    assert out.gene_p.iloc[0] == pytest.approx(_brute(obs[idx], null[idx]))
    assert out.n_pos.iloc[0] == 3


def test_gene_p_unchanged_by_restricting_the_gene_set(scan_dir):
    """Dropping P2 from the run must not move P1's gene_p -- unlike `fwer`, which is
    corrected against every gene in the run and does move."""
    results, reference, _ = scan_dir
    both = compute_fdr(str(results), 0.05, None, str(reference), 'x.pvals.h5',
                       gene_file='both.gene.tsv')
    one = compute_fdr(str(results), 0.05, pd.DataFrame({'uniprot_id': ['P1']}),
                      str(reference), 'x.pvals.h5', gene_file='one.gene.tsv')

    g_both = pd.read_csv(results / 'both.gene.tsv', sep='\t')
    g_one = pd.read_csv(results / 'one.gene.tsv', sep='\t')
    assert (g_both[g_both.uniprot_id == 'P1'].gene_p.iloc[0]
            == g_one[g_one.uniprot_id == 'P1'].gene_p.iloc[0])

    # The contrast: the neighborhood-level fwer does depend on the gene set.
    f_both = both[both.uniprot_id == 'P1'].sort_values('aa_pos').fwer.to_numpy()
    f_one = one[one.uniprot_id == 'P1'].sort_values('aa_pos').fwer.to_numpy()
    assert np.any(f_both > f_one)


def test_null_searches_only_the_centres_the_observed_did():
    """A non-finite observed p drops that centre from the observed minimum, so its
    null row has to go too -- otherwise the null searched more neighborhoods than the
    observed and gene_p is biased upward."""
    obs = [np.nan, 0.5]
    null = np.array([[0.01] * 10,      # masked-out centre; would beat the observed
                     [0.90] * 10])     # the only centre actually tested
    out = gene_level_pvalues(_obs('P1', obs), {'P1': null})
    assert out.n_null_le_obs.iloc[0] == 0
    assert out.gene_p.iloc[0] == pytest.approx(1 / 11)


def test_alignment_survives_the_global_p_value_sort():
    """_load_all_pvalues sorts by p_value before this runs, so the mask has to be put
    back in aa_pos order before it can index the null rows."""
    obs_frame = _obs('P1', [np.nan, 0.5]).sort_values('p_value')
    null = np.array([[0.01] * 10, [0.90] * 10])
    out = gene_level_pvalues(obs_frame, {'P1': null})
    assert out.n_null_le_obs.iloc[0] == 0


def test_misaligned_null_falls_back_to_every_centre(caplog):
    """If the two sides cannot be aligned, say so rather than index the wrong rows."""
    with caplog.at_level('WARNING'):
        out = gene_level_pvalues(_obs('P1', [np.nan, 0.5]),
                                 {'P1': np.full((3, 10), 0.9)})
    assert 'cannot align' in caplog.text
    assert len(out) == 1
