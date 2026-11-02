"""Self-check for components.py: the clustering rule and the per-component table.

Run directly: python tests/test_components.py
Or under pytest: pytest tests/test_components.py

Adjacency matrices are built by hand rather than loaded from a structure, so the
clustering rule is tested on its own and the tests need no reference data.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components import (build_components, component_table, enriched_mask,
                        parse_centers_detail, top_component)


def banded_adjacency(n_res, half_width):
    """Residue i is adjacent to those within half_width along the sequence."""
    idx = np.arange(n_res)
    return (np.abs(idx[:, None] - idx[None, :]) <= half_width).astype(int)


def _results(aa_pos, p, fdr, fwer, radius=15.0, **extra):
    df = pd.DataFrame({'uniprot_id': 'P1', 'gene_name': 'GENE', 'aa_pos': aa_pos,
                       'p_value': p, 'fdr': fdr, 'fwer': fwer, 'radius': radius})
    for k, v in extra.items():
        df[k] = v
    return df


def _mut(aa_pos, ac_case, ac_control=None, n_res=40):
    df = pd.DataFrame({'aa_pos': np.arange(1, n_res + 1), 'ac_case': 0, 'ac_control': 0})
    df.loc[df.aa_pos.isin(aa_pos), 'ac_case'] = ac_case
    if ac_control is not None:
        df['ac_control'] = ac_control
    return df


# ---------------------------------------------------------------------------
# The clustering rule: a shared CASE-MUTATED residue, not mere overlap
# ---------------------------------------------------------------------------

def test_shared_mutation_connects():
    adj = {15.0: banded_adjacency(40, 3)}
    # Centres 10 and 12 both reach residue 11, which is mutated.
    labels, _, shared = build_components([(10, 15.0), (12, 15.0)], adj, {11})
    assert labels[0] == labels[1]
    assert shared[0] == shared[1] == {11}


def test_overlap_without_a_shared_mutation_does_not_connect():
    """Plain residue overlap would chain distinct sites across a domain. The rule is
    about the variants: two windows join only when they report the same ones."""
    adj = {15.0: banded_adjacency(40, 3)}
    # 10 and 12 overlap heavily, but the only mutations are at 7 and 15, one in each.
    labels, _, shared = build_components([(10, 15.0), (12, 15.0)], adj, {7, 15})
    assert shared[0] == {7} and shared[1] == {15}
    assert labels[0] != labels[1]


def test_transitive_through_a_middle_centre():
    """A and C join through B without sharing a mutation themselves -- a run of
    overlapping windows walking along one cluster is one signal."""
    adj = {15.0: banded_adjacency(60, 3)}
    labels, _, _ = build_components([(10, 15.0), (14, 15.0), (18, 15.0)], adj, {11, 16})
    assert labels[0] == labels[1] == labels[2]


def test_each_centre_uses_its_own_radius():
    """Under --neighborhood-radius multiple-* every position picks its own."""
    adj = {3.0: banded_adjacency(40, 1), 15.0: banded_adjacency(40, 6)}
    # At radius 3 centre 20 reaches only 19..21; at 15 it reaches 14..26.
    labels_narrow, nbhd_narrow, _ = build_components([(20, 3.0), (26, 3.0)], adj, {25})
    labels_wide, nbhd_wide, _ = build_components([(20, 15.0), (26, 15.0)], adj, {25})
    assert len(nbhd_narrow[0]) < len(nbhd_wide[0])
    assert labels_narrow[0] != labels_narrow[1]   # narrow: 20 misses residue 25
    assert labels_wide[0] == labels_wide[1]       # wide: both reach it


# ---------------------------------------------------------------------------
# The per-component row
# ---------------------------------------------------------------------------

def test_component_table_contents():
    adj = {15.0: banded_adjacency(40, 3)}
    centers = [(10, 15.0), (12, 15.0)]
    df_results = _results([10, 12], p=[1e-6, 1e-3], fdr=[0.01, 0.04], fwer=[0.02, 0.05])
    df_mut = _mut([11, 13], ac_case=2, ac_control=1)

    t = component_table(df_results, 'P1', 'GENE', centers, adj, df_mut,
                        p_col='p_value', mu_mode=False)
    assert len(t) == 1
    row = t.iloc[0]
    assert row.n_centers == 2
    # The lead is the best-scoring centre by p_col.
    assert row.lead_aa_pos == 10 and row.lead_p == 1e-6
    # best_* are minima over the component's OWN members, not values read off the lead.
    assert row.best_fdr == 0.01 and row.best_fwer == 0.02
    assert row.centers_pymol == '10+12'
    assert row.centers_detail == '10:15,12:15'
    assert row.n_case_alleles == 4 and row.n_case_residues == 2
    assert row.seq_span_aa == 2


def test_best_values_come_from_members_not_the_lead():
    """The lead is chosen by p_col; under the mutation-rate null fdr/fwer are maxima
    over both tests and so are not monotone in it. Ranking on the lead's value would
    mask the wrong component."""
    adj = {15.0: banded_adjacency(40, 3)}
    # Centre 10 has the best p but the WORSE fdr/fwer.
    df_results = _results([10, 12], p=[1e-6, 1e-3], fdr=[0.04, 0.01], fwer=[0.05, 0.02])
    t = component_table(df_results, 'P1', 'GENE', [(10, 15.0), (12, 15.0)], adj,
                        _mut([11], ac_case=2, ac_control=1), p_col='p_value', mu_mode=False)
    row = t.iloc[0]
    assert row.lead_aa_pos == 10
    assert row.lead_fdr == 0.04        # the lead's own value
    assert row.best_fdr == 0.01        # the component's minimum
    assert row.best_gate == 0.02


def test_caseless_neighborhoods_are_counted_not_clustered():
    """A neighborhood with no case variant cannot share one, so it arrives as a
    component of one with nothing in it. Under a two-sided scan these are depletion
    calls; they are counted and dropped, not reported as findings."""
    adj = {15.0: banded_adjacency(40, 2)}
    df_results = _results([10, 30], p=[1e-6, 1e-6], fdr=[0.01, 0.01], fwer=[0.02, 0.02])
    df_mut = _mut([11], ac_case=2, ac_control=1)     # nothing near centre 30
    t = component_table(df_results, 'P1', 'GENE', [(10, 15.0), (30, 15.0)], adj,
                        df_mut, p_col='p_value', mu_mode=False)
    assert len(t) == 1 and t.iloc[0].lead_aa_pos == 10
    assert t.attrs['n_caseless'] == 1


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

def test_enriched_mask_uses_the_test_not_the_ratio_column():
    """`ratio` is regularized, so in a gene where controls outnumber cases it sits
    below 1 for neighborhoods the Fisher test considers enriched."""
    g = pd.DataFrame({'nbhd_case': [10, 1], 'nbhd_control': [10, 30]})
    df_mut = pd.DataFrame({'ac_case': [20], 'ac_control': [60]})   # 25% of alleles are cases
    mask = enriched_mask(g, df_mut, mu_mode=False)
    # Row 0: 10 of 20 observed vs 5 expected -> enriched. Row 1: 1 of 31 vs ~7.75 -> not.
    assert list(mask) == [True, False]


def test_enriched_mask_mu_mode_prefers_the_matched_column():
    """obs_exp_<test> is computed inside one test's own table, so observed and
    expected are at the same radius. nbhd_case comes only from the Poisson side."""
    g = pd.DataFrame({'obs_exp_binomial': [2.0, 0.5], 'nbhd_case': [1, 99],
                      'nbhd_expected': [99, 1]})
    assert list(enriched_mask(g, None, mu_mode=True)) == [True, False]


def test_enriched_mask_returns_none_when_it_cannot_tell():
    assert enriched_mask(pd.DataFrame({'aa_pos': [1]}), None, mu_mode=True) is None


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def test_top_component_ranks_on_best_gate():
    df = pd.DataFrame({'uniprot_id': ['P1', 'P1'], 'best_gate': [0.04, 0.01],
                       'lead_p': [1e-9, 1e-3], 'lead_fwer': [0.04, 0.01]})
    assert top_component(df, 'P1').best_gate == 0.01     # not the better lead_p
    assert top_component(df, 'ABSENT') is None
    # An empty frame has no uniprot_id column at all, which an unguarded lookup
    # turns into AttributeError after a full exome scan has already been paid for.
    assert top_component(pd.DataFrame(), 'P1') is None


def test_parse_centers_detail_and_its_fallbacks():
    Row = lambda **kw: pd.Series(kw)
    assert parse_centers_detail(Row(centers_detail='10:15,20:9')) == [(10, 15.0), (20, 9.0)]
    # No centers_detail: fall back to centers_pymol plus the component's one radius.
    assert parse_centers_detail(Row(centers_detail=None, centers_pymol='10+12',
                                    radius=9.0)) == [(10, 9.0), (12, 9.0)]
    # Neither: the lead alone, so a hand-written file still draws something.
    assert parse_centers_detail(Row(centers_detail=None, centers_pymol=None,
                                    radius=None, lead_aa_pos=7)) == [(7, 15.0)]


def test_member_mask_decides_membership_not_a_single_cutoff():
    """The two corrections are not nested: FWER significance does not imply FDR
    significance at the same cutoff. On a real CAMK2A run the only FWER-significant
    neighborhood had fwer 0.045 against fdr 0.070, so an FDR-only member rule admits
    the gene on that neighborhood and then leaves it out, giving an empty table for a
    gene that qualified. run.py unions the gate into the mask for this reason.
    """
    df_results = _results([10, 12], p=[1e-6, 1e-3],
                          fdr=[0.070, 0.9],      # neither clears fdr < 0.05
                          fwer=[0.045, 0.9])     # the first clears fwer < 0.05
    fdr_only = df_results['fdr'] < 0.05
    assert not fdr_only.any(), 'the fixture must not clear the FDR gate'
    unioned = fdr_only | (df_results['fwer'] < 0.05)
    assert list(unioned) == [True, False]

    adj = {15.0: banded_adjacency(40, 3)}
    captured = {}

    def read_mut(uniprot_id):
        return _mut([11], ac_case=2, ac_control=1)

    from components import components_for_genes
    import components as comp_mod

    def fake_adjacency(guide, pdb_dir, pae_dir, uid, radius, pae):
        captured['radius'] = radius
        return adj[15.0]

    real = comp_mod.get_adjacency_matrix
    comp_mod.get_adjacency_matrix = fake_adjacency
    try:
        out = components_for_genes(df_results, ['P1'], 'ref', 15.0, read_mut,
                                   'p_value', unioned, ('radius',), False)
    finally:
        comp_mod.get_adjacency_matrix = real

    assert len(out) == 1, 'the gate-passing neighborhood must be clustered'
    assert out.iloc[0].lead_aa_pos == 10


def test_no_centers_is_not_an_error():
    labels, nbhd, shared = build_components([], {}, set())
    assert len(labels) == 0 and nbhd == [] and shared == []


if __name__ == '__main__':
    n_run = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            print(f'  {name}')
            fn()
            n_run += 1
    assert n_run > 0, 'the test_* scan matched nothing'
    print(f'ok ({n_run} tests)')
