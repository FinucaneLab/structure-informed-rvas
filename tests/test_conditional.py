"""Self-check for conditional_scan's round bookkeeping.

Run directly: python tests/test_conditional.py
Or under pytest: pytest tests/test_conditional.py

The masking primitives are covered by tests/test_conditioning.py. What is tested
here is which genes stay in each round's multiple-testing pool, because getting
that wrong shifts FDR and FWER for every gene in the analysis, not only the one
that went astray.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conditional_scan import (FDR_COLUMN, FWER_COLUMN, RESULT_COLUMNS, _carry_gene_totals,
                              _column_for, _drop_failed_rescans, _genes_passing_gate,
                              _round_path)


def _filter(*uids):
    return pd.DataFrame({'uniprot_id': list(uids)})


# ---------------------------------------------------------------------------
# A gene that was not re-scanned must leave the pool
# ---------------------------------------------------------------------------

def test_a_failed_rescan_leaves_the_round_pool():
    """A round copies the previous round's p-value file and overwrites only the genes
    it re-scans, so a skipped gene keeps its UNCONDITIONED p-values there. Those are
    small by construction -- they are why it was selected for conditioning. Leaving it
    in means this round's corrections are computed against a pool containing a signal
    the analysis believes it masked, deflating FWER and inflating the false-discovery
    count for every other gene."""
    active, pool = _drop_failed_rescans(
        ['P1', 'P2', 'P3'], scanned={'P1', 'P3'},
        round_filter=_filter('P1', 'P2', 'P3'), round_idx=1)
    assert active == ['P1', 'P3']
    assert pool.uniprot_id.tolist() == ['P1', 'P3'], 'P2 must leave the pool too'


def test_nothing_changes_when_every_rescan_succeeded():
    pool = _filter('P1', 'P2')
    active, out = _drop_failed_rescans(['P1', 'P2'], {'P1', 'P2'}, pool, 1)
    assert active == ['P1', 'P2'] and out is pool


def test_the_pool_keeps_a_position_level_filter():
    """When --df-filter carries aa_pos the filter is position-level. Dropping that
    column would silently correct later rounds over every residue in the protein
    instead of the filtered subset."""
    pool = pd.DataFrame({'uniprot_id': ['P1', 'P1', 'P2'], 'aa_pos': [10, 20, 30]})
    _, out = _drop_failed_rescans(['P1', 'P2'], {'P1'}, pool, 1)
    assert list(out.columns) == ['uniprot_id', 'aa_pos']
    assert out.aa_pos.tolist() == [10, 20], 'P1 keeps both of its positions'


# ---------------------------------------------------------------------------
# Per-gene totals describe the round that produced them
# ---------------------------------------------------------------------------

def test_totals_follow_the_rounds_not_round_zero():
    """A gene re-scanned in round k that then fails the gene gate leaves `active` but
    stays in the round filter, so it reappears in round k+1 carrying its round-k
    masked p-values. Falling back to round 0's unmasked totals there would print a
    40-de-novo total beside p-values computed against the 10 that survived, and the
    two round tables would disagree about the same gene."""
    results = pd.DataFrame({'uniprot_id': ['P1'], 'aa_pos': [1]})
    round0 = pd.DataFrame({'uniprot_id': ['P1'], 'n_denovo_gene': [40],
                           'mu_gene': [1.0], 'mu_coverage': [0.9]})
    # Round 1 re-scanned P1 and left it with 10.
    r1 = _carry_gene_totals(results, {'P1': (10, 0.25, 0.9)}, round0)
    assert r1.set_index('uniprot_id').loc['P1', 'n_denovo_gene'] == 10

    # Round 2 does not re-scan P1. Carrying round 1's values keeps the table honest;
    # carrying round 0's would restore the 40.
    known = r1[['uniprot_id', 'n_denovo_gene', 'mu_gene', 'mu_coverage']]
    r2 = _carry_gene_totals(results, {}, known)
    assert r2.set_index('uniprot_id').loc['P1', 'n_denovo_gene'] == 10, \
        'round 2 must not revert to the unconditioned total'

    # The above tests the helper. The wiring -- that the round loop passes the
    # running totals rather than round 0's -- is one argument inside a loop that
    # needs a full scan to reach, so it is pinned by name instead.
    import inspect, conditional_scan
    body = inspect.getsource(conditional_scan.conditional_scan)
    assert '_carry_gene_totals(df_results, round_totals, known_totals)' in body
    assert 'round0_results)' not in body, \
        'the round loop must not fall back to round 0 totals'


def test_each_round_reports_its_own_gene_totals():
    """Conditioning masks residues out of original_case and original_mu, and Test B's
    null uses the masked totals. A gene entering with 40 de novos whose mask took 30
    must not report n_denovo_gene = 40 beside p-values computed against 10."""
    results = pd.DataFrame({'uniprot_id': ['P1', 'P2'], 'aa_pos': [1, 1]})
    round0 = pd.DataFrame({'uniprot_id': ['P1', 'P2'],
                           'n_denovo_gene': [40, 7], 'mu_gene': [1.0, 2.0],
                           'mu_coverage': [0.9, 0.8]})
    # Only P1 was re-scanned this round.
    out = _carry_gene_totals(results, {'P1': (10, 0.25, 0.9)}, round0)
    by = out.set_index('uniprot_id')
    assert by.loc['P1', 'n_denovo_gene'] == 10, "the re-scanned gene reports this round"
    # P2 was not touched, so round 0's values are exactly right for it -- and leaving
    # it NaN would make the round table ragged against round 0.
    assert by.loc['P2', 'n_denovo_gene'] == 7
    assert not out[['n_denovo_gene', 'mu_gene', 'mu_coverage']].isna().any().any()


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------

def test_the_gene_gate_reads_the_named_correction():
    df = pd.DataFrame({'uniprot_id': ['P1', 'P2', 'P3'],
                       'fdr': [0.01, 0.40, 0.01],
                       'fwer': [0.40, 0.01, 0.01]})
    assert _genes_passing_gate(df, ['P1', 'P2', 'P3'], 'fwer', 0.05) == ['P2', 'P3']
    assert _genes_passing_gate(df, ['P1', 'P2', 'P3'], 'fdr', 0.05) == ['P1', 'P3']
    # Restricted to the genes still active, never reviving one that already left.
    assert _genes_passing_gate(df, ['P1'], 'fdr', 0.05) == ['P1']


def test_a_criterion_names_a_correction_not_a_mode_specific_column():
    """Both nulls write their corrections under the same two names, so a criterion is
    just a choice between them -- the member gate and the gene gate make the same
    choice twice and cannot collide into a table with no FDR column."""
    assert _column_for('fdr') == FDR_COLUMN
    assert _column_for('fwer') == FWER_COLUMN
    assert FDR_COLUMN == 'fdr' and FWER_COLUMN == 'fwer'


def test_result_columns_differ_only_in_the_p_value_and_radius():
    """A mutation-rate table has no p_value column, and each test picks its own best
    radius; Test B is the test of interest, so its radius defines the neighborhood."""
    std, mu = RESULT_COLUMNS[False], RESULT_COLUMNS[True]
    assert std.p == 'p_value' and std.radius == ('radius',)
    assert mu.p == 'p_binomial'
    assert mu.radius[0] == 'radius_binomial', 'Test B first'


# ---------------------------------------------------------------------------
# Round files
# ---------------------------------------------------------------------------

def test_round_paths_are_derived_from_the_stem():
    assert _round_path('all.fdr.tsv', 1) == 'all.fdr.round1.tsv'
    assert _round_path('all.pvals.h5', 2) == 'all.pvals.round2.h5'
    # A multi-part extension keeps its stem: splitext takes only the last suffix.
    assert _round_path('all.fdr.tsv', 3) == 'all.fdr.round3.tsv'
    # Round 0 is not routed through here at all -- it writes to the plain --fdr-file
    # and --pval-file, so a conditional run's round 0 is byte-for-byte a normal run.
    import inspect, conditional_scan
    body = inspect.getsource(conditional_scan.conditional_scan)
    assert '_round_path(fdr_file, 0)' not in body and '_round_path(pval_file, 0)' not in body


if __name__ == '__main__':
    n_run = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            print(f'  {name}')
            fn()
            n_run += 1
    assert n_run > 0, 'the test_* scan matched nothing'
    print(f'ok ({n_run} tests)')
