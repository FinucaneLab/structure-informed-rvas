"""Self-check for conditioning: masking a called region out before a re-scan.

Run directly: python tests/test_conditioning.py
Or under pytest: pytest tests/test_conditioning.py

Adjacency matrices are built by hand, so nothing here needs reference data.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mutation_rate_test import apply_conditioning
from scan_test import (DEFAULT_MIN_RETAINED, _blank_counts, _neutralize,
                       neutralized_positions)
from utils import (check_keep_mask, conditioning_keep_mask,
                   get_case_control_ac_matrix, neighborhood_retention,
                   residues_in_nbhd_idx)


def banded(n_res, half_width):
    i = np.arange(n_res)
    return (np.abs(i[:, None] - i[None, :]) <= half_width).astype(int)


# ---------------------------------------------------------------------------
# The mask itself
# ---------------------------------------------------------------------------

def test_keep_mask_length_mismatch_is_refused():
    """The mask is positional -- entry i is residue i+1 -- so a wrong length would
    silently shift every residue rather than failing."""
    assert check_keep_mask(None, 10) is None
    assert check_keep_mask(np.ones(10, bool), 10).shape == (10,)
    try:
        check_keep_mask(np.ones(9, bool), 10, 'P1')
        raise AssertionError('a length mismatch must be refused')
    except ValueError as e:
        assert 'positional' in str(e)


def test_residues_in_nbhd_idx_is_one_based_in_zero_based_out():
    adj = banded(20, 2)
    idx = residues_in_nbhd_idx(adj, 10)          # centre is 1-based
    assert (idx + 1).tolist() == [8, 9, 10, 11, 12]
    for bad in (0, 21):
        try:
            residues_in_nbhd_idx(adj, bad)
            raise AssertionError(f'aa_pos {bad} is outside the structure')
        except ValueError:
            pass


def test_explicit_residues_mask_without_a_structure():
    """Conditional analysis masks a whole component, whose residue set is already
    known, so re-deriving it from centres and radii would rebuild the same union."""
    keep, n_res, removed = conditioning_keep_mask(
        None, None, None, 'P1', None, 15.0, residues=[5, 6, 7], n_res=20)
    assert n_res == 20
    assert np.flatnonzero(~keep).tolist() == [4, 5, 6]      # 1-based 5,6,7
    assert removed == {'residues': 3}


# ---------------------------------------------------------------------------
# Counts and opportunity move together
# ---------------------------------------------------------------------------

def test_apply_conditioning_clears_counts_and_opportunity():
    """Zeroing only the counts would leave both nulls still expecting de novos at
    residues whose variants were just removed: every neighborhood overlapping the
    conditioned region would look depleted, and the neighborhoods beside it --
    exactly where a second independent signal would sit -- would lose power."""
    x = np.array([[3], [1], [4], [2]])
    m = np.array([[0.1], [0.2], [0.3], [0.4]])
    keep = np.array([True, False, False, True])
    xc, mc = apply_conditioning(x, m, keep)
    assert xc.ravel().tolist() == [3, 0, 0, 2]
    assert mc.ravel().tolist() == [0.1, 0.0, 0.0, 0.4]
    # Nothing is dropped: the arrays keep their length and their positions.
    assert xc.shape == x.shape and mc.shape == m.shape


def test_apply_conditioning_is_a_no_op_without_a_mask():
    x = np.array([[1], [2]])
    m = np.array([[0.5], [0.5]])
    xc, mc = apply_conditioning(x, m, None)
    assert xc is x and mc is m


def test_case_control_masks_before_the_null_draw():
    """Zeroing after the draw would leave the null redistributing alleles into
    masked residues -- the case/control analogue of masking counts but not
    opportunity."""
    df = pd.DataFrame({'aa_pos': [1, 2, 3, 4], 'ac_case': [5, 5, 5, 5],
                       'ac_control': [5, 5, 5, 5]})
    keep = np.array([True, False, False, True])
    case_m, ctrl_m = get_case_control_ac_matrix(df, 4, 20, seed=0, keep=keep)
    assert case_m[:, 0].astype(int).tolist() == [5, 0, 0, 5]
    assert int(case_m[:, 0].sum()) == 10           # totals see only what is left
    assert (case_m[~keep, 1:] == 0).all(), 'the null must never land on a masked residue'
    assert (ctrl_m[~keep, 1:] == 0).all()


# ---------------------------------------------------------------------------
# Which centres stop being tested
# ---------------------------------------------------------------------------

def test_a_kept_residue_can_still_lose_its_neighborhood():
    """What is left of a gutted neighborhood is a different neighborhood from the one
    the scan reported, so testing it would change what the p-value refers to."""
    adj = banded(8, 1)                       # each neighborhood is 3 residues
    keep = np.ones(8, bool)
    keep[[0, 2]] = False                     # residue 1 survives, its neighbours do not
    ret = neighborhood_retention(adj, keep)
    dead = neutralized_positions(adj, keep, DEFAULT_MIN_RETAINED)
    assert keep[1] and dead[1], 'residue 1 is kept but only 1/3 of its neighborhood is'
    assert abs(ret[1] - 1 / 3) < 1e-12
    assert not dead[4], 'an untouched neighborhood is still tested'


def test_masked_residues_are_always_dead_even_at_min_retained_zero():
    """A min_retained of 0 must still not test a neighborhood with nothing
    left in it: under the mutation-rate null its expected count is 0, which would
    hand any stray count an arbitrarily small p-value."""
    adj = np.eye(4, dtype=int)               # each residue is its own neighborhood
    keep = np.array([True, False, True, False])
    dead = neutralized_positions(adj, keep, 0.0)
    assert dead.tolist() == [False, True, False, True]


def test_neutralize_sets_observed_and_null_alike_and_keeps_every_row():
    """Rows are never removed: the row index carries aa_pos, and empirical_fdr
    rebuilds positions from row order."""
    pm = np.full((5, 4), 0.001)
    dead = np.array([False, True, False, True, False])
    out = _neutralize(pm, dead)
    assert out.shape == pm.shape
    assert (out[dead] == 1.0).all()           # observed column and every null column
    assert (out[~dead] == 0.001).all()
    assert (pm == 0.001).all(), 'the input must not be modified in place'


def test_neutralize_without_a_mask_is_a_no_op():
    pm = np.full((3, 2), 0.5)
    assert _neutralize(pm, None) is pm
    assert _neutralize(pm, np.zeros(3, bool)) is pm


def test_dead_centres_report_nothing_in_every_column():
    """A dead centre's p-value is 1.0, but its counts would otherwise describe the
    SURVIVING fragment of a neighborhood nobody tested. That fragment can outrank
    the hits that were tested -- a few de novos beside a constrained residue give a
    high obs/exp -- so anyone sorting the table by effect size surfaces untested rows
    clustered exactly where the conditioning happened."""
    nbhd = np.array([6.0, 6.0, 6.0, 2.0])
    exp = np.array([0.5, 0.5, 2.0, 2.0])
    dead = np.array([True, True, False, False])

    before = (nbhd + 1) / (exp + 1)
    assert before[0] > before[2], 'the fixture must have a dead row outranking a live one'

    nb = _blank_counts(nbhd, dead)
    eb = _blank_counts(exp, dead)
    after = (nb + 1) / (eb + 1)
    assert list(nb) == [0.0, 0.0, 6.0, 2.0]
    assert list(eb) == [0.0, 0.0, 2.0, 2.0]
    # (0 + 1) / (0 + 1): a dead row reads as exactly neutral, not as a strong hit.
    assert after[0] == 1.0 and after[1] == 1.0
    assert after[2] == before[2] and after[3] == before[3], 'live rows untouched'


def test_blank_counts_copies_and_no_ops():
    arr = np.array([[3.0, 4.0], [5.0, 6.0]])
    assert _blank_counts(arr, None) is arr
    assert _blank_counts(arr, np.zeros(2, bool)) is arr
    out = _blank_counts(arr, np.array([True, False]))
    assert out.tolist() == [[0.0, 0.0], [5.0, 6.0]]
    assert arr.tolist() == [[3.0, 4.0], [5.0, 6.0]], 'input must not be modified'


# ---------------------------------------------------------------------------
# The gates that run around conditioning
# ---------------------------------------------------------------------------

def test_allele_gate_counts_what_conditioning_left():
    """The old code dropped the conditioned rows before the gate, so it was
    post-conditioning for free. Masking happens later, inside compute_all_pvals, so
    the gate has to apply the mask itself -- otherwise a gene whose case alleles all
    sit inside the masked region is scanned with n_case = 0 and read_p_values divides
    by that count."""
    import scan_test as st

    # Every case allele sits inside the region about to be masked; the control
    # alleles sit outside it. So after conditioning ONLY the case clause fails --
    # if the gate read the unmasked frame, both clauses would pass and the gene
    # would be scanned with n_case = 0.
    df = pd.DataFrame({'uniprot_id': ['P1'] * 4, 'aa_pos': [10, 11, 12, 40],
                       'ac_case': [2, 2, 2, 0], 'ac_control': [0, 0, 0, 10]})
    assert df.ac_case.sum() >= 5 and df.ac_control.sum() >= 5, \
        'the gene must clear both floors unconditioned'

    # The mirror image, so BOTH clauses of the gate are covered: here the control
    # alleles are the ones inside the mask.
    df_ctrl = df.rename(columns={'ac_case': 'ac_control', 'ac_control': 'ac_case'})

    for frame, which in [(df, 'case'), (df_ctrl, 'control')]:
        calls = []
        real_one, real_mask = st.scan_test_one_protein, st.build_conditioning_mask
        st.scan_test_one_protein = lambda *a, **k: calls.append(a)
        # Mask residues 1..13, which is where the depleted side's alleles sit.
        st.build_conditioning_mask = lambda *a, **k: (
            np.array([False] * 13 + [True] * 37), None, {10: 13})
        try:
            _, scanned = st._process_proteins_batch(
                frame, ['P1'], '/tmp', 15, 15, '/tmp', 10, {'P1': [(10, 15.0)]},
                'x.h5', seed=1, min_alleles=5)
        finally:
            st.scan_test_one_protein, st.build_conditioning_mask = real_one, real_mask

        assert calls == [], f'conditioning emptied the {which} side; must not scan'
        assert scanned == set()


def test_duplicate_centers_are_counted_once_each():
    """--remove-nbhd 100,100 used to record k on the first pass and overwrite it with
    0 on the second, logging "removes 0 residues" while k were masked."""
    import utils as u
    adj = banded(20, 2)
    real = u.get_adjacency_matrix
    u.get_adjacency_matrix = lambda *a, **k: adj
    try:
        keep, _, removed = conditioning_keep_mask(
            'g', 'p', 'pa', 'P1', [(10, 15.0), (10, 15.0)], 15.0)
    finally:
        u.get_adjacency_matrix = real
    assert sum(removed.values()) == int((~keep).sum()), 'the log must match reality'
    assert removed == {10: 5}


def test_fdr_filter_names_the_genes_actually_scanned():
    """A skipped gene keeps its PREVIOUS datasets in the p-value file. Filtering FDR
    on the requested list would load those stale, unconditioned values and report a
    conditioned-out hit at full significance, as if the re-scan had confirmed it.
    Conditioning makes skips common, so this is the normal path, not a corner."""
    import scan_test as st
    import inspect

    src = inspect.getsource(st.scan_test)
    assert 'sorted(scanned)' in src, 'the default FDR filter must come from `scanned`'
    assert "'uniprot_id': uniprot_id_list" not in src, \
        'filtering on the requested list reports stale p-values for skipped genes'

    # And the set really does exclude a gene the batch skipped.
    df = pd.DataFrame({'uniprot_id': ['P1'] * 2 + ['P2'] * 2,
                       'aa_pos': [10, 11, 10, 11],
                       'ac_case': [9, 9, 9, 9], 'ac_control': [9, 9, 9, 9]})
    real_one, real_mask = st.scan_test_one_protein, st.build_conditioning_mask
    st.scan_test_one_protein = lambda *a, **k: None
    st.build_conditioning_mask = lambda *a, **k: (
        np.array([False] * 13 + [True] * 37), None, {10: 13})   # empties P1 only
    try:
        _, scanned = st._process_proteins_batch(
            df, ['P1', 'P2'], '/tmp', 15, 15, '/tmp', 10, {'P1': [(10, 15.0)]},
            'x.h5', seed=1, min_alleles=5)
    finally:
        st.scan_test_one_protein, st.build_conditioning_mask = real_one, real_mask
    assert scanned == {'P2'}, 'the conditioned-away gene must not be reported'


def test_mu_path_rechecks_min_denovo_after_conditioning():
    """_select_genes applies --min-denovo to the unconditioned counts. Without a
    re-check, conditioning out the cluster holding all of a gene's de novos leaves
    n_g = 0, a full table of p = 1.0 rows is written, and those rows enter the FDR
    and FWER pools as tested hypotheses."""
    import mutation_rate_test as mr

    # All five de novos sit in residues 1..13, which conditioning masks.
    df = pd.DataFrame({'uniprot_id': ['P1'] * 3, 'aa_pos': [10, 11, 12],
                       'ac_case': [2, 2, 1]})
    assert df.ac_case.sum() >= 5, 'the gene must clear --min-denovo unconditioned'

    calls = []
    real_compute, real_write, real_mask = (
        mr.compute_all_pvals_mu, mr.write_df_pvals_mu, mr.build_conditioning_mask)
    mr.compute_all_pvals_mu = lambda *a, **k: calls.append(a) or (None, None, None, 0.0)
    mr.write_df_pvals_mu = lambda *a, **k: None
    mr.build_conditioning_mask = lambda *a, **k: (
        np.array([False] * 13 + [True] * 37), None, {10: 13})
    try:
        totals = mr._process_proteins_batch_mu(
            df, ['P1'], '/tmp', 15, 15, '/tmp', 10, 'x.h5', 1.0, {},
            conditioning={'P1': [(10, 15.0)]}, min_denovo=5)
    finally:
        (mr.compute_all_pvals_mu, mr.write_df_pvals_mu,
         mr.build_conditioning_mask) = real_compute, real_write, real_mask

    assert calls == [], 'a gene with no de novos left must not be scanned'
    assert totals == {}, 'and must not appear in the gene totals'


def test_explicit_df_filter_is_intersected_not_substituted():
    """A supplied --df-filter taken whole walks straight past the `scanned` set, so a
    gene conditioning skipped is reported from its previous, unconditioned datasets --
    the same stale-p-value failure, through the other branch."""
    import scan_test as st
    import inspect
    src = inspect.getsource(st.scan_test)
    assert 'df_fdr_filter.uniprot_id.isin(scanned)' in src, \
        'an explicit filter must be intersected with the genes actually written'

    # The intersection itself: a filter naming a skipped gene must not carry it.
    supplied = pd.DataFrame({'uniprot_id': ['P1', 'P2']})
    scanned = {'P2'}
    out = supplied[supplied.uniprot_id.isin(scanned)]
    assert out.uniprot_id.tolist() == ['P2']


def test_a_gene_with_nothing_left_to_test_is_refused():
    """Writing a table of p = 1.0 for every residue puts n_res rows into the FDR and
    FWER pools as tested hypotheses, inflating the burden for the genes that do still
    have signal -- and with every centre dead, n_case_total is 0 and read_p_values
    divides by it."""
    import scan_test as st

    adj = np.eye(6, dtype=int)          # each residue is its own neighborhood
    keep = np.zeros(6, bool)            # nothing survives
    assert st.neutralized_positions(adj, keep, DEFAULT_MIN_RETAINED).all()

    df = pd.DataFrame({'aa_pos': [1, 2, 3], 'ac_case': [3, 3, 3],
                       'ac_control': [3, 3, 3]})
    real = st.get_adjacency_matrix
    st.get_adjacency_matrix = lambda *a, **k: adj
    try:
        st.compute_all_pvals(df, 'g', 'p', 'pa', 'P1', n_sims=3, radius=15,
                             pae_cutoff=15, seed=0, keep_residues=keep)
        raise AssertionError('a gene with no testable neighborhood must be refused')
    except ValueError as e:
        assert 'no testable neighborhood' in str(e), e
    finally:
        st.get_adjacency_matrix = real


if __name__ == '__main__':
    n_run = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            print(f'  {name}')
            fn()
            n_run += 1
    assert n_run > 0, 'the test_* scan matched nothing'
    print(f'ok ({n_run} tests)')
