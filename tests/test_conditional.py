"""
Correctness tests for conditional 3DNT.

Two things can go wrong here and both are silent.

The first is indexing. Every per-residue array in this codebase is positional -- row i
is residue i + 1 -- and read_p_values does not store aa_pos at all, it rebuilds it from
row order. A mask that is off by one, or that drops rows, would shift every position in
the output without raising anything. Tests 1-3 pin that down.

The second is the mutation-rate null. Masking observed counts without masking the
mutational opportunity leaves the null expecting de novos where the variants were just
removed. Tests 4-5 show what that costs and that the code does not do it.

Run: python tests/test_conditional.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (check_keep_mask, residues_in_nbhd_idx, neighborhood_retention,
                   best_radius_for_hit, write_dataset)
from scan_test import (neutralized_positions, _neutralize, get_case_control_ac_matrix,
                       get_ac_per_residue)
from mutation_rate_test import (apply_conditioning, _pvals_for_radius,
                                simulate_poisson_null, simulate_multinomial_null)

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def banded_adjacency(n_res, half_width):
    """A 1-D stand-in for a structure: residue i is adjacent to those within half_width."""
    idx = np.arange(n_res)
    return (np.abs(idx[:, None] - idx[None, :]) <= half_width).astype(int)


def keep_mask_from_centers(adj, centers):
    """The mask conditioning_keep_mask builds, without needing PDB files on disk."""
    keep = np.ones(adj.shape[0], dtype=bool)
    for aa_pos in centers:
        keep[residues_in_nbhd_idx(adj, aa_pos)] = False
    return keep


# ---------------------------------------------------------------------------
def test_indexing_contract():
    """aa_pos is one-based, array rows are zero-based, and nothing may shift."""
    print('\n1. Indexing contract (row i is residue i+1)')
    n_res, hw = 60, 4
    adj = banded_adjacency(n_res, hw)

    # residues_in_nbhd_idx returns ZERO-based indices; +1 recovers aa_pos.
    idx = residues_in_nbhd_idx(adj, 30)
    check('neighborhood of aa_pos 30 is residues 26..34',
          set(idx + 1) == set(range(30 - hw, 30 + hw + 1)),
          f'got {idx.min() + 1}..{idx.max() + 1}')

    # The edges are where an off-by-one shows up.
    check('neighborhood of aa_pos 1 is clipped at the N-terminus',
          set(residues_in_nbhd_idx(adj, 1) + 1) == set(range(1, hw + 2)))
    check('neighborhood of aa_pos n_res is clipped at the C-terminus',
          set(residues_in_nbhd_idx(adj, n_res) + 1) == set(range(n_res - hw, n_res + 1)))

    for bad in (0, -1, n_res + 1):
        try:
            residues_in_nbhd_idx(adj, bad)
            check(f'aa_pos {bad} rejected', False, 'no error raised')
        except ValueError:
            check(f'aa_pos {bad} rejected', True)

    # A mask built against a different structure must not be applied silently.
    try:
        check_keep_mask(np.ones(n_res - 1, dtype=bool), n_res, 'TEST')
        check('length-mismatched mask rejected', False, 'no error raised')
    except ValueError as e:
        check('length-mismatched mask rejected', 'positional' in str(e))

    check('check_keep_mask passes None through', check_keep_mask(None, n_res) is None)
    check('check_keep_mask coerces to bool',
          check_keep_mask(np.array([1, 0, 1]), 3).dtype == bool)


# ---------------------------------------------------------------------------
def test_mask_zeroes_the_right_rows():
    """The residues that go to zero are exactly the ones named, in both scans."""
    print('\n2. Masking zeroes exactly the named residues')
    n_res, hw = 80, 3
    adj = banded_adjacency(n_res, hw)
    center = 40
    keep = keep_mask_from_centers(adj, [center])
    masked_aa_pos = set(range(center - hw, center + hw + 1))

    check('mask covers the neighborhood and nothing else',
          set(np.flatnonzero(~keep) + 1) == masked_aa_pos)

    # Standard scan: one allele on every residue, so any shift is visible.
    df = pd.DataFrame({'aa_pos': np.arange(1, n_res + 1),
                       'ac_case': np.ones(n_res, dtype=int),
                       'ac_control': np.ones(n_res, dtype=int)})
    case_mat, ctrl_mat = get_case_control_ac_matrix(df, n_res, 20, seed=0, keep=keep)

    check('case counts zeroed at exactly the masked residues',
          set(np.flatnonzero(case_mat[:, 0] == 0) + 1) == masked_aa_pos)
    check('control counts zeroed at exactly the masked residues',
          set(np.flatnonzero(ctrl_mat[:, 0] == 0) + 1) == masked_aa_pos)
    check('row count preserved -- aa_pos is rebuilt from row order',
          case_mat.shape[0] == n_res)
    check('gene totals exclude the masked residues',
          case_mat[:, 0].sum() == n_res - len(masked_aa_pos))
    check('the null never places an allele on a masked residue',
          case_mat[~keep, 1:].sum() == 0 and ctrl_mat[~keep, 1:].sum() == 0)
    check('the null preserves the post-mask case total',
          set(case_mat[:, 1:].sum(axis=0)) == {case_mat[:, 0].sum()})

    # Mutation-rate scan: counts and opportunity go together.
    x_obs = np.arange(1, n_res + 1).reshape(-1, 1)
    m = np.full((n_res, 1), 0.5)
    x_masked, m_masked = apply_conditioning(x_obs, m, keep)
    check('counts zeroed at exactly the masked residues',
          set(np.flatnonzero(x_masked[:, 0] == 0) + 1) == masked_aa_pos)
    check('mutational opportunity zeroed at the same residues',
          set(np.flatnonzero(m_masked[:, 0] == 0) + 1) == masked_aa_pos)
    check('unmasked residues untouched',
          np.array_equal(x_masked[keep], x_obs[keep]) and np.array_equal(m_masked[keep], m[keep]))
    check('apply_conditioning does not mutate its inputs',
          x_obs[center - 1, 0] == center and m[center - 1, 0] == 0.5)


# ---------------------------------------------------------------------------
def test_neutralization_rule():
    """Which centres stop being tested, and that the rule cannot see the data."""
    print('\n3. Neutralization of centres inside the conditioned region')
    n_res, hw = 200, 5
    adj = banded_adjacency(n_res, hw)
    center = 100
    keep = keep_mask_from_centers(adj, [center])

    retention = neighborhood_retention(adj, keep)
    check('retention is 0 at the centre of the removed neighborhood',
          retention[center - 1] == 0.0)
    check('retention is 1 far from it', retention[9] == 1.0)
    check('retention is between 0 and 1 at the rim',
          0 < retention[center - 1 + hw + 2] < 1, f'{retention[center - 1 + hw + 2]:.3f}')

    dead = neutralized_positions(adj, keep, 0.5)
    check('every masked residue is neutralized', dead[~keep].all())
    check('nothing far away is neutralized', not dead[:center - 3 * hw].any())
    check('a centre keeping less than half its neighborhood is neutralized',
          dead[retention < 0.5].all())

    # min_retained = 0 must still not test an empty neighborhood: under the Poisson null
    # an expected count of zero would hand any stray count an arbitrarily small p-value.
    dead0 = neutralized_positions(adj, keep, 0.0)
    check('retention 0 is neutralized even at --conditional-min-retained 0',
          dead0[retention == 0.0].all())

    # The rule reads geometry and the mask only, so observed and null columns are
    # neutralized identically -- this is what keeps the empirical null calibrated.
    rng = np.random.default_rng(0)
    pvals = rng.random((n_res, 51))
    out = _neutralize(pvals, dead)
    check('all columns neutralized together, observed and null alike',
          np.all(out[dead] == 1.0))
    check('untouched rows are bit-identical', np.array_equal(out[~dead], pvals[~dead]))
    check('_neutralize does not mutate its input', not np.all(pvals[dead] == 1.0))
    check('shape preserved', out.shape == pvals.shape)


# ---------------------------------------------------------------------------
def _mu_scan(adj, x, m, lam, n_sims, rng, keep=None, min_retained=0.5):
    """One masked mutation-rate scan, the way compute_all_pvals_mu assembles it."""
    x, m = apply_conditioning(x, m, keep)
    n_g, m_g = int(x.sum()), float(m.sum())
    x_a = np.hstack([x, simulate_poisson_null(m, lam, n_sims, rng)])
    x_b = np.hstack([x, simulate_multinomial_null(m, n_g, n_sims, rng)])
    p_a, p_b, _, _, _ = _pvals_for_radius(adj, x_a, x_b, m, lam, n_g, m_g)
    if keep is not None:
        dead = neutralized_positions(adj, keep, min_retained)
        p_a, p_b = _neutralize(p_a, dead), _neutralize(p_b, dead)
    return p_a, p_b


def test_masking_mu_is_required():
    """Masking counts but not opportunity breaks the null. Masking both keeps it."""
    print('\n4. Masking m as well as the counts')
    n_res, hw, n_sims = 300, 6, 300
    adj = banded_adjacency(n_res, hw)
    m = np.full((n_res, 1), 1.0)
    lam = 40.0 / m.sum()
    keep = keep_mask_from_centers(adj, [150])
    dead = neutralized_positions(adj, keep, 0.5)
    tested = ~dead
    retention = neighborhood_retention(adj, keep)

    # The centres that matter: still tested, but their neighborhoods lost residues to
    # the conditioned region. This is where a second nearby signal would have to show up.
    partial = tested & (retention < 1.0)

    # --- the mechanism ---------------------------------------------------------
    # Zeroing the counts without zeroing m leaves the neighborhood's expectation intact
    # while its observed count drops by whatever share of the neighborhood was masked.
    ratio_wrong, ratio_right = [], []
    for rep in range(100):
        rng = np.random.default_rng(rep)
        x = rng.poisson(lam * m, size=(n_res, 1))

        x_wrong = x * keep[:, np.newaxis]        # counts masked, m left in place
        _, _, nbhd_w, exp_w, _ = _pvals_for_radius(
            adj, x_wrong, x_wrong, m, lam, int(x_wrong.sum()), float(m.sum()))

        x_right, m_right = apply_conditioning(x, m, keep)   # the code under test
        _, _, nbhd_r, exp_r, _ = _pvals_for_radius(
            adj, x_right, x_right, m_right, lam, int(x_right.sum()), float(m_right.sum()))

        ratio_wrong.append((nbhd_w[partial] / exp_w[partial]).mean())
        ratio_right.append((nbhd_r[partial] / exp_r[partial]).mean())

    mean_wrong, mean_right = float(np.mean(ratio_wrong)), float(np.mean(ratio_right))
    mean_retention = float(retention[partial].mean())
    print(f'      observed/expected in partly-masked neighborhoods: '
          f'{mean_wrong:.3f} with m left in place, {mean_right:.3f} with m masked too '
          f'(mean surviving fraction of those neighborhoods: {mean_retention:.3f})')
    check('leaving m in place drives observed/expected down to the surviving fraction',
          abs(mean_wrong - mean_retention) < 0.05, f'{mean_wrong:.3f} vs {mean_retention:.3f}')
    check('masking m keeps observed/expected at 1', abs(mean_right - 1.0) < 0.08,
          f'{mean_right:.3f}')

    check('expected count in a fully masked neighborhood is zero once m is masked',
          float((adj @ (m * keep[:, np.newaxis]))[149, 0]) == 0.0)
    check('...and nonzero if m is left alone', float((adj @ m)[149, 0]) > 0)

    # --- what it costs ---------------------------------------------------------
    # A real second signal beside the conditioned region is tested against an inflated
    # expectation, so it is much harder to detect. This is the power the conditional
    # analysis exists to have.
    c2, amp, n_reps = 157, 0.5, 80
    hits_wrong = hits_right = 0
    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        x = rng.poisson(lam * m, size=(n_res, 1))
        x[c2 - hw - 1:c2 + hw] += rng.poisson(amp, size=(2 * hw + 1, 1))
        for mode in ('wrong', 'right'):
            if mode == 'wrong':
                xx, mm = x * keep[:, np.newaxis], m
            else:
                xx, mm = apply_conditioning(x, m, keep)
            null = simulate_poisson_null(mm, lam, n_sims, np.random.default_rng(rep + 7))
            x_a = np.hstack([xx, null])
            p_a, _, _, _, _ = _pvals_for_radius(adj, x_a, x_a, mm, lam, int(xx.sum()),
                                                float(mm.sum()))
            p_a = _neutralize(p_a, dead)
            obs = p_a[tested, 0].min()
            sig = (p_a[:, 1:].min(axis=0) <= obs).mean() < 0.05
            top = int(np.argmin(np.where(tested, p_a[:, 0], np.inf))) + 1
            found = sig and abs(top - c2) <= 2 * hw
            if mode == 'wrong':
                hits_wrong += found
            else:
                hits_right += found

    power_wrong, power_right = hits_wrong / n_reps, hits_right / n_reps
    print(f'      power to recover a real cluster beside the conditioned region '
          f'(retention {retention[c2 - 1]:.2f}): {power_wrong:.2f} with m left in place, '
          f'{power_right:.2f} with m masked too')
    check('masking m recovers a neighboring signal the other way misses',
          power_right > 2 * power_wrong and power_right > 0.25,
          f'{power_right:.2f} vs {power_wrong:.2f}')


def test_conditional_null_calibration():
    """Under the null, a conditional round must not invent signal. Per-gene FWER."""
    print('\n5. Calibration of a conditional round under the null')
    n_res, hw, n_sims, n_genes = 250, 6, 300, 250
    adj = banded_adjacency(n_res, hw)
    rng = np.random.default_rng(11)
    m = np.full((n_res, 1), 1.0)
    lam = 60.0 / m.sum()
    keep = keep_mask_from_centers(adj, [125])
    tested = ~neutralized_positions(adj, keep, 0.5)

    hits_a = hits_b = 0
    for _ in range(n_genes):
        x = rng.poisson(lam * m, size=(n_res, 1))
        p_a, p_b = _mu_scan(adj, x, m, lam, n_sims, rng, keep=keep)
        for p, which in ((p_a, 'a'), (p_b, 'b')):
            obs = p[tested, 0].min()
            # Empirical FWER, exactly as empirical_fdr._compute_fwer does it.
            sig = (p[:, 1:].min(axis=0) <= obs).mean() < 0.05
            hits_a += sig if which == 'a' else 0
            hits_b += sig if which == 'b' else 0

    rate_a, rate_b = 100 * hits_a / n_genes, 100 * hits_b / n_genes
    print(f'      per-gene FWER after conditioning:  Test A {rate_a:.1f}%   '
          f'Test B {rate_b:.1f}%')
    check('Test A stays near nominal after conditioning', rate_a <= 9.0, f'{rate_a:.1f}%')
    check('Test B stays near nominal after conditioning', rate_b <= 9.0, f'{rate_b:.1f}%')


# ---------------------------------------------------------------------------
def test_second_signal_recovery():
    """Two separated clusters: round 0 finds one, the conditional round finds the other."""
    print('\n6. Recovery of a second independent signal')
    n_res, hw, n_sims = 300, 6, 400
    adj = banded_adjacency(n_res, hw)
    m = np.full((n_res, 1), 1.0)
    lam = 40.0 / m.sum()
    pos_1, pos_2 = 80, 220          # far enough apart that neither mask touches the other

    found_both = found_second_only = 0
    n_reps = 40
    for rep in range(n_reps):
        rng = np.random.default_rng(1000 + rep)
        x = rng.poisson(lam * m, size=(n_res, 1))
        x[pos_1 - hw - 1:pos_1 + hw] += rng.poisson(1.6, size=(2 * hw + 1, 1))
        x[pos_2 - hw - 1:pos_2 + hw] += rng.poisson(1.2, size=(2 * hw + 1, 1))

        _, p_b = _mu_scan(adj, x, m, lam, n_sims, rng)
        top_0 = int(np.argmin(p_b[:, 0])) + 1
        sig_0 = (p_b[:, 1:].min(axis=0) <= p_b[top_0 - 1, 0]).mean() < 0.05
        if not sig_0:
            continue

        keep = keep_mask_from_centers(adj, [top_0])
        _, p_b1 = _mu_scan(adj, x, m, lam, n_sims, rng, keep=keep)
        tested = ~neutralized_positions(adj, keep, 0.5)
        top_1 = int(np.argmin(np.where(tested, p_b1[:, 0], np.inf))) + 1
        sig_1 = (p_b1[:, 1:].min(axis=0) <= p_b1[top_1 - 1, 0]).mean() < 0.05

        near_1 = abs(top_0 - pos_1) <= 2 * hw
        near_2 = abs(top_1 - pos_2) <= 2 * hw
        if sig_1 and near_1 and near_2:
            found_both += 1
        if sig_1 and near_2:
            found_second_only += 1

    print(f'      {100 * found_both / n_reps:.0f}% of replicates recovered cluster 1 '
          f'in round 0 and cluster 2 in round 1')
    check('the conditional round finds the second cluster',
          found_second_only / n_reps > 0.7, f'{found_second_only / n_reps:.2f}')

    # The mirror image: one cluster only, so the conditional round must find nothing.
    false_second = 0
    for rep in range(n_reps):
        rng = np.random.default_rng(5000 + rep)
        x = rng.poisson(lam * m, size=(n_res, 1))
        x[pos_1 - hw - 1:pos_1 + hw] += rng.poisson(1.6, size=(2 * hw + 1, 1))

        _, p_b = _mu_scan(adj, x, m, lam, n_sims, rng)
        top_0 = int(np.argmin(p_b[:, 0])) + 1
        if not (p_b[:, 1:].min(axis=0) <= p_b[top_0 - 1, 0]).mean() < 0.05:
            continue
        keep = keep_mask_from_centers(adj, [top_0])
        _, p_b1 = _mu_scan(adj, x, m, lam, n_sims, rng, keep=keep)
        tested = ~neutralized_positions(adj, keep, 0.5)
        top_1 = np.min(np.where(tested, p_b1[:, 0], np.inf))
        if (p_b1[:, 1:].min(axis=0) <= top_1).mean() < 0.05:
            false_second += 1

    rate = 100 * false_second / n_reps
    print(f'      with only one cluster planted, a second signal was called in '
          f'{rate:.0f}% of replicates')
    check('conditioning on the only cluster leaves nothing significant', rate <= 10.0,
          f'{rate:.1f}%')


# ---------------------------------------------------------------------------
def test_best_radius_lookup():
    """The conditioning radius comes from the radius the scan actually chose."""
    print('\n7. Per-position best-radius lookup')
    df = pd.DataFrame({
        'uniprot_id': ['P1', 'P1', 'P2'],
        'aa_pos': [10, 20, 10],
        'radius': [9.0, 21.0, np.nan],
    })
    check('reads the radius recorded for that position',
          best_radius_for_hit(df, 'P1', 20, radius_cols=('radius',)) == 21.0)
    check('positions are not confused across proteins',
          best_radius_for_hit(df, 'P1', 10, radius_cols=('radius',)) == 9.0)
    check('falls back when the recorded radius is null',
          best_radius_for_hit(df, 'P2', 10, default=15.0, radius_cols=('radius',)) == 15.0)
    check('falls back when the position is absent',
          best_radius_for_hit(df, 'P3', 99, default=15.0, radius_cols=('radius',)) == 15.0)

    # Mutation-rate mode: Test B is the test of interest, so its radius wins.
    df_mu = pd.DataFrame({'uniprot_id': ['P1'], 'aa_pos': [10],
                          'radius_poisson': [9.0], 'radius_binomial': [18.0]})
    check('mutation-rate mode conditions on the Test B radius',
          best_radius_for_hit(df_mu, 'P1', 10,
                              radius_cols=('radius_binomial', 'radius_poisson')) == 18.0)


# ---------------------------------------------------------------------------
def _write_fake_protein(fid, uid, pvals, nulls):
    """One protein's datasets, shaped the way scan_test writes them."""
    n = len(pvals)
    write_dataset(fid, uid, pd.DataFrame({'p_value': pvals}))
    write_dataset(fid, f'{uid}_null_pval', pd.DataFrame(nulls))
    write_dataset(fid, f'{uid}_nbhd', pd.DataFrame({'nbhd_case': np.ones(n),
                                                    'nbhd_control': np.ones(n)}))
    write_dataset(fid, f'{uid}_radius', pd.DataFrame({'radius': np.full(n, 15.0)}))


def test_round_carries_the_whole_null_pool():
    """
    A later round re-scans only the genes with signal, yet its FDR and FWER must still
    be exome-wide. The mechanism is that the round's p-value file starts as a copy of
    the previous round's and only the re-scanned genes are overwritten, so the untouched
    genes keep contributing their nulls. If that ever stopped working, later rounds
    would be corrected against a handful of genes and every round-1 hit would look far
    more significant than it is.
    """
    print('\n8. A conditional round keeps the exome-wide null pool')
    import h5py
    from empirical_fdr import compute_fdr

    n_res, n_sims = 20, 200
    genes = [f'P{i}' for i in range(5)]
    rng = np.random.default_rng(3)
    tmp = tempfile.mkdtemp()
    try:
        pd.DataFrame({'uniprot_id': genes, 'gene_name': [f'G{i}' for i in range(5)]}) \
            .to_csv(os.path.join(tmp, 'protein_sequence_guide.tsv'), sep='\t', index=False)

        with h5py.File(os.path.join(tmp, 'r0.h5'), 'a') as fid:
            for uid in genes:
                _write_fake_protein(fid, uid, rng.uniform(0.01, 1.0, n_res),
                                    rng.uniform(0, 1, (n_res, n_sims)))
            fid.attrs['n_sims'] = n_sims

        df0 = compute_fdr(tmp, 0.05, pd.DataFrame({'uniprot_id': genes}), tmp, 'r0.h5')
        check('round 0 covers every gene', sorted(df0.uniprot_id.unique()) == genes)
        check('round 0 has one row per residue per gene',
              len(df0) == n_res * len(genes))

        # Round 1: copy, then overwrite one gene the way the driver does.
        shutil.copyfile(os.path.join(tmp, 'r0.h5'), os.path.join(tmp, 'r1.h5'))
        new_p = np.full(n_res, 0.5)
        new_p[7] = 0.002
        with h5py.File(os.path.join(tmp, 'r1.h5'), 'a') as fid:
            _write_fake_protein(fid, 'P2', new_p, rng.uniform(0, 1, (n_res, n_sims)))

        df1 = compute_fdr(tmp, 0.05, pd.DataFrame({'uniprot_id': genes}), tmp, 'r1.h5')
        check('round 1 still covers every gene, not just the re-scanned one',
              sorted(df1.uniprot_id.unique()) == genes)
        check('round 1 still has one row per residue per gene',
              len(df1) == n_res * len(genes))

        p2 = df1[df1.uniprot_id == 'P2'].sort_values('aa_pos')
        check('the re-scanned gene picked up its new p-values',
              np.isclose(p2.p_value.min(), 0.002) and
              int(p2.loc[p2.p_value.idxmin(), 'aa_pos']) == 8)

        # Untouched genes must be bit-identical between the rounds.
        a = df0[df0.uniprot_id != 'P2'].sort_values(['uniprot_id', 'aa_pos'])
        b = df1[df1.uniprot_id != 'P2'].sort_values(['uniprot_id', 'aa_pos'])
        check('untouched genes keep exactly their round-0 p-values',
              np.array_equal(a.p_value.values, b.p_value.values))

        # And their nulls are still in the pool. The direct test: correcting the
        # re-scanned gene on its own -- which is what a round would do if it wrote a
        # fresh file instead of copying -- has to come out less stringent, because the
        # FWER null is a minimum over fewer genes.
        with h5py.File(os.path.join(tmp, 'alone.h5'), 'a') as fid_a, \
             h5py.File(os.path.join(tmp, 'r1.h5'), 'r') as fid_r:
            for key in ('P2', 'P2_null_pval', 'P2_nbhd', 'P2_radius'):
                write_dataset(fid_a, key, fid_r[key][:])
            fid_a.attrs['n_sims'] = n_sims
        df_alone = compute_fdr(tmp, 0.05, pd.DataFrame({'uniprot_id': ['P2']}), tmp,
                               'alone.h5')

        merged = df1[df1.uniprot_id == 'P2'].sort_values('aa_pos').fwer.values
        alone = df_alone.sort_values('aa_pos').fwer.values
        check('correcting against one gene alone is less stringent',
              (merged >= alone).all() and merged.mean() > alone.mean(),
              f'mean fwer {merged.mean():.3f} merged vs {alone.mean():.3f} alone')
    finally:
        shutil.rmtree(tmp)


def test_driver_helpers():
    """Round file naming and criterion columns."""
    print('\n9. Driver bookkeeping')
    from conditional_scan import _round_path, _criterion_column, _radius_columns

    check('p-value file is versioned per round',
          _round_path('all_proteins.pvals.h5', 2) == 'all_proteins.pvals.round2.h5')
    check('fdr file is versioned per round',
          _round_path('all_proteins.fdr.tsv', 1) == 'all_proteins.fdr.round1.tsv')
    check('round 0 file is never overwritten by a later round',
          _round_path('x.h5', 1) != 'x.h5')

    check('standard scan judges on fdr', _criterion_column('fdr', False) == 'fdr')
    check('standard scan judges on fwer', _criterion_column('fwer', False) == 'fwer')
    check('mutation-rate mode requires both tests to pass',
          _criterion_column('fdr', True) == 'fdr_max' and
          _criterion_column('fwer', True) == 'fwer_max')
    check('mutation-rate mode conditions on the Test B radius first',
          _radius_columns(True)[0] == 'radius_binomial')
    check('standard mode has a single radius column', _radius_columns(False) == ('radius',))


def test_position_filter_survives_the_rounds():
    """
    A position-level --df-filter (SCHEMA uses one: AlphaMissense above the 99th
    percentile) must restrict every round, not just round 0. If a later round dropped
    the aa_pos column it would correct over every residue in the protein instead of the
    filtered subset -- a different number of tests and a different null pool, so the
    rounds would not be comparable and round-1 hits would be judged against the wrong
    burden.
    """
    print('\n10. A position-level --df-filter carries into later rounds')
    from empirical_fdr import _prepare_fdr_filters

    df_filter = pd.DataFrame({'uniprot_id': ['A', 'A', 'B', 'C'],
                              'aa_pos': [10, 20, 5, 7]})
    analysis_genes = ['A', 'B']          # C dropped out of round 0

    # This is the expression conditional_scan builds for the later rounds.
    round_filter = df_filter[df_filter['uniprot_id'].isin(analysis_genes)]
    check('the aa_pos column survives', 'aa_pos' in round_filter.columns)
    check('positions are kept for every analysed gene',
          sorted(round_filter.aa_pos) == [5, 10, 20])
    check('genes absent from round 0 are dropped',
          'C' not in set(round_filter.uniprot_id))

    uniprot_list, aa_pos_filters = _prepare_fdr_filters(round_filter)
    check('empirical_fdr reads it as a position filter, not a gene filter',
          aa_pos_filters is not None and aa_pos_filters['A'] == {10, 20})

    # A gene-only filter must still behave as a gene filter.
    gene_only = pd.DataFrame({'uniprot_id': analysis_genes})
    _, none_filter = _prepare_fdr_filters(gene_only)
    check('a gene-only filter stays position-unrestricted', none_filter is None)


if __name__ == '__main__':
    test_indexing_contract()
    test_mask_zeroes_the_right_rows()
    test_neutralization_rule()
    test_masking_mu_is_required()
    test_conditional_null_calibration()
    test_second_signal_recovery()
    test_best_radius_lookup()
    test_round_carries_the_whole_null_pool()
    test_driver_helpers()
    test_position_filter_survives_the_rounds()
    print('\n' + ('ALL TESTS PASSED' if not FAILURES else f'FAILURES: {FAILURES}'))
    sys.exit(1 if FAILURES else 0)
