"""
Correctness tests for grouping significant neighborhoods into components.

Three things can go wrong quietly here.

The edge rule is about mutations, not geometry: two neighborhoods join only when they
share a residue that actually carries a case mutation. Sharing unmutated residues must
NOT connect them, because that is ordinary overlap between adjacent windows and would
chain distinct sites across a domain. Tests 1-2 pin that down in both directions.

Conditioning now masks a component's whole residue set rather than one centre's
neighborhood, which goes through a different branch of conditioning_keep_mask. The
positional indexing contract in utils.py applies unchanged -- entry i is residue i+1,
and nothing may be dropped -- so test 4 checks the mask lines up with the residues asked
for, off-by-one included.

Test 5 covers the reason the early-stop guard exists: masking a component can take a
gene below the de novo count it needed to be analysed at all.

Run: python tests/test_components.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components import build_components, component_table
from utils import (conditioning_keep_mask, positions_to_pymol_notation,
                   parse_pymol_positions, check_keep_mask)

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def banded_adjacency(n_res, half_width):
    """A 1-D 'structure': residue i is adjacent to those within half_width."""
    idx = np.arange(n_res)
    return (np.abs(idx[:, None] - idx[None, :]) <= half_width).astype(int)


# ---------------------------------------------------------------------------
def test_shared_mutation_connects():
    """Two neighborhoods sharing a mutated residue must land in one component."""
    print('\n1. A shared CASE-MUTATED residue connects')
    adj = {15.0: banded_adjacency(100, 5)}

    # Centres 20 and 28 both reach residue 24, which is mutated.
    centers = [(20, 15.0), (28, 15.0)]
    labels, _, shared = build_components(centers, adj, case_positions={24})
    check('centres sharing a mutated residue are one component',
          labels[0] == labels[1], f'labels={labels.tolist()}')
    check('the shared set is the mutated overlap',
          shared[0] == {24} and shared[1] == {24})

    # Same geometry, but the mutation sits outside the overlap: 20 reaches 16..25 and
    # 28 reaches 24..33, so a mutation at 16 is in only one of them.
    labels2, _, _ = build_components(centers, adj, case_positions={16})
    check('a mutation in only one neighborhood does not connect',
          labels2[0] != labels2[1], f'labels={labels2.tolist()}')


# ---------------------------------------------------------------------------
def test_overlap_without_mutation_does_not_connect():
    """Residue overlap alone must not connect -- this is the chaining guard."""
    print('\n2. Overlap WITHOUT a shared mutation does not connect')
    adj = {15.0: banded_adjacency(100, 5)}
    centers = [(20, 15.0), (24, 15.0)]      # overlap heavily: 15..25 and 19..29

    nbhd_overlap = set(range(15, 26)) & set(range(19, 30))
    check('the two neighborhoods really do overlap', len(nbhd_overlap) > 0,
          f'{len(nbhd_overlap)} residues shared')

    labels, _, _ = build_components(centers, adj, case_positions={60})
    check('heavy overlap with no mutation in it leaves them separate',
          labels[0] != labels[1], f'labels={labels.tolist()}')

    # Chaining is still allowed THROUGH a mutation: A-B share one, B-C share another.
    centers3 = [(20, 15.0), (30, 15.0), (40, 15.0)]
    labels3, _, _ = build_components(centers3, adj, case_positions={25, 35})
    check('transitive chaining through mutated residues is one component',
          len(set(labels3.tolist())) == 1, f'labels={labels3.tolist()}')


# ---------------------------------------------------------------------------
def test_component_table_contents():
    """The table must report residues and mutations faithfully, in PyMOL notation."""
    print('\n3. Component table contents')
    n_res = 100
    adj = {15.0: banded_adjacency(n_res, 5)}
    centers = [(20, 15.0), (28, 15.0), (70, 15.0)]

    df_mut = pd.DataFrame({
        'aa_pos': np.arange(1, n_res + 1),
        'ac_case': 0,
        'mu': 1e-8,
    })
    for pos, n in ((24, 3), (26, 1), (70, 2)):
        df_mut.loc[df_mut.aa_pos == pos, 'ac_case'] = n

    df_results = pd.DataFrame({
        'uniprot_id': 'P1', 'gene_name': 'G1',
        'aa_pos': [20, 28, 70],
        'p_binomial': [1e-9, 1e-7, 1e-6],
        'fdr_max': [0.001, 0.002, 0.01],
        'fwer_max': [0.01, 0.02, 0.04],
    })

    out = component_table(df_results, 'P1', 'G1', centers, adj, df_mut,
                          'p_binomial', 'fdr_max', 'fwer_max', mu_mode=True)

    check('two components (20+28 joined, 70 alone)', len(out) == 2, f'{len(out)} rows')
    big = out[out.n_centers == 2].iloc[0]
    check('lead is the best-scoring centre of its component', big.lead_aa_pos == 20,
          str(big.lead_aa_pos))
    check('residues are the union of member neighborhoods',
          parse_pymol_positions(big.residues_pymol) == set(range(15, 34)),
          big.residues_pymol)
    check('case alleles are counted, not just residues',
          big.n_case_alleles == 4 and big.n_case_residues == 2,
          f'{big.n_case_alleles} alleles over {big.n_case_residues} residues')
    check('case positions round-trip through PyMOL notation',
          parse_pymol_positions(big.case_pymol) == {24, 26}, big.case_pymol)
    check('case_detail keeps recurrence', big.case_detail == '24:3,26:1', big.case_detail)
    check('mu_component is summed over the component', abs(big.mu_component - 19e-8) < 1e-15,
          str(big.mu_component))
    check('mutation-rate mode reports no control columns',
          'n_control_alleles' not in out.columns)

    # case/control mode must report controls instead
    df_cc = df_mut.drop(columns=['mu']).copy()
    df_cc['ac_control'] = 0
    df_cc.loc[df_cc.aa_pos == 30, 'ac_control'] = 5
    out_cc = component_table(df_results, 'P1', 'G1', centers, adj, df_cc,
                             'p_binomial', 'fdr_max', 'fwer_max', mu_mode=False)
    big_cc = out_cc[out_cc.n_centers == 2].iloc[0]
    check('case/control mode reports control alleles',
          big_cc.n_control_alleles == 5 and big_cc.control_pymol == '30',
          f'{big_cc.n_control_alleles} at {big_cc.control_pymol}')


# ---------------------------------------------------------------------------
def test_residue_conditioning_mask():
    """Masking by explicit residues must respect the positional indexing contract."""
    print('\n4. Keep-mask built from an explicit residue set')
    n_res = 50
    residues = {1, 2, 10, 50}          # include both ends to catch off-by-one
    keep, got_n, removed = conditioning_keep_mask(
        None, None, None, 'P1', None, 15.0, residues=residues, n_res=n_res)

    check('mask has one entry per residue, nothing dropped', keep.shape[0] == n_res,
          f'{keep.shape[0]} vs {n_res}')
    check('got_n matches', got_n == n_res)
    check('exactly the requested residues are masked',
          set(np.flatnonzero(~keep) + 1) == residues,
          f'masked {sorted(np.flatnonzero(~keep) + 1)}')
    check('residue 1 maps to index 0', not keep[0])
    check('residue 50 maps to index 49', not keep[49])
    check('an unrequested neighbour is untouched', keep[2] and keep[48])
    check('removed count is reported', removed.get('residues') == len(residues),
          str(removed))
    check('check_keep_mask accepts it', check_keep_mask(keep, n_res, 'P1') is not None)

    # out of range must be caught, not silently wrapped
    try:
        conditioning_keep_mask(None, None, None, 'P1', None, 15.0,
                               residues={51}, n_res=n_res)
        check('a residue past the end is rejected', False)
    except ValueError:
        check('a residue past the end is rejected', True)


# ---------------------------------------------------------------------------
def test_masking_can_exhaust_a_gene():
    """
    Masking a component can leave fewer de novos than entry to the analysis required.

    This is why the conditional loop stops such genes instead of re-scanning them: Test B
    conditions on the gene's remaining de novo total, so at N_g <= 1 no neighborhood can
    reach significance and an empty round is arithmetic, not evidence.
    """
    print('\n5. Masking a component can exhaust a gene')
    n_res = 100
    adj = {15.0: banded_adjacency(n_res, 5)}
    df_mut = pd.DataFrame({'aa_pos': np.arange(1, n_res + 1), 'ac_case': 0, 'mu': 1e-8})
    # 8 de novos, 7 of them packed into one site
    for pos, n in ((24, 4), (26, 3), (80, 1)):
        df_mut.loc[df_mut.aa_pos == pos, 'ac_case'] = n
    total = int(df_mut.ac_case.sum())

    centers = [(20, 15.0), (28, 15.0)]
    df_results = pd.DataFrame({
        'uniprot_id': 'P1', 'gene_name': 'G1', 'aa_pos': [20, 28],
        'p_binomial': [1e-9, 1e-8], 'fdr_max': [0.001, 0.002], 'fwer_max': [0.01, 0.02]})
    out = component_table(df_results, 'P1', 'G1', centers, adj, df_mut,
                          'p_binomial', 'fdr_max', 'fwer_max', mu_mode=True)
    masked = parse_pymol_positions(out.iloc[0].residues_pymol)
    left = int(df_mut.loc[~df_mut.aa_pos.isin(masked), 'ac_case'].sum())

    check('the gene entered with more than --min-denovo de novos', total > 5, f'{total}')
    check('masking the component leaves it at or below the bar', left <= 5,
          f'{total} -> {left} de novos remain')
    check('the component holds the mass, not the residue count',
          out.iloc[0].n_case_alleles == 7, str(out.iloc[0].n_case_alleles))


# ---------------------------------------------------------------------------
def test_units_from_components():
    """
    visualize_and_interpret must take a component's residues from the file.

    Re-deriving them from the lead centre and radius would annotate a different region
    from the one the conditional analysis actually masked -- a component is the union of
    many overlapping neighborhoods, so its lead's own neighborhood is a strict subset.
    """
    print('\n6. Reading components into visualize_and_interpret units')
    from visualize_and_interpret import _units_from_components

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'x.components.tsv')
        pd.DataFrame([
            {'uniprot_id': 'P1', 'gene_name': 'G1', 'component_id': 1, 'round': 0,
             'n_centers': 3, 'lead_aa_pos': 20, 'lead_p': 1e-9,
             'residues_pymol': '15-25+40', 'n_case_alleles': 4,
             'centers_detail': '18:15,20:15,22:12', 'radius': 15.0},
            {'uniprot_id': 'P1', 'gene_name': 'G1', 'component_id': 2, 'round': 1,
             'n_centers': 1, 'lead_aa_pos': 70, 'lead_p': 1e-6,
             'residues_pymol': '65-75', 'n_case_alleles': 2, 'radius': 15.0},
        ]).to_csv(path, sep='\t', index=False)

        units = _units_from_components(path, {})

    check('one unit per component row', len(units) == 2, f'{len(units)} units')
    u = units[0]
    check('residues come from the file, gaps included',
          u['residues'] == set(range(15, 26)) | {40}, str(sorted(u['residues'])[:5]))
    check('label distinguishes components of the same protein',
          [x['label'] for x in units] == ['c1_20', 'c2_70'],
          str([x['label'] for x in units]))
    check('kind marks it as a component', u['kind'] == 'component')
    check('gene name is taken from the file, not the fallback map',
          u['gene_name'] == 'G1')
    check('the later round is carried through',
          units[1]['info'].startswith('component 2 (round 1)'), units[1]['info'])
    check('centers_detail gives each centre its own radius',
          u['centers'] == [(18, 15.0), (20, 15.0), (22, 12.0)], str(u['centers']))
    check('a component with no centre columns falls back to its lead',
          units[1]['centers'] == [(70, 15.0)], str(units[1]['centers']))

    # A file without the component columns must fail loudly, not half-work.
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, 'bad.tsv')
        pd.DataFrame([{'uniprot_id': 'P1', 'aa_pos': 20}]).to_csv(bad, sep='\t', index=False)
        try:
            _units_from_components(bad, {})
            check('a non-components file is rejected', False)
        except ValueError:
            check('a non-components file is rejected', True)


# ---------------------------------------------------------------------------
def test_sphere_thinning():
    """
    The component envelope thins overlapping spheres, and the result is approximate.

    One sphere per member centre is what a component literally is, but 100 overlapping
    transparent spheres render opaque. The thinning must drop near-duplicates and keep
    spread-out centres -- and the docstring must not claim the drawn union still covers
    the component, because it does not.
    """
    print('\n7. Component sphere thinning')
    from pymol_code import _thin_centers

    r = 15.0
    # Three centres 2 A apart, then one far away.
    placed = [((0.0, 0.0, 0.0), r), ((2.0, 0.0, 0.0), r), ((4.0, 0.0, 0.0), r),
              ((40.0, 0.0, 0.0), r)]
    kept = _thin_centers(placed, separation_factor=0.5)
    check('near-duplicate centres collapse to one',
          len(kept) == 2, f'{len(kept)} kept from {len(placed)}')
    check('the distant centre is kept',
          any(abs(k[0][0] - 40.0) < 1e-9 for k in kept))
    check('the first centre wins ties (input order decides)',
          abs(kept[0][0][0]) < 1e-9, str(kept[0][0]))

    # A larger separation keeps fewer; a smaller one keeps more.
    spread = [((float(i) * 5.0, 0.0, 0.0), r) for i in range(10)]
    counts = {f: len(_thin_centers(spread, separation_factor=f))
              for f in (1.0, 0.5, 0.25)}
    check('fewer spheres at larger separation',
          counts[1.0] <= counts[0.5] <= counts[0.25], str(counts))
    check('separation 0.25r keeps every centre 5 A apart',
          counts[0.25] == len(spread), str(counts))

    # The honest part: a dropped centre's own sphere is NOT inside the kept sphere, so
    # the drawn union falls short of the component at its edges. At separation 0.5r a
    # centre 5 A away is dropped (5 < 7.5), but its neighborhood reaches 5 + r = 20 A
    # from the kept centre, well outside the kept sphere of radius r.
    dropped_at = 5.0
    kept2 = _thin_centers([((0.0, 0.0, 0.0), r), ((dropped_at, 0.0, 0.0), r)],
                          separation_factor=0.5)
    check('a centre inside the separation is dropped', len(kept2) == 1,
          f'{len(kept2)} kept')
    far_edge = dropped_at + r          # furthest residue the dropped centre covered
    check('the dropped centre reached beyond the kept sphere, so the envelope '
          'under-covers', far_edge > r, f'{far_edge} A vs radius {r} A')
    # and a centre exactly at the separation is kept, so the rule is >= not >
    kept3 = _thin_centers([((0.0, 0.0, 0.0), r), ((r * 0.5, 0.0, 0.0), r)],
                          separation_factor=0.5)
    check('a centre exactly at the separation is kept', len(kept3) == 2,
          f'{len(kept3)} kept')


# ---------------------------------------------------------------------------
def test_enriched_only():
    """
    Only enriched neighborhoods are clustered, and case-free ones never become
    components.

    The standard case/control scan is two-sided. Clustering its full significant set
    produced, on ClinVar, 1440 "components" of which 81% were lone neighborhoods holding
    no case variant -- depletion calls, which a cluster of case variants is not.
    """
    print('\n8. Enriched-only clustering')
    from components import components_for_genes

    n_res = 60
    adj = banded_adjacency(n_res, 5)

    import components as comp_mod
    real_adj = comp_mod.get_adjacency_matrix
    comp_mod.get_adjacency_matrix = lambda *a, **k: adj
    try:
        df_mut = pd.DataFrame({'aa_pos': np.arange(1, n_res + 1), 'ac_case': 0,
                               'ac_control': 0})
        df_mut.loc[df_mut.aa_pos == 20, 'ac_case'] = 3      # enriched site
        df_mut.loc[df_mut.aa_pos == 50, 'ac_control'] = 9   # depleted site, no cases

        df_results = pd.DataFrame({
            'uniprot_id': 'P1', 'gene_name': 'G1',
            'aa_pos': [20, 50],
            'p_value': [1e-9, 1e-8],
            'fdr': [0.001, 0.002], 'fwer': [0.01, 0.02],
            'nbhd_case': [12, 0], 'nbhd_control': [1, 9],   # enriched / depleted
            'ratio': [8.0, 0.1],
        })
        kw = dict(reference_dir='x', pae_cutoff=15.0, read_mut=lambda u: df_mut,
                  p_col='p_value', fdr_col='fdr', fwer_col='fwer',
                  radius_cols=('radius',), member_cutoff=0.05, mu_mode=False,
                  default_radius=15.0)

        both = components_for_genes(df_results, {'P1'}, enriched_only=False, **kw)
        check('without the filter the case-free depleted one is still dropped',
              len(both) == 1 and both.iloc[0].lead_aa_pos == 20,
              f'{len(both)} components')

        enriched = components_for_genes(df_results, {'P1'}, **kw)
        check('with the effect column only the enriched neighborhood clusters',
              len(enriched) == 1 and enriched.iloc[0].lead_aa_pos == 20,
              f'{len(enriched)} components')
        check('the surviving component carries its case variants',
              enriched.iloc[0].n_case_alleles == 3,
              str(enriched.iloc[0].n_case_alleles))

        # A gene whose ONLY significant neighborhood is depleted yields nothing at all.
        only_dep = df_results[df_results.aa_pos == 50]
        none = components_for_genes(only_dep, {'P1'}, **kw)
        check('a gene with only depleted hits produces no component', len(none) == 0,
              f'{len(none)} components')
    finally:
        comp_mod.get_adjacency_matrix = real_adj


# ---------------------------------------------------------------------------
def test_one_sided_lookup():
    """
    The case/control Fisher lookup is one-sided by default.

    Depleted cells must come back p = 1 and enriched cells must be untouched, because
    the same grid serves the observed column and every null column -- so this is also
    what keeps the empirical FDR calibrated once depletion is dropped.
    """
    print('\n9. One-sided Fisher lookup')
    from scan_test import get_pval_lookup_case_control
    from scipy.stats import fisher_exact

    n_case, n_ctrl = 100, 100
    cases = np.array([[30, 5]])      # a=30 in nbhd vs b=5: enriched
    ctrls = np.array([[5, 30]])      # and the mirror: depleted
    one = get_pval_lookup_case_control(cases, ctrls, n_case, n_ctrl)
    two = get_pval_lookup_case_control(cases, ctrls, n_case, n_ctrl,
                                       include_depletion=True)

    enriched_p = two[30, 5]
    depleted_p = two[5, 30]
    check('two-sided mode calls both directions',
          enriched_p < 0.05 and depleted_p < 0.05,
          f'enriched {enriched_p:.2e}, depleted {depleted_p:.2e}')
    check('one-sided mode sets the depleted cell to exactly 1',
          one[5, 30] == 1.0, str(one[5, 30]))
    check('one-sided mode leaves the enriched cell unchanged',
          one[30, 5] == enriched_p, f'{one[30, 5]:.3e} vs {enriched_p:.3e}')
    check('the enriched value is the real two-sided Fisher p',
          abs(one[30, 5] - fisher_exact([[30, 5], [70, 95]])[1]) < 1e-12)

    # The OR = 1 boundary is deliberately inclusive (a >= expected_a), but it cannot be
    # observed through this function: a cell sitting exactly at its expectation has
    # chi2 = 0, so the pre-filter discards it either way and it keeps p = 1 regardless.
    # Asserting on it would be a test that cannot fail.

    # Everything that is not enriched must be 1 across the whole grid.
    big_c = np.array([[0, 40]]); big_t = np.array([[0, 40]])
    g = get_pval_lookup_case_control(big_c, big_t, n_case, n_ctrl)
    a = np.arange(g.shape[0])[:, None]
    b = np.arange(g.shape[1])[None, :]
    # expected cases in nbhd, at fixed margins
    exp_a = (a + b) * n_case / (n_case + n_ctrl)
    depleted = a < exp_a
    check('no depleted cell anywhere in the grid has p < 1',
          bool((g[depleted] == 1.0).all()),
          f'{int((g[depleted] < 1.0).sum())} depleted cells below 1')
    check('some enriched cell does have p < 1 (the grid is not all ones)',
          bool((g[~depleted] < 1.0).any()))


if __name__ == '__main__':
    test_shared_mutation_connects()
    test_overlap_without_mutation_does_not_connect()
    test_component_table_contents()
    test_residue_conditioning_mask()
    test_masking_can_exhaust_a_gene()
    test_units_from_components()
    test_sphere_thinning()
    test_enriched_only()
    test_one_sided_lookup()
    print('\n' + ('ALL TESTS PASSED' if not FAILURES else f'FAILURES: {FAILURES}'))
    sys.exit(1 if FAILURES else 0)
