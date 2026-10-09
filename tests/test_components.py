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


if __name__ == '__main__':
    test_shared_mutation_connects()
    test_overlap_without_mutation_does_not_connect()
    test_component_table_contents()
    test_residue_conditioning_mask()
    test_masking_can_exhaust_a_gene()
    print('\n' + ('ALL TESTS PASSED' if not FAILURES else f'FAILURES: {FAILURES}'))
    sys.exit(1 if FAILURES else 0)
