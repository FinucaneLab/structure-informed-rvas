"""
Regression tests for the defects found reviewing this PR.

Each test here corresponds to a bug that was found, reproduced and fixed during review.
They are gathered in one file deliberately: what they have in common is not a module but
a provenance, and several of them guard a fix that was itself incomplete the first time
-- the component ranking and the per-round gene totals each took two attempts, and the
second attempt is the one with a test.

The bugs they pin down, in the order they appear below:

1-2.  The component reported and masked by a conditional round, and the values written
      for it. Under the mutation-rate null the gate and criterion columns are maxima
      over both tests and so are NOT monotone in the p_binomial that picks a
      component's lead. Ranking on the lead's values, or reporting them, lets a
      component that never cleared the gene gate be masked while the one that did
      resurfaces next round -- or writes an above-cutoff value for a signal the
      analysis accepted, which the documented --all-signals follow-up then drops.

3.    Per-gene totals in a conditional round's table. Conditioning changes n_denovo_gene
      and mu_gene, so round 0's values are wrong for a re-scanned gene; but a round
      re-scans only the active genes, so its own totals are missing for every other gene
      in the table. Both sources are needed.

4.    The radius a conditional row records, under the name the reader for that mode
      actually consults.

5.    An empty components file, which a conditional run produces whenever no gene passes
      the gene gate -- the common case, per the PR description.

6.    The allele guard that must count what the scan will see, not the unmasked frame.

7.    Rate-column resolution: more than one plausible column must raise rather than be
      guessed, because the same file feeds the scan and the per-residue table.

8.    The annotation caches: a failed fetch must not be cached, and a cached alignment
      must actually be used.

9.    Gene-name resolution, which must agree across the artefacts one run writes for one
      protein.

10.   Chromosome discovery in make_uniprot_genomic_coords.

Run: python tests/test_review_regressions.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def _two_centre_component(df_results, **kwargs):
    """
    One component holding two centres, built from a 6-residue toy structure.

    Centres 2 and 4 both cover residue 3, which carries the case mutation, so the
    mutation-sharing edge rule joins them. `df_results` supplies the per-centre
    statistics, which is what each test varies.
    """
    from components import component_table

    n_res = 6
    adj = np.zeros((n_res, n_res), dtype=int)
    for centre, covered in {2: (2, 3), 4: (3, 4)}.items():
        for pos in covered:
            adj[centre - 1, pos - 1] = 1
    df_mut = pd.DataFrame({'aa_pos': [2, 3, 4], 'ac_case': [0, 1, 0], 'mu': [1e-6] * 3})
    return component_table(df_results, 'P1', 'G1', [(2, 10.0), (4, 10.0)],
                           {10.0: adj}, df_mut, 'p_binomial',
                           kwargs.pop('fdr_col', 'fdr_max'),
                           kwargs.pop('fwer_col', 'fwer_max'),
                           mu_mode=True, **kwargs)


# ---------------------------------------------------------------------------
def test_component_ranking_follows_the_gene_gate():
    """
    The masked component must be one that cleared the gate, not one that looks best
    under p_binomial alone.
    """
    print('\n1. Component selection is ranked on the gene gate')
    from components import top_component

    # C1 wins on p_binomial but never cleared the gate; C2 is why the gene is here.
    comps = pd.DataFrame([
        {'uniprot_id': 'P1', 'component_id': 1, 'lead_p': 1e-9,
         'lead_fwer': 0.40, 'best_gate': 0.40},
        {'uniprot_id': 'P1', 'component_id': 2, 'lead_p': 1e-5,
         'lead_fwer': 0.01, 'best_gate': 0.01},
    ])
    check('the component that cleared the gate is the one selected',
          int(top_component(comps, 'P1').component_id) == 2)

    # A component can also clear the gate through a member that is not its lead.
    split = pd.DataFrame([
        {'uniprot_id': 'P1', 'component_id': 1, 'lead_p': 1e-9,
         'lead_fwer': 0.04, 'best_gate': 0.001},
        {'uniprot_id': 'P1', 'component_id': 2, 'lead_p': 1e-5,
         'lead_fwer': 0.01, 'best_gate': 0.010},
    ])
    check("a non-lead member's gate value still counts for its component",
          int(top_component(split, 'P1').component_id) == 1,
          'ranking on lead_fwer would pick 2')

    check('ties fall back to the p-value',
          int(top_component(pd.DataFrame([
              {'uniprot_id': 'P1', 'component_id': 1, 'lead_p': 1e-3,
               'lead_fwer': 0.0, 'best_gate': 0.0},
              {'uniprot_id': 'P1', 'component_id': 2, 'lead_p': 1e-9,
               'lead_fwer': 0.0, 'best_gate': 0.0},
          ]), 'P1').component_id) == 2)

    legacy = pd.DataFrame([
        {'uniprot_id': 'P1', 'component_id': 1, 'lead_p': 1e-9, 'lead_fwer': 0.40},
        {'uniprot_id': 'P1', 'component_id': 2, 'lead_p': 1e-5, 'lead_fwer': 0.01},
    ])
    check('a components file predating best_gate still resolves',
          int(top_component(legacy, 'P1').component_id) == 2)
    check('an empty frame yields no component', top_component(pd.DataFrame(), 'P1') is None)


# ---------------------------------------------------------------------------
def test_component_row_reports_its_own_best():
    """
    The row written for a component must carry the component's values, not its lead's,
    or the documented --all-signals follow-up drops a signal this analysis masked.
    """
    print('\n2. A component row reports the component, not its lead')

    # The lead has the best p_binomial and a failing fdr_max; centre 4 cleared the gate.
    t = _two_centre_component(pd.DataFrame({
        'aa_pos':     [2,    4],
        'p_binomial': [1e-9, 1e-5],
        'fdr_max':    [0.30, 0.004],
        'fwer_max':   [0.40, 0.010],
    }))
    check('the two centres form one component', len(t) == 1 and int(t.iloc[0].n_centers) == 2)
    row = t.iloc[0]
    check('the lead is still the best-scoring centre', int(row.lead_aa_pos) == 2)
    check('best_crit is the minimum over members, not the lead value',
          row.best_crit == 0.004 and row.lead_fdr == 0.30)
    check('best_gate is the minimum over members',
          row.best_gate == 0.010 and row.lead_fwer == 0.40)

    # This is what the conditional writer puts in the crit column, and what
    # _units_from_fdr then filters on.
    check('the reported value survives an --all-signals cutoff the lead would fail',
          row.best_crit < 0.05 <= row.lead_fdr)

    # Monotone case: nothing should move in case/control mode, where the lead's values
    # and the component's minima coincide by construction.
    mono = _two_centre_component(pd.DataFrame({
        'aa_pos':     [2,    4],
        'p_binomial': [1e-9, 1e-5],
        'fdr_max':    [0.001, 0.004],
        'fwer_max':   [0.002, 0.010],
    })).iloc[0]
    check('with monotone columns the lead already is the best',
          mono.best_crit == mono.lead_fdr and mono.best_gate == mono.lead_fwer)


# ---------------------------------------------------------------------------
def test_round_tables_carry_the_right_gene_totals():
    """
    A re-scanned gene needs this round's totals; an untouched gene needs round 0's.
    Taking either source alone leaves half the table wrong or empty.
    """
    print('\n3. Per-gene totals in a conditional round')
    from conditional_scan import _carry_gene_totals

    round0 = pd.DataFrame({'uniprot_id': ['A', 'A', 'B', 'C'],
                           'n_denovo_gene': [40, 40, 12, 7],
                           'mu_gene': [4e-5, 4e-5, 1e-5, 9e-6],
                           'mu_coverage': [0.99, 0.99, 0.98, 1.0]})
    results = pd.DataFrame({'uniprot_id': ['A', 'B', 'C'], 'aa_pos': [1, 2, 3]})
    out = _carry_gene_totals(results, {'A': (10, 1.5e-5, 0.97)}, round0).set_index('uniprot_id')

    check('a re-scanned gene reports the masked total',
          int(out.loc['A', 'n_denovo_gene']) == 10, 'round 0 said 40')
    check('an untouched gene keeps round 0 totals',
          int(out.loc['B', 'n_denovo_gene']) == 12 and int(out.loc['C', 'n_denovo_gene']) == 7)
    check('no gene is left without totals',
          not out[['n_denovo_gene', 'mu_gene', 'mu_coverage']].isna().any().any())
    check('an empty round leaves the table alone',
          list(_carry_gene_totals(results, {}, None).columns) == ['uniprot_id', 'aa_pos'])
    check('round 0 alone still fills the table',
          _carry_gene_totals(results, {}, round0).n_denovo_gene.tolist() == [40, 12, 7])


# ---------------------------------------------------------------------------
def test_conditional_rows_record_a_usable_radius():
    """
    The radius must be written under the name the reader for that mode consults, or the
    annotation step silently falls back to the flat --radius default.
    """
    print('\n4. The radius a conditional row records')
    from conditional_scan import _radius_columns
    from utils import best_radius_for_hit
    from visualize_and_interpret import _fdr_schema, MU_FDR_SCHEMA, STD_FDR_SCHEMA

    for mu_mode, expected_schema in ((False, STD_FDR_SCHEMA), (True, MU_FDR_SCHEMA)):
        col = _radius_columns(mu_mode)[0]
        rec = {'uniprot_id': 'P1', 'aa_pos': 42, 'round': 0, col: 8.0}
        rec.update({'fdr_max': 0.01, 'fwer_max': 0.02, 'p_binomial': 1e-5} if mu_mode
                   else {'fdr': 0.01, 'fwer': 0.02, 'p_value': 1e-5})
        df = pd.DataFrame([rec])
        schema = _fdr_schema(df)
        mode = 'mutation-rate' if mu_mode else 'case/control'
        check(f'{mode}: the table is read with the expected schema',
              schema is expected_schema)
        got = best_radius_for_hit(df, 'P1', 42, default=15.0,
                                  radius_cols=schema['radius_columns']['binomial'])
        check(f'{mode}: the recorded radius is the one the reader finds',
              got == 8.0, f'got {got}, default is 15.0')

    # The contract above only matters if the writer uses the mode's column name.
    import inspect
    import conditional_scan
    writer = inspect.getsource(conditional_scan.conditional_scan)
    check('the conditional writer records the radius under the mode-specific name',
          writer.count('radius_cols[0]: float(comp.radius)') == 2,
          'expected one per rec dict (round 0 and later rounds)')


# ---------------------------------------------------------------------------
def test_empty_components_file_explains_itself():
    """A conditional run with no gene past the gate writes an empty file; reading it
    must say so rather than failing inside pandas."""
    print('\n5. An empty components file')
    from visualize_and_interpret import _units_from_components

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'x.components.tsv')
    pd.DataFrame().to_csv(path, sep='\t', index=False)
    try:
        _units_from_components(path, {})
        check('an empty components file raises a readable error', False, 'no error raised')
    except ValueError as e:
        check('an empty components file raises a readable error',
              'empty' in str(e) and 'gene gate' in str(e))
    except Exception as e:
        check('an empty components file raises a readable error', False,
              f'{type(e).__name__}: {e}')


# ---------------------------------------------------------------------------
def test_allele_guards_count_what_the_scan_sees():
    """
    The case/control guard must count residues surviving conditioning, and the CMH guard
    must use the requested threshold rather than a hardcoded 5.
    """
    print('\n6. Allele guards')
    import inspect
    import scan_test
    import scan_test_cmh

    # Drive the real batch loop with the structure work stubbed out, and record
    # whether the protein reached the scan. keep[i] is residue i+1, so this mask
    # leaves residues 1 and 4 -- 2 of the gene's 20 case alleles.
    df = pd.DataFrame({'uniprot_id': ['P1'] * 4, 'aa_pos': [1, 2, 3, 4],
                       'ac_case': [1, 9, 9, 1], 'ac_control': [1, 9, 9, 1]})
    keep = np.array([True, False, False, True])
    scanned = []
    orig_mask, orig_one = scan_test.build_conditioning_mask, scan_test.scan_test_one_protein
    scan_test.build_conditioning_mask = lambda *a, **k: (keep, len(keep), {'residues': 2})
    scan_test.scan_test_one_protein = lambda *a, **k: scanned.append(a[5])
    try:
        scan_test._process_proteins_batch(
            df, ['P1'], 'ref', 15.0, 25.0, 'out', 10,
            {'P1': [2, 3]}, 'p.h5', seed=1, min_alleles=5)
        skipped_when_masked = scanned == []
        scan_test._process_proteins_batch(
            df, ['P1'], 'ref', 15.0, 25.0, 'out', 10,
            None, 'p.h5', seed=1, min_alleles=5)
        scanned_when_not = scanned == ['P1']
    finally:
        scan_test.build_conditioning_mask, scan_test.scan_test_one_protein = orig_mask, orig_one

    check('a gene whose masked component held its alleles is skipped',
          skipped_when_masked, '2 of 20 case alleles survive --min-alleles 5')
    check('the same gene unconditioned is still scanned', scanned_when_not)

    cmh_src = inspect.getsource(scan_test_cmh.scan_test_cmh)
    check('the CMH guard uses min_alleles, not a hardcoded 5',
          'ac_case"].sum() < min_alleles' in cmh_src and 'ac_case"].sum() < 5' not in cmh_src)
    check('CMH shares the default rather than redefining it',
          scan_test_cmh.DEFAULT_MIN_ALLELES == scan_test.DEFAULT_MIN_ALLELES)
    check('the CMH gene filter is inclusive',
          scan_test_cmh._filter_proteins(
              pd.DataFrame({'uniprot_id': ['P1', 'P1'], 'ac_case': [2, 2],
                            'ac_control': [2, 2]}), None, 3) == ['P1'],
          '4 alleles must pass --min-alleles 3')


# ---------------------------------------------------------------------------
def test_rate_column_is_never_guessed():
    """More than one plausible rate column must raise: the same file feeds the scan and
    the per-residue table, and a silent pick lets them disagree."""
    print('\n7. Rate-column resolution')
    from read_data import identify_rate_column

    try:
        picked = identify_rate_column(['chrom', 'pos', 'mu', 'mutation_rate'], 'f.parquet')
        check('an ambiguous rate file raises rather than guessing', False,
              f'silently picked {picked}')
    except ValueError as e:
        check('an ambiguous rate file raises rather than guessing',
              'Could not identify' in str(e))

    check('a single mu column still resolves',
          identify_rate_column(['chrom', 'pos', 'mu'], '') == 'mu')
    check('a single mu_snp column still resolves',
          identify_rate_column(['chrom', 'pos', 'mu_snp'], '') == 'mu_snp')
    check('roulette_rate keeps its precedence',
          identify_rate_column(['chrom', 'pos', 'roulette_rate', 'mu_snp'], '')
          == 'roulette_rate')
    check('an unlisted mu_ column still works',
          identify_rate_column(['chrom', 'pos', 'mu_custom'], '') == 'mu_custom')


# ---------------------------------------------------------------------------
def test_annotation_caches_do_their_job():
    """A failed fetch must not be cached, and a cached alignment must be used."""
    print('\n8. Annotation caches')
    import uniprot_annotation as ua

    tmp = tempfile.mkdtemp()
    original = ua.requests.get

    def _fail(*args, **kwargs):
        raise Exception('simulated timeout')

    ua.requests.get = _fail
    try:
        out = ua.get_functional_sites_detailed('P12345', cache_dir=tmp)
    finally:
        ua.requests.get = original
    check('a failed fetch returns nothing', out == {})
    check('a failed fetch writes no cache entry', os.listdir(tmp) == [],
          f'wrote {os.listdir(tmp)}')

    # A cached entry that IS written must still be honoured.
    with open(os.path.join(tmp, 'P00001.json'), 'w') as fid:
        json.dump({'7': [{'position': 7, 'feature_type': 'Binding site',
                          'description': 'cached'}]}, fid)
    ua.requests.get = _fail
    try:
        hit = ua.get_functional_sites_detailed('P00001', cache_dir=tmp)
    finally:
        ua.requests.get = original
    check('a cached entry is read without a fetch', list(hit) == ['7'])

    # The alignment cache: a loaded pair must give the same transfer as aligning again.
    query, target = 'MKTAYIAKQRQISFVK', 'MKTAWIAKQRQGGISFVK'
    pair = ua.alignment_to_strings(ua.get_pairwise_alignment(query, target))
    check('an already-gapped pair passes through alignment_to_strings',
          ua.alignment_to_strings(pair) == pair)
    annotations = {'5': [{'position': 5, 'feature_type': 'Binding site',
                          'description': 'x'}]}
    expected = ua.transfer_all_annotations(query, target, annotations)

    # Count the alignments: handing over a cached pair must do none.
    calls = []
    orig_align = ua.get_pairwise_alignment
    ua.get_pairwise_alignment = lambda *a, **k: (calls.append(1), orig_align(*a, **k))[1]
    try:
        with_cache = ua.transfer_all_annotations(query, target, annotations, aligned=pair)
        n_with_cache = len(calls)
        calls.clear()
        ua.transfer_all_annotations(query, target, annotations)
        n_without = len(calls)
    finally:
        ua.get_pairwise_alignment = orig_align

    check('a cached alignment is used rather than recomputed',
          n_with_cache == 0, f'{n_with_cache} alignment(s) computed anyway')
    check('without a cached pair it still aligns', n_without >= 1)
    check('the cached path gives the same transfer', with_cache == expected)


# ---------------------------------------------------------------------------
def test_one_gene_name_per_protein():
    """The resolvers must agree, or one protein's artefacts land under two stems."""
    print('\n9. Gene-name resolution')
    import inspect
    import pymol_code
    from visualize_and_interpret import _load_gene_map

    tmp = tempfile.mkdtemp()
    pd.DataFrame({'uniprot_id': ['P69905', 'P69905'],
                  'gene_name': ['HBA1', 'HBA2']}).to_csv(
        os.path.join(tmp, 'protein_sequence_guide.tsv'), sep='\t', index=False)

    by_map = _load_gene_map(tmp)['P69905']
    by_name = pymol_code._load_gene_name('P69905', tmp)
    check('both resolvers pick the same name for a duplicated accession',
          by_map == by_name, f'map={by_map} name={by_name}')
    # The override must actually win: re-deriving the name here is what split one
    # protein's PSEs from its TSVs.
    seen = []
    originals = {name: getattr(pymol_code, name)
                 for name in ('pymol_rvas', 'pymol_scan_test', '_load_gene_name')}
    for name in ('pymol_rvas', 'pymol_scan_test'):
        setattr(pymol_code, name, lambda *a, **k: seen.append(k.get('gene_name')))
    pymol_code._load_gene_name = lambda *a, **k: 'WRONG'
    try:
        pymol_code.run_all('P69905', 'out', tmp, 'p.h5', gene_name='HBA1_HBA2')
        overridden = seen[:]
        seen.clear()
        pymol_code.run_all('P69905', 'out', tmp, 'p.h5')
        fell_back = seen[:]
    finally:
        for name, fn in originals.items():
            setattr(pymol_code, name, fn)

    check('a passed gene_name reaches every PSE writer',
          overridden and set(overridden) == {'HBA1_HBA2'}, f'got {overridden}')
    check('without one it still falls back to the guide',
          fell_back and set(fell_back) == {'WRONG'}, f'got {fell_back}')


# ---------------------------------------------------------------------------
def test_chromosome_discovery_skips_other_keys():
    """A key that is not {chrom}_{field} must not abort the run."""
    print('\n10. Chromosome discovery')
    import h5py
    import make_uniprot_genomic_coords as mugc

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 't.h5')
    with h5py.File(path, 'w') as fid:
        fid.create_dataset('1_uniprot_id', data=np.array([[b'P1'], [b'P1']]))
        fid.create_dataset('1_pos', data=np.array([[100], [140]]))
        fid.create_dataset('metadata', data=np.array([[1]]))      # not {chrom}_{field}
        fid.create_dataset('2_uniprot_id', data=np.array([[b'P2']]))   # no 2_pos

    with h5py.File(path, 'r') as fid:
        chroms = mugc.discover_chroms(fid)
        check('only fully-formed chromosomes are enumerated', chroms == ['1'],
              f'got {chroms}')
        part = mugc.spans_for_chrom(fid, chroms[0])
        check('the surviving chromosome still reads correctly',
              len(part) == 1 and int(part.n_variants.iloc[0]) == 2)


# ---------------------------------------------------------------------------
def test_component_row_always_has_both_corrections():
    """
    --conditional-criterion fwer makes the member criterion and the gene gate the same
    column. Keying the row on both would write a table with no FDR column, which
    _fdr_schema cannot classify.
    """
    print('\n11. A conditional row carries both corrections')
    from conditional_scan import _criterion_column, _gene_gate_column
    from visualize_and_interpret import _fdr_schema, MU_FDR_SCHEMA, STD_FDR_SCHEMA

    t = _two_centre_component(pd.DataFrame({
        'aa_pos':     [2,    4],
        'p_binomial': [1e-9, 1e-5],
        'fdr_max':    [0.30, 0.004],
        'fwer_max':   [0.40, 0.010],
    }))
    row = t.iloc[0]
    check('a component reports its best under both canonical names',
          row.best_fdr == 0.004 and row.best_fwer == 0.010)

    for mu_mode in (False, True):
        mode = 'mutation-rate' if mu_mode else 'case/control'
        check(f'{mode}: --conditional-criterion fwer does collide',
              _criterion_column('fwer', mu_mode) == _gene_gate_column('fwer', mu_mode),
              'the case this test exists for')

    # Under --conditional-criterion fwer the criterion column IS the gate column, so
    # best_crit is a FWER value. best_fdr must still come from the FDR column.
    collided = _two_centre_component(
        pd.DataFrame({'aa_pos': [2, 4], 'p_binomial': [1e-9, 1e-5],
                      'fdr_max': [0.30, 0.004], 'fwer_max': [0.40, 0.010]}),
        fdr_col='fwer_max', fwer_col='fwer_max',
        gate_col='fwer_max', fdr_name='fdr_max', fwer_name='fwer_max').iloc[0]
    check('with the criterion set to fwer, best_crit is a FWER value',
          collided.best_crit == 0.010)
    check('best_fdr still comes from the FDR column',
          collided.best_fdr == 0.004, f'got {collided.best_fdr}')
    check('best_fwer still comes from the FWER column', collided.best_fwer == 0.010)

    # And the writer must key on those names, not on crit_col/gate_col.
    import inspect
    import conditional_scan
    writer = inspect.getsource(conditional_scan.conditional_scan)
    check('both rec dicts are keyed on the canonical names',
          writer.count('fdr_name: float(comp.best_fdr), fwer_name: float(comp.best_fwer)') == 2)
    check('neither rec dict is keyed on the colliding pair',
          'crit_col: float(comp.' not in writer and 'gate_col: float(comp.' not in writer)

    # A table written that way classifies in both modes.
    for mu_mode, expected in ((False, STD_FDR_SCHEMA), (True, MU_FDR_SCHEMA)):
        mode = 'mutation-rate' if mu_mode else 'case/control'
        df = pd.DataFrame([{'uniprot_id': 'P1', 'aa_pos': 2, 'round': 0,
                            _criterion_column('fdr', mu_mode): 0.004,
                            _criterion_column('fwer', mu_mode): 0.010,
                            ('p_binomial' if mu_mode else 'p_value'): 1e-9}])
        check(f'{mode}: the written table is classifiable downstream',
              _fdr_schema(df) is expected)


# ---------------------------------------------------------------------------
def test_gate_is_rechecked_on_the_component():
    """
    The gene gate admits a gene on some neighborhood, which need not belong to the
    component that gets reported and masked.
    """
    print('\n12. The gene gate is re-checked on the component')
    import inspect
    import conditional_scan

    src_cs = inspect.getsource(conditional_scan.conditional_scan)
    check('both accept loops re-check the component against the gene gate',
          src_cs.count('float(comp.best_gate) < gate_cutoff') == 2,
          'one for round 0, one for later rounds')

    # The predicate itself, on the values that make it matter.
    gate_cutoff = 0.05
    accepted = [best for best in (0.001, 0.04, 0.40) if float(best) < gate_cutoff]
    check('a component above the gene gate is not accepted', accepted == [0.001, 0.04])
    check('NaN is not accepted', not (float('nan') < gate_cutoff))


# ---------------------------------------------------------------------------
def test_allele_guard_excludes_out_of_structure_positions():
    """
    get_ac_per_residue drops aa_pos outside the structure with a warning rather than
    raising, so the guard must not count those alleles as usable.
    """
    print('\n13. Out-of-structure positions do not satisfy --min-alleles')
    import scan_test

    tmp = tempfile.mkdtemp()
    guide = os.path.join(tmp, 'pdb_pae_file_pos_guide.tsv')
    # structure_length reads the maximum end of pos_covered: 4 residues here.
    pd.DataFrame({'uniprot_id': ['P1'], 'pdb_filename': ['P1.pdb'],
                  'pos_covered': ['[1, 4]']}).to_csv(guide, sep='\t', index=False)

    # 8 case alleles on paper, but 6 of them sit past the end of the structure.
    df = pd.DataFrame({'uniprot_id': ['P1'] * 4, 'aa_pos': [1, 2, 90, 91],
                       'ac_case': [1, 1, 3, 3], 'ac_control': [1, 1, 3, 3]})
    scanned = []
    orig = scan_test.scan_test_one_protein
    scan_test.scan_test_one_protein = lambda *a, **k: scanned.append(a[5])
    try:
        scan_test._process_proteins_batch(df, ['P1'], tmp, 15.0, 25.0, tmp, 10,
                                          None, 'p.h5', seed=1, min_alleles=5)
    finally:
        scan_test.scan_test_one_protein = orig
    check('a gene is judged on the alleles inside its structure',
          scanned == [], '2 of 8 case alleles are in range, below --min-alleles 5')

    # The same gene with every position in range is still scanned.
    df_ok = df.assign(aa_pos=[1, 2, 3, 4])
    scanned.clear()
    scan_test.scan_test_one_protein = lambda *a, **k: scanned.append(a[5])
    try:
        scan_test._process_proteins_batch(df_ok, ['P1'], tmp, 15.0, 25.0, tmp, 10,
                                          None, 'p.h5', seed=1, min_alleles=5)
    finally:
        scan_test.scan_test_one_protein = orig
    check('the same alleles inside the structure pass', scanned == ['P1'])


# ---------------------------------------------------------------------------
def test_documented_conditional_columns_exist():
    """The README names the columns a conditional run adds; they must be the real ones."""
    print('\n14. README column names match the writer')
    import inspect
    import conditional_scan

    readme = io_open_readme()
    writer = inspect.getsource(conditional_scan.conditional_scan)
    for col in ('n_conditioned_residues', 'conditioned_on', 'component_id',
                'n_centers', 'n_residues', 'n_case_alleles',
                'n_residues_overlapping_mask'):
        check(f'README and writer agree on {col}',
              col in readme and f"'{col}'" in writer)
    check('the old name is gone from the README', 'n_conditioned_nbhds' not in readme)


def io_open_readme():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'README.md')) as fid:
        return fid.read()


# ---------------------------------------------------------------------------
def test_empty_rate_file_explains_itself():
    """A file with no {chrom}_pos / {chrom}_uniprot_id pair must say so."""
    print('\n15. A rate file with no chromosomes')
    import h5py
    import make_uniprot_genomic_coords as mugc

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'all_missense_variants_gr38.h5')
    with h5py.File(path, 'w') as fid:
        fid.create_dataset('metadata', data=np.array([[1]]))

    sys_argv = sys.argv[:]
    sys.argv = ['make_uniprot_genomic_coords.py', '--reference-dir', tmp,
                '--out', os.path.join(tmp, 'out.tsv')]
    try:
        mugc.main()
        check('an unusable rate file raises a readable error', False, 'no error raised')
    except ValueError as e:
        check('an unusable rate file raises a readable error',
              'No chromosomes found' in str(e), str(e)[:60])
    except Exception as e:
        check('an unusable rate file raises a readable error', False,
              f'{type(e).__name__}: {str(e)[:50]}')
    finally:
        sys.argv = sys_argv


if __name__ == '__main__':
    test_component_ranking_follows_the_gene_gate()
    test_component_row_reports_its_own_best()
    test_round_tables_carry_the_right_gene_totals()
    test_conditional_rows_record_a_usable_radius()
    test_empty_components_file_explains_itself()
    test_allele_guards_count_what_the_scan_sees()
    test_rate_column_is_never_guessed()
    test_annotation_caches_do_their_job()
    test_one_gene_name_per_protein()
    test_chromosome_discovery_skips_other_keys()
    test_component_row_always_has_both_corrections()
    test_gate_is_rechecked_on_the_component()
    test_allele_guard_excludes_out_of_structure_positions()
    test_documented_conditional_columns_exist()
    test_empty_rate_file_explains_itself()
    print('\n' + ('ALL TESTS PASSED' if not FAILURES else f'FAILURES: {FAILURES}'))
    sys.exit(1 if FAILURES else 0)
