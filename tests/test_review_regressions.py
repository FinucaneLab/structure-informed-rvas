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

    # The lead has the best p_binomial and a failing fdr; centre 4 cleared the gate.
    t = _two_centre_component(pd.DataFrame({
        'aa_pos':     [2,    4],
        'p_binomial': [1e-9, 1e-5],
        'fdr':    [0.30, 0.004],
        'fwer':   [0.40, 0.010],
    }))
    check('the two centres form one component', len(t) == 1 and int(t.iloc[0].n_centers) == 2)
    row = t.iloc[0]
    check('the lead is still the best-scoring centre', int(row.lead_aa_pos) == 2)
    check('best_fdr is the minimum over members, not the lead value',
          row.best_fdr == 0.004 and row.lead_fdr == 0.30)
    check('best_gate is the minimum over members',
          row.best_gate == 0.010 and row.lead_fwer == 0.40)

    # This is what the conditional writer puts in the crit column, and what
    # _units_from_fdr then filters on.
    check('the reported value survives an --all-signals cutoff the lead would fail',
          row.best_fdr < 0.05 <= row.lead_fdr)

    # Monotone case: nothing should move in case/control mode, where the lead's values
    # and the component's minima coincide by construction.
    mono = _two_centre_component(pd.DataFrame({
        'aa_pos':     [2,    4],
        'p_binomial': [1e-9, 1e-5],
        'fdr':    [0.001, 0.004],
        'fwer':   [0.002, 0.010],
    })).iloc[0]
    check('with monotone columns the lead already is the best',
          mono.best_fdr == mono.lead_fdr and mono.best_gate == mono.lead_fwer)


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
    from conditional_scan import RESULT_COLUMNS
    from utils import best_radius_for_hit
    from visualize_and_interpret import (_result_columns, MU_RESULT_COLUMNS,
                                         STD_RESULT_COLUMNS)

    for mu_mode, expected_columns in ((False, STD_RESULT_COLUMNS), (True, MU_RESULT_COLUMNS)):
        col = RESULT_COLUMNS[mu_mode].radius[0]
        rec = {'uniprot_id': 'P1', 'aa_pos': 42, 'round': 0, col: 8.0}
        rec.update({'fdr': 0.01, 'fwer': 0.02, 'p_binomial': 1e-5} if mu_mode
                   else {'fdr': 0.01, 'fwer': 0.02, 'p_value': 1e-5})
        df = pd.DataFrame([rec])
        layout = _result_columns(df)
        mode = 'mutation-rate' if mu_mode else 'case/control'
        check(f'{mode}: the table is read with the expected schema',
              layout is expected_columns)
        got = best_radius_for_hit(df, 'P1', 42, default=15.0,
                                  radius_cols=layout['radius_columns']['binomial'])
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

    # A guide row with no usable name. The two resolvers used to disagree here --
    # '/'.join([]) returned '' from one, which _safe turns into 'unnamed', while the
    # other dropped the accession and fell back to it -- so `visualize` and a later
    # `run_all` stemmed the same protein's PSEs differently and stopped sharing the
    # grey chain _gray_pse_path points at.
    for blank in (np.nan, '', '  '):
        pd.DataFrame({'uniprot_id': ['P12345'], 'gene_name': [blank]}).to_csv(
            os.path.join(tmp, 'protein_sequence_guide.tsv'), sep='\t', index=False)
        by_map = _load_gene_map(tmp).get('P12345', 'P12345')
        by_name = pymol_code._load_gene_name('P12345', tmp)
        check(f'a {"missing" if blank is np.nan else "blank"} name falls back to the '
              'accession in both resolvers',
              by_map == by_name == 'P12345', f'map={by_map!r} name={by_name!r}')
        check('so both stem the PSE the same way',
              pymol_code._pse_base('AF-P12345.pdb', by_name, 'P12345')
              == pymol_code._pse_base('AF-P12345.pdb', by_map, 'P12345')
              == 'P12345_P12345',
              f'got {pymol_code._pse_base("AF-P12345.pdb", by_name, "P12345")}')


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
    _result_columns cannot classify.
    """
    print('\n11. A conditional row carries both corrections')
    from conditional_scan import _column_for, FDR_COLUMN, FWER_COLUMN
    from visualize_and_interpret import (_result_columns, MU_RESULT_COLUMNS,
                                         STD_RESULT_COLUMNS)

    t = _two_centre_component(pd.DataFrame({
        'aa_pos':     [2,    4],
        'p_binomial': [1e-9, 1e-5],
        'fdr':    [0.30, 0.004],
        'fwer':   [0.40, 0.010],
    }))
    row = t.iloc[0]
    check('a component reports its best under both canonical names',
          row.best_fdr == 0.004 and row.best_fwer == 0.010)

    for mu_mode in (False, True):
        mode = 'mutation-rate' if mu_mode else 'case/control'
        check(f'{mode}: --conditional-criterion fwer selects the gate column',
              _column_for('fwer') == FWER_COLUMN,
              'member gate and gene gate name the same column -- the case this covers')

    # With --conditional-criterion fwer the member gate IS the FWER column. Each
    # recorded correction must still come from its own column.
    collided = _two_centre_component(
        pd.DataFrame({'aa_pos': [2, 4], 'p_binomial': [1e-9, 1e-5],
                      'fdr': [0.30, 0.004], 'fwer': [0.40, 0.010]}),
        gate_col='fwer').iloc[0]
    check('best_fdr comes from the FDR column',
          collided.best_fdr == 0.004, f'got {collided.best_fdr}')
    check('best_fwer comes from the FWER column', collided.best_fwer == 0.010)
    check('best_gate follows the gene criterion', collided.best_gate == 0.010)

    # Structural: component_table cannot be handed a selector as a column name, so a
    # row can no longer be keyed on one -- what made the collision possible.
    import inspect
    import conditional_scan
    from components import component_table
    check('component_table takes no selector-shaped column parameters',
          not ({'fdr_col', 'fwer_col'} & set(inspect.signature(component_table).parameters)))
    writer = inspect.getsource(conditional_scan.conditional_scan)
    check('both rec dicts are keyed on the canonical names',
          writer.count('FDR_COLUMN: float(comp.best_fdr), FWER_COLUMN: float(comp.best_fwer)') == 2)
    check('neither rec dict is keyed on the colliding pair',
          'crit_col: float(comp.' not in writer and 'gate_col: float(comp.' not in writer)

    # A table written that way classifies in both modes.
    for mu_mode, expected in ((False, STD_RESULT_COLUMNS), (True, MU_RESULT_COLUMNS)):
        mode = 'mutation-rate' if mu_mode else 'case/control'
        df = pd.DataFrame([{'uniprot_id': 'P1', 'aa_pos': 2, 'round': 0,
                            FDR_COLUMN: 0.004,
                            FWER_COLUMN: 0.010,
                            ('p_binomial' if mu_mode else 'p_value'): 1e-9}])
        check(f'{mode}: the written table is classifiable downstream',
              _result_columns(df) is expected)


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


# ---------------------------------------------------------------------------
def _drive_conditional(best_gates, gate_cutoff=0.05, max_rounds=1, fails_rescan=(),
                       moving_lead=False, spent=()):
    """
    Run conditional_scan for real over a synthetic two-gene result table, with the
    scan and the component builder stubbed.

    `best_gates` maps uniprot_id -> the component's best_gate. `moving_lead` walks each
    round's component to fresh residues, so a gene survives the masked-lead guard and
    the run reaches round 2; `spent` names genes whose remaining allele counts fall
    below the bar, which the real run reads out of the p-value file. Returns
    (genes re-scanned in round 1, the conditional rows written, the components table).
    """
    import h5py
    import utils
    import conditional_scan as cs

    tmp = tempfile.mkdtemp()
    genes = sorted(best_gates)
    df_results = pd.DataFrame({
        'uniprot_id': genes, 'gene_name': genes, 'aa_pos': [10] * len(genes),
        'p_value': [1e-8] * len(genes), 'fdr': [0.001] * len(genes),
        'fwer': [0.001] * len(genes), 'radius': [10.0] * len(genes),
    })
    comps = pd.DataFrame([
        {'uniprot_id': g, 'gene_name': g, 'component_id': 1, 'round': 0,
         'n_centers': 1, 'lead_aa_pos': 10, 'lead_p': 1e-8, 'lead_fdr': 0.001,
         'lead_fwer': best_gates[g], 'best_gate': best_gates[g],
         'best_fdr': 0.001, 'best_fwer': best_gates[g],
         'radius': 10.0, 'n_residues': 3, 'n_case_alleles': 4,
         'residues_pymol': '9-11', 'centers_pymol': '10',
         'centers_detail': '10:10', 'seq_span_aa': 3,
         'n_case_residues': 1, 'case_pymol': '10', 'case_detail': '10:4',
         'n_control_residues': 0, 'n_control_alleles': 0, 'control_pymol': ''}
        for g in genes])

    rescanned = []
    pools = []          # the gene filter each round's corrections were computed over

    def fake_scan(*args, **kwargs):
        # Round 0 passes uniprot_id_list=None; later rounds pass the active set.
        asked = kwargs.get('uniprot_id_list')
        if asked is not None:
            rescanned.append(sorted(asked))
        fdr_name = args[11]
        df_results.to_csv(os.path.join(tmp, fdr_name), sep='\t', index=False)
        # A real (empty) HDF5 file: the exhaustion check opens this one for real.
        h5py.File(os.path.join(tmp, args[12]), 'a').close()
        # scan_test reports which genes it actually wrote. `fails_rescan` names genes
        # the scan skips -- too few alleles after conditioning, a missing structure --
        # which keep their previous, unconditioned p-values in the copied file.
        if asked is None:
            return set(genes)
        return {g for g in asked if g not in fails_rescan}

    def fake_components(*a, **k):
        r = k.get('round_idx', 0)
        out = comps.assign(round=r)
        if moving_lead and r:
            # A fresh signal each round, 10 residues along from the last.
            out = out.assign(lead_aa_pos=10 + 10 * r,
                             residues_pymol=f'{9 + 10 * r}-{11 + 10 * r}')
        return out

    # The real read_mut opens the current p-value file and reads per-residue counts.
    # Give the spent genes everything inside the mask round 1 applies (9-11) and the
    # others everything outside it.
    def fake_mut(node, uniprot_id, mu_mode=False):
        pos = 10 if uniprot_id in spent else 50
        return pd.DataFrame({'aa_pos': [pos], 'ac_case': [10], 'ac_control': [10],
                             'mu': [1e-6]})

    originals = (cs.scan_test, cs.components_for_genes, cs.compute_fdr)
    utils_originals = (utils.resolve_pval_node, utils.read_original_mutation_data)
    cs.scan_test = fake_scan
    cs.components_for_genes = fake_components
    if spent:
        utils.resolve_pval_node = lambda fid, test: (None, False)
        utils.read_original_mutation_data = fake_mut
    def fake_fdr(results_dir, fdr_cutoff, df_fdr_filter, *a, **k):
        pools.append(sorted(df_fdr_filter['uniprot_id'].unique()))
        return df_results
    cs.compute_fdr = fake_fdr
    try:
        cs.conditional_scan(
            df_rvas=pd.DataFrame({'uniprot_id': genes, 'aa_pos': [10] * len(genes),
                                  'ac_case': [4] * len(genes),
                                  'ac_control': [4] * len(genes)}),
            reference_dir=tmp, radius=10.0, pae_cutoff=25.0, results_dir=tmp,
            n_sims=10, fdr_cutoff=0.05, df_filter=None, fdr_file='r.fdr.tsv',
            pval_file='r.pvals.h5', seed=1, mutation_rate_mode=False,
            ignore_ac=False, criterion='fdr', cutoff=0.05,
            gene_criterion='fwer', gene_cutoff=gate_cutoff,
            max_rounds=max_rounds, min_retained=0.5, include_depletion=False,
            min_alleles=5, min_denovo=5, mu_kwargs={},
        )
    finally:
        cs.scan_test, cs.components_for_genes, cs.compute_fdr = originals
        utils.resolve_pval_node, utils.read_original_mutation_data = utils_originals

    stem = 'r.fdr'
    rows = pd.read_csv(os.path.join(tmp, f'{stem}.conditional.tsv'), sep='\t')
    comp_rows = pd.read_csv(os.path.join(tmp, f'{stem}.components.tsv'), sep='\t')
    return rescanned, rows, comp_rows, pools


def test_rejected_gene_is_not_masked_or_rescanned():
    """
    The round-0 guard must stop the masking it names, not just the report. Rounds >= 1
    filter the carried-forward set to the genes they accepted; round 0 did not.
    """
    print('\n16. A gene the gate guard rejects is dropped entirely')

    # ACCEPT clears the gate; REJECT's component does not.
    rescanned, rows, comp_rows, _pools = _drive_conditional({'ACCEPT': 0.001, 'REJECT': 0.30})

    check('only the accepted gene is reported in round 0',
          sorted(rows[rows['round'] == 0].uniprot_id) == ['ACCEPT'],
          f"got {sorted(rows[rows['round'] == 0].uniprot_id)}")
    check('the rejected gene is not carried into round 1',
          rescanned and 'REJECT' not in rescanned[0],
          f'round 1 re-scanned {rescanned[0] if rescanned else None}')
    check('the accepted gene is carried into round 1',
          rescanned and rescanned[0] == ['ACCEPT'])
    check('the rejected gene gets no components row either',
          'REJECT' not in set(comp_rows.uniprot_id),
          f'components.tsv holds {sorted(set(comp_rows.uniprot_id))}')
    check('the rejected gene appears nowhere in the conditional table',
          'REJECT' not in set(rows.uniprot_id))

    # A gene that passes the gate but whose round-1 re-scan never happens must be
    # dropped from that round rather than carried with stale p-values.
    rescanned2, rows2, _, pools2 = _drive_conditional(
        {'ACCEPT': 0.001, 'OTHER': 0.001}, max_rounds=2, fails_rescan={'ACCEPT'})
    check('a gene asked for in round 1 is still asked for',
          rescanned2 and 'ACCEPT' in rescanned2[0])
    check('a gene whose re-scan failed is cut from that round\'s correction pool',
          pools2 and 'ACCEPT' not in pools2[-1],
          f'pool was {pools2[-1] if pools2 else None}; its stale p-values would '
          'otherwise be corrected against')
    check('the genes that were re-scanned stay in the pool',
          pools2 and 'OTHER' in pools2[-1])
    check('it reports no round-1 signal either',
          'ACCEPT' not in set(rows2[rows2['round'] == 1].uniprot_id))


# ---------------------------------------------------------------------------
def test_lead_columns_are_never_mislabelled():
    """
    lead_fdr and lead_fwer must hold the correction they are named for, whichever
    column happens to be the criterion or the gene gate.
    """
    print('\n17. lead_fdr / lead_fwer hold what they say')

    # Gene criterion fdr: the gate is the FDR column. lead_fwer must still be FWER.
    row = _two_centre_component(
        pd.DataFrame({'aa_pos': [2, 4], 'p_binomial': [1e-9, 1e-5],
                      'fdr': [0.30, 0.004], 'fwer': [0.40, 0.010]}),
        gate_col='fdr', fdr_name='fdr', fwer_name='fwer').iloc[0]
    check('lead_fwer is a FWER value, not the gate column',
          row.lead_fwer == 0.40, f'got {row.lead_fwer} (fdr at the lead is 0.30)')
    check('lead_fdr is an FDR value', row.lead_fdr == 0.30)
    check('best_gate still follows the gene criterion', row.best_gate == 0.004)

    # Structural: there is no parameter through which the gate column could reach
    # lead_fwer any more -- component_table takes the correction names and the gate
    # separately, so the mix-up is unrepresentable rather than merely fixed.
    import inspect
    from components import component_table
    params = inspect.signature(component_table).parameters
    check('component_table names the corrections, not a gate-shaped column',
          'fwer_name' in params and 'fwer_col' not in params)
    check('fwer_name defaults to the canonical column',
          params['fwer_name'].default == 'fwer')


# ---------------------------------------------------------------------------
def test_cmh_remove_nbhd_is_per_protein():
    """
    --remove-nbhd names residues in one protein. The CMH backend used to take the raw
    string and apply it to every protein it scanned.
    """
    print('\n18. CMH --remove-nbhd applies to one protein')
    import scan_test_cmh

    adj = np.zeros((20, 20), dtype=int)
    adj[9, 8:12] = 1                      # neighborhood of residue 10 is 9..12
    df = pd.DataFrame({'uniprot_id': ['P1'] * 3, 'aa_pos': [9, 10, 15],
                       'ac_case': [1, 1, 1], 'ac_control': [1, 1, 1]})

    out = scan_test_cmh._remove_neighborhood(df, adj, [(10, 10.0)], 'P1')
    check('the named centre removes its neighborhood',
          sorted(out.aa_pos) == [15], f'kept {sorted(out.aa_pos)}')
    check('a centre outside the structure is rejected',
          _raises(scan_test_cmh._remove_neighborhood, df, adj, [(99, 10.0)], 'P1'))

    # Drive the real scan loop over two proteins with only one named, and record the
    # frame each one was scanned with. This is where the bug lived: the loop applied
    # the raw --remove-nbhd string to every protein it touched.
    tmp = tempfile.mkdtemp()
    df_rvas = pd.DataFrame({
        'uniprot_id': ['P1'] * 3 + ['P2'] * 3,
        'aa_pos': [9, 10, 15] * 2,
        'ac_case': [3] * 6, 'ac_control': [3] * 6, 'stratum': ['s'] * 6,
    })
    seen = {}
    originals = {n: getattr(scan_test_cmh, n) for n in
                 ('get_adjacency_matrix', 'compute_all_cmh_pvals', 'write_cmh_pvals')}
    scan_test_cmh.get_adjacency_matrix = lambda *a, **k: adj
    scan_test_cmh.compute_all_cmh_pvals = (
        lambda **k: (seen.__setitem__(k['uniprot_id'], sorted(k['df'].aa_pos)),
                     (pd.DataFrame({'p_value': [0.5]}), None))[1])
    scan_test_cmh.write_cmh_pvals = lambda *a, **k: None
    try:
        scan_test_cmh.scan_test_cmh(
            df_rvas, tmp, 10.0, 25.0, tmp, 10, True, False, 0.05, None, False,
            'f.tsv', 'p.h5', {'P1': [(10, 10.0)]}, 'stratum', min_alleles=1)
    finally:
        for name, fn in originals.items():
            setattr(scan_test_cmh, name, fn)

    check('both proteins were scanned', sorted(seen) == ['P1', 'P2'], f'saw {sorted(seen)}')
    check('the named protein loses its neighborhood',
          seen.get('P1') == [15], f"P1 scanned with {seen.get('P1')}")
    check('every other protein is untouched',
          seen.get('P2') == [9, 10, 15], f"P2 scanned with {seen.get('P2')}")

    # run.py must hand the backend the per-protein dict, not the raw flag.
    import inspect
    import run
    main_src = inspect.getsource(run.main)
    cmh_call = main_src[main_src.index('scan_test_cmh('):]
    check('run.py gives the CMH backend the conditioning dict',
          'remove_nbhd_conditioning(args)' in cmh_call[:cmh_call.index(')\n')],
          'it used to pass args.remove_nbhd')


def _raises(fn, *args):
    try:
        fn(*args)
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------------------
def test_gene_name_matches_the_results_table():
    """
    Three places name a protein: the results table, _load_gene_map and
    _load_gene_name. All three must agree or one run writes two sets of files.
    """
    print('\n19. Gene names agree with the results table')
    import pymol_code
    from visualize_and_interpret import _load_gene_map

    tmp = tempfile.mkdtemp()
    guide = pd.DataFrame({'uniprot_id': ['P69905', 'P69905', 'P1'],
                          'gene_name': ['HBA2', 'HBA1', 'SOLO']})
    guide.to_csv(os.path.join(tmp, 'protein_sequence_guide.tsv'), sep='\t', index=False)

    # What empirical_fdr.compute_fdr writes into the results table's gene_name column.
    from_table = (guide[['uniprot_id', 'gene_name']].dropna().drop_duplicates()
                       .groupby('uniprot_id')['gene_name']
                       .apply(lambda names: '/'.join(sorted(names))).to_dict())

    by_map = _load_gene_map(tmp)
    check('_load_gene_map matches the results table',
          by_map['P69905'] == from_table['P69905'] == 'HBA1/HBA2',
          f"map={by_map['P69905']} table={from_table['P69905']}")
    check('_load_gene_name matches too',
          pymol_code._load_gene_name('P69905', tmp) == from_table['P69905'],
          f"name={pymol_code._load_gene_name('P69905', tmp)}")
    check('an accession with one name is unaffected',
          by_map['P1'] == pymol_code._load_gene_name('P1', tmp) == 'SOLO')


# ---------------------------------------------------------------------------
def test_mu_fdr_is_the_max_over_both_tests():
    """
    `fdr`/`fwer` under the mutation-rate null are max(poisson, binomial), not the
    Poisson value.

    In practice the two are usually equal, because Test A -- the regional-constraint
    guard -- is the binding one. That makes the max look redundant and invites
    collapsing it to fdr_poisson. It is not redundant: the max is what makes passing
    require passing BOTH tests, and Test B is the test of interest. Wherever
    fdr_binomial exceeds fdr_poisson, dropping the max would call a neighborhood
    significant that currently fails. Renaming these columns from fdr_max/fwer_max was
    deliberate; changing what they hold would be a separate statistical decision.
    """
    print('\n20. The mutation-rate fdr/fwer keep max semantics')
    from mutation_rate_test import _merge_test_results

    # Row 0: Poisson binds. Row 1: binomial binds -- the case that matters.
    base = {'uniprot_id': ['P1', 'P1'], 'aa_pos': [1, 2],
            'nbhd_expected': [1.0, 1.0], 'obs_exp': [2.0, 2.0], 'radius': [10.0, 10.0]}
    a = pd.DataFrame({**base, 'p_value': [1e-3, 1e-9],
                      'fdr': [0.20, 0.001], 'fwer': [0.30, 0.002]})
    b = pd.DataFrame({**base, 'p_value': [1e-9, 1e-3],
                      'fdr': [0.001, 0.40], 'fwer': [0.002, 0.50]})
    m = _merge_test_results(a, b, 0.05, 0.05).sort_values('aa_pos').reset_index(drop=True)

    check('fdr is the max over the two tests',
          m.fdr.tolist() == [0.20, 0.40], f'got {m.fdr.tolist()}')
    check('fwer is the max over the two tests',
          m.fwer.tolist() == [0.30, 0.50], f'got {m.fwer.tolist()}')
    check('where binomial binds, fdr is NOT fdr_poisson',
          m.fdr.iloc[1] == 0.40 and m.fdr_poisson.iloc[1] == 0.001,
          'collapsing to fdr_poisson would call this neighborhood significant')
    check('passing requires passing both tests',
          m.sig_fdr_both.tolist() == [False, False],
          'each row passes one test and fails the other')
    check('the per-test columns remain for inspection',
          all(c in m.columns for c in ('fdr_poisson', 'fdr_binomial',
                                       'fwer_poisson', 'fwer_binomial')))
    check('the old fdr_max / fwer_max names are gone',
          'fdr_max' not in m.columns and 'fwer_max' not in m.columns)


# ---------------------------------------------------------------------------
def test_old_results_tables_still_read():
    """A table written before the rename holds the same values under the old names."""
    print('\n21. Tables predating the fdr_max rename')
    from visualize_and_interpret import (normalize_result_columns, _result_columns,
                                         MU_RESULT_COLUMNS)

    old = pd.DataFrame({'uniprot_id': ['P1'], 'aa_pos': [1],
                        'fdr_max': [0.01], 'fwer_max': [0.02],
                        'p_binomial': [1e-5], 'p_poisson': [1e-4]})
    norm = normalize_result_columns(old)
    check('fdr_max is read as fdr', norm['fdr'].iloc[0] == 0.01)
    check('fwer_max is read as fwer', norm['fwer'].iloc[0] == 0.02)
    check('an old mutation-rate table is still recognised as one',
          _result_columns(norm) is MU_RESULT_COLUMNS)

    current = pd.DataFrame({'uniprot_id': ['P1'], 'aa_pos': [1], 'fdr': [0.01],
                            'fwer': [0.02], 'p_binomial': [1e-5]})
    check('a current table is passed through untouched',
          normalize_result_columns(current).equals(current))


# ---------------------------------------------------------------------------
def test_get_nbhd_finds_the_recorded_radius():
    """
    --get-nbhd under a multi-radius scan looks the chosen radius up in the results
    file. A mutation-rate table records it per test, so checking only `radius` found
    nothing and silently fell back to 15 -- the opposite of what the branch is for.
    """
    print('\n22. --get-nbhd reads the radius the scan chose')
    from utils import best_radius_for_hit

    cols = ('radius', 'radius_binomial', 'radius_poisson')
    std = pd.DataFrame({'uniprot_id': ['P1'], 'aa_pos': [42], 'radius': [8.0]})
    mu = pd.DataFrame({'uniprot_id': ['P1'], 'aa_pos': [42],
                       'radius_binomial': [12.0], 'radius_poisson': [20.0]})

    check('a case/control table yields its radius',
          best_radius_for_hit(std, 'P1', 42, default=15.0, radius_cols=cols) == 8.0)
    check('a mutation-rate table yields the Test B radius',
          best_radius_for_hit(mu, 'P1', 42, default=15.0, radius_cols=cols) == 12.0,
          'checking only `radius` fell back to 15.0 here')
    check('an absent position still falls back',
          best_radius_for_hit(mu, 'P1', 99, default=15.0, radius_cols=cols) == 15.0)

    # The position as argparse actually supplies it. --aa-pos is type=str, and a
    # string never matches an int64 aa_pos column, so the lookup returned the default
    # no matter what the table held -- the previous version of this test passed an int
    # and so could not see that.
    import run
    parsed = run.build_parser().parse_args(['--get-nbhd', '--aa-pos', '42'])
    check('--aa-pos arrives as a string', isinstance(parsed.aa_pos, str))
    row = pd.DataFrame({'uniprot_id': ['P1'], 'aa_pos': [42], 'radius_binomial': [12.0]})
    check('the raw string finds nothing',
          best_radius_for_hit(row, 'P1', parsed.aa_pos, default=15.0,
                              radius_cols=cols) == 15.0,
          'which is why the call site must coerce')
    check('coercing it finds the recorded radius',
          best_radius_for_hit(row, 'P1', int(parsed.aa_pos), default=15.0,
                              radius_cols=cols) == 12.0)

    # run.py must use the lookup, on a coerced position, rather than one hardcoded column.
    import inspect
    src_main = inspect.getsource(run.main)
    branch = src_main[src_main.index('elif args.get_nbhd:'):]
    branch = branch[:branch.index('did_nothing = False')]
    check('--get-nbhd uses best_radius_for_hit', 'best_radius_for_hit' in branch)
    check('--get-nbhd coerces the position before looking it up',
          'int(args.aa_pos)' in branch)
    check('--get-nbhd does not test for a single radius column',
          "'radius' in df_fdr_lookup.columns" not in branch)


# ---------------------------------------------------------------------------
def test_failed_rescan_leaves_the_round_pool():
    """
    A gene whose re-scan does not happen keeps the previous round's unconditioned
    p-values in the copied file. Those must not enter the round's FDR/FWER pool.
    """
    print('\n23. A failed re-scan is dropped from the round')
    from conditional_scan import _drop_failed_rescans

    round_filter = pd.DataFrame({'uniprot_id': ['A', 'B', 'C', 'D']})
    active, kept_filter = _drop_failed_rescans(['A', 'B'], {'A'}, round_filter, 1)
    check('a gene that was not re-scanned leaves the active set', active == ['A'])
    check('and leaves the pool the corrections are computed over',
          sorted(kept_filter.uniprot_id) == ['A', 'C', 'D'],
          'B keeps unconditioned p-values, so it must not be corrected against')
    check('genes that were never active are untouched',
          {'C', 'D'} <= set(kept_filter.uniprot_id))

    unchanged, same = _drop_failed_rescans(['A', 'B'], {'A', 'B'}, round_filter, 1)
    check('a fully successful round is left alone',
          unchanged == ['A', 'B'] and same is round_filter)

    none_left, _ = _drop_failed_rescans(['A'], set(), round_filter, 1)
    check('a round where nothing was re-scanned yields no active genes',
          none_left == [])

    # The grey-chain PSE path, which the same loop builds from a full path.
    from pymol_code import _gray_pse_path
    check('a dotted results directory keeps its path',
          _gray_pse_path('/data/v1.2/out/HBA1_P1.pse') == '/data/v1.2/out/HBA1_P1_gray.pse',
          "split('.') truncated this to /data/v1_gray.pse")
    check('a relative path keeps its directory',
          _gray_pse_path('./results/HBA1_P1.pse') == './results/HBA1_P1_gray.pse')
    check('a plain name still works',
          _gray_pse_path('plain.pse') == 'plain_gray.pse')

    # scan_test must report what it wrote, or the caller cannot tell.
    import inspect
    import scan_test
    batch = inspect.getsource(scan_test._process_proteins_batch)
    check('the scan batch records the genes it wrote',
          'scanned.add(uniprot_id)' in batch and batch.rstrip().endswith('return scanned'))
    check('scan_test passes that back to its caller',
          inspect.getsource(scan_test.scan_test).rstrip().endswith('return scanned'))


# ---------------------------------------------------------------------------
def test_genes_that_stop_being_rescanned_stay_out_of_the_pool():
    """
    Two ways a gene stops being re-scanned, one pool it must leave -- and stay out of.

    `_drop_failed_rescans` already cut the genes a scan skipped, but only from the
    round-local copy of the filter, and the exhaustion check ahead of it cut its genes
    from `active` alone. Both leave unconditioned p-values in the pool every later
    round copies.
    """
    print('\n24. A gene that is not re-scanned leaves the pool for good')

    # Exhausted: SPENT's alleles all sit inside the mask round 1 applies, so the
    # round-1 check drops it before the re-scan is paid for.
    rescanned, _rows, _comps, pools = _drive_conditional(
        {'SPENT': 0.001, 'KEEPS': 0.001}, max_rounds=1, moving_lead=True,
        spent={'SPENT'})
    check('an exhausted gene is not re-scanned',
          rescanned and rescanned[0] == ['KEEPS'],
          f'round 1 re-scanned {rescanned[0] if rescanned else None}')
    check('and leaves the pool its round is corrected over',
          pools and 'SPENT' not in pools[0],
          f'pool was {pools[0] if pools else None}; SPENT keeps round 0\'s '
          'unconditioned p-values, which would shift every other gene\'s FDR')
    check('the gene that was re-scanned stays in the pool',
          pools and pools[0] == ['KEEPS'])

    # Failed re-scan: the exclusion has to outlive the round that made it, because
    # round 2 copies round 1's p-value file -- stale values and all.
    rescanned2, _rows2, _comps2, pools2 = _drive_conditional(
        {'FAILS': 0.001, 'KEEPS': 0.001}, max_rounds=2, fails_rescan={'FAILS'},
        moving_lead=True)
    check('the run reached round 2', len(pools2) == 2, f'{len(pools2)} rounds corrected')
    check('a failed re-scan leaves the round it failed in',
          pools2 and 'FAILS' not in pools2[0], f'round 1 pool was {pools2[0]}')
    check('and is still out in the next round',
          len(pools2) > 1 and 'FAILS' not in pools2[1],
          f'round 2 pool was {pools2[1] if len(pools2) > 1 else None}; the exclusion '
          'used to be thrown away with the round-local filter')


# ---------------------------------------------------------------------------
def test_cmh_warns_that_it_ignores_the_conditional_radius():
    """
    --conditional-radius is honoured by the standard path and discarded by the CMH
    one, which removes the --neighborhood-radius neighborhood. That has to be said.
    """
    print('\n25. --test-method cmh says it is ignoring --conditional-radius')
    import logging
    import run

    class _Args:
        remove_nbhd = '10'
        uniprot_id = 'P1'
        neighborhood_radius = 15.0
        conditional_radius = 25.0
        test_method = 'cmh'

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    run.logger.addHandler(handler)
    try:
        cond = run.remove_nbhd_conditioning(_Args())
        _Args.test_method = 'standard'
        n_before = len(records)
        run.remove_nbhd_conditioning(_Args())
        quiet = len(records) == n_before
    finally:
        run.logger.removeHandler(handler)

    warned = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    check('cmh warns that --conditional-radius is ignored',
          any('conditional-radius' in m and 'cmh' in m for m in warned),
          f'logged {warned}')
    check('the warning names the radius actually used',
          any('15' in m for m in warned), f'logged {warned}')
    check('the standard path is not warned at', quiet,
          'it honours the flag, so there is nothing to say')
    check('the conditioning dict is built either way', cond == {'P1': [(10, 25.0)]},
          f'got {cond}')


# ---------------------------------------------------------------------------
def test_structures_resolve_by_accession_not_filename_substring():
    """
    A 6-character UniProt accession is a legal prefix of a 10-character one. Matching
    the structure guide on `pdb_filename.str.contains` therefore pulled the longer
    protein's fragments into the shorter one, while structure_length -- which sizes the
    mask check -- resolved by the uniprot_id column and did not.
    """
    print('\n26. Structures are resolved by the accession column')
    import utils

    tmp = tempfile.mkdtemp()
    guide = os.path.join(tmp, 'pdb_pae_file_pos_guide.tsv')
    # A0A024 is one 4-residue fragment; A0A024R161, whose accession it prefixes, is a
    # 400-residue protein in two fragments.
    pd.DataFrame({
        'pdb_filename': ['AF-A0A024-F1.pdb.gz', 'AF-A0A024R161-F1.pdb.gz',
                         'AF-A0A024R161-F2.pdb.gz'],
        'pae_filename': ['AF-A0A024-F1.json.gz', 'AF-A0A024R161-F1.json.gz',
                         'AF-A0A024R161-F2.json.gz'],
        'pos_covered': ['[1, 4]', '[1, 200]', '[201, 400]'],
        'uniprot_id': ['A0A024', 'A0A024R161', 'A0A024R161'],
    }).to_csv(guide, sep='\t', index=False)

    sizes = {'AF-A0A024-F1': 4, 'AF-A0A024R161-F1': 200, 'AF-A0A024R161-F2': 200}

    def _n_rows(path, args):
        # The real readers take an optional 1-based inclusive (i, j) sub-range, and the
        # multi-fragment branch relies on it. A stub that ignored it would make the
        # unfixed code raise a broadcast error instead of returning its wrong shape.
        n = sizes[os.path.basename(path).split('.')[0]]
        if len(args) > 1:
            return args[1] - args[0] + 1
        if len(args) > 0:
            return n - args[0] + 1
        return n

    def fake_distances(pdb_file, *args):
        n = _n_rows(pdb_file, args)
        return np.zeros((n, n))

    def fake_paes(pae_file, *args):
        n = _n_rows(pae_file, args)
        return np.zeros((n, n))

    for name in sizes:
        open(os.path.join(tmp, f'{name}.json.gz'), 'a').close()

    originals = (utils.get_pairwise_distances, utils.get_paes)
    utils.get_pairwise_distances = fake_distances
    utils.get_paes = fake_paes
    try:
        dist = utils.get_distance_matrix_structure(guide, tmp, 'A0A024')
        pae = utils.get_pae_matrix_structure(guide, tmp, 'A0A024')
        long_dist = utils.get_distance_matrix_structure(guide, tmp, 'A0A024R161')
    finally:
        utils.get_pairwise_distances, utils.get_paes = originals

    n_guide = utils.structure_length(guide, 'A0A024')
    check('the short accession gets only its own structure',
          dist is not None and dist.shape == (4, 4),
          f'got {None if dist is None else dist.shape}; the substring match took the '
          'multi-fragment branch and sized it 400')
    check('its PAE matrix is sized to match',
          pae is not None and pae.shape == (4, 4),
          f'got {None if pae is None else pae.shape}')
    check('and the adjacency agrees with structure_length',
          dist is not None and dist.shape[0] == n_guide,
          f'{None if dist is None else dist.shape[0]} vs {n_guide}; a disagreement is '
          'the check_keep_mask ValueError that drops the gene from a round')
    check('the longer accession still gets both its fragments',
          long_dist is not None and long_dist.shape == (400, 400),
          f'got {None if long_dist is None else long_dist.shape}')
    check('which is what structure_length reports for it',
          long_dist is not None
          and long_dist.shape[0] == utils.structure_length(guide, 'A0A024R161'))


# ---------------------------------------------------------------------------
def test_a_failed_annotation_is_not_cached_forever():
    """
    _annotate_and_save returns an existing file verbatim, on a path that prints
    nothing. So writing an empty table after a failed fetch suppresses that protein's
    annotations in every later run -- the failure mode the PR fixed one layer down in
    get_functional_sites_detailed, which this layer reintroduced.
    """
    print('\n27. An empty feature table is not cached')
    import visualize_and_interpret as vi

    tmp = tempfile.mkdtemp()
    features_path = os.path.join(tmp, 'P1_features.tsv')
    cache_dir = os.path.join(tmp, 'cache')

    originals = {n: getattr(vi, n) for n in
                 ('resolve_family_members', 'aggregate_annotations_across_family',
                  'generate_simple_feature_table')}
    # A family lookup that failed: resolve_family_members swallows the exception and
    # returns [], so the pipeline runs to completion and produces nothing.
    vi.resolve_family_members = lambda *a, **k: []
    vi.aggregate_annotations_across_family = lambda *a, **k: pd.DataFrame()
    try:
        vi.generate_simple_feature_table = lambda *a, **k: pd.DataFrame()
        empty = vi._annotate_and_save('P1', features_path, cache_dir)
        check('a failed fetch still returns a frame', empty is not None and empty.empty)
        check('but writes no file, so the next run retries',
              not os.path.exists(features_path),
              'a header-only TSV here is read back as "this protein has no sites"')

        # The retry, now that UniProt is answering.
        real = pd.DataFrame({'aa_pos': [10], 'feature_type': ['BINDING']})
        vi.generate_simple_feature_table = lambda *a, **k: real
        got = vi._annotate_and_save('P1', features_path, cache_dir)
        check('a successful fetch is cached', os.path.exists(features_path))
        check('and returns its rows', len(got) == 1)

        # And the cache is used: a third call must not re-run the pipeline.
        vi.generate_simple_feature_table = lambda *a, **k: pd.DataFrame()
        cached = vi._annotate_and_save('P1', features_path, cache_dir)
        check('a cached table is read back rather than refetched', len(cached) == 1,
              f'got {len(cached)} rows')
    finally:
        for name, fn in originals.items():
            setattr(vi, name, fn)


# ---------------------------------------------------------------------------
def test_every_uniprot_call_has_a_timeout():
    """
    A hung connection to rest.uniprot.org otherwise stalls a whole run_all with no
    output. The per-protein handler in run_all turns a raised timeout into a warning
    and moves on; it cannot do anything about a socket that never returns.
    """
    print('\n28. Every UniProt request carries a timeout')
    import ast
    import inspect
    import uniprot_annotation

    tree = ast.parse(inspect.getsource(uniprot_annotation))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'get'
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'requests']
    check('the module still makes UniProt calls', len(calls) >= 5, f'{len(calls)} found')
    missing = [c.lineno for c in calls
               if not any(k.arg == 'timeout' for k in c.keywords)]
    check('every one of them passes a timeout', not missing,
          f'uniprot_annotation.py lines {missing} do not')


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
    test_rejected_gene_is_not_masked_or_rescanned()
    test_lead_columns_are_never_mislabelled()
    test_cmh_remove_nbhd_is_per_protein()
    test_gene_name_matches_the_results_table()
    test_mu_fdr_is_the_max_over_both_tests()
    test_old_results_tables_still_read()
    test_get_nbhd_finds_the_recorded_radius()
    test_failed_rescan_leaves_the_round_pool()
    test_genes_that_stop_being_rescanned_stay_out_of_the_pool()
    test_cmh_warns_that_it_ignores_the_conditional_radius()
    test_structures_resolve_by_accession_not_filename_substring()
    test_a_failed_annotation_is_not_cached_forever()
    test_every_uniprot_call_has_a_timeout()
    print('\n' + ('ALL TESTS PASSED' if not FAILURES else f'FAILURES: {FAILURES}'))
    sys.exit(1 if FAILURES else 0)
