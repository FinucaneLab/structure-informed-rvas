"""Value tracking: each mode logs its filtering funnel, and the counts never go up.

Run directly: python tests/test_value_tracking.py
Or under pytest: pytest tests/test_value_tracking.py

The per-protein statistics are stubbed out: what is tested is which count lines each
mode writes and that every one of them is consistent with the step before it.
"""

import logging
import os
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import value_tracking as vt

FUNNEL_FIELDS = ('vars', 'res', 'nbhds_max', 'prots')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        if hasattr(record, 'counts'):
            self.lines.append((record.step, dict(record.counts)))


@contextmanager
def captured():
    h = _Capture()
    vt.logger.addHandler(h)
    try:
        yield h.lines
    finally:
        vt.logger.removeHandler(h)


@contextmanager
def patched(module, **attrs):
    old = {k: getattr(module, k) for k in attrs}
    for k, v in attrs.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def make_reference_dir(root):
    """Sequence guide (P1 100 aa, P2 50, P3 80) plus common-variant and LCR tables."""
    ref = os.path.join(root, 'ref')
    os.makedirs(ref)
    pd.DataFrame({'uniprot_id': ['P1', 'P2', 'P3'],
                  'gene_name': ['G1', 'G2', 'G3'],
                  'sequence': ['A' * 100, 'A' * 50, 'A' * 80]}) \
        .to_csv(os.path.join(ref, 'protein_sequence_guide.tsv'), sep='\t', index=False)
    pd.DataFrame({'uniprot_id': ['P1'], 'aa_pos': [10], 'aa_ref': ['A'], 'aa_alt': ['V']}) \
        .to_csv(os.path.join(ref, 'common_variants_uniprot.tsv'), sep='\t', index=False)
    pd.DataFrame({'uniprot_id': ['P2'], 'aa_pos': [5]}) \
        .to_csv(os.path.join(ref, 'lcr_positions_uniprot.tsv'), sep='\t', index=False)
    return ref


def toy_rvas():
    """P1 and P2 pass a 5-allele gate; P3 has too few alleles."""
    rows = []
    for pos in (10, 11, 12, 20, 30):
        rows.append(('P1', pos, 'A', 'V', 3, 2))
    rows.append(('P1', 11, 'A', 'G', 1, 1))          # second variant at a residue
    rows.append(('P1', 99, 'A', 'V', 5000, 5000))    # common by AC
    for pos in (5, 6, 7, 8, 45):
        rows.append(('P2', pos, 'A', 'V', 2, 2))
    rows.append(('P3', 1, 'A', 'V', 1, 0))
    return pd.DataFrame(rows, columns=['uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt',
                                       'ac_case', 'ac_control'])


def assert_funnel(lines, expected_steps=None):
    """Counts must never increase from one line to the next (where both have them)."""
    assert lines, 'no count lines were logged'
    for (s0, c0), (s1, c1) in zip(lines, lines[1:]):
        for f in FUNNEL_FIELDS:
            if f in c0 and f in c1:
                assert c1[f] <= c0[f], f'{f} went up from "{s0}" ({c0[f]}) to "{s1}" ({c1[f]})'
    if expected_steps is not None:
        steps = [s for s, _ in lines]
        for want in expected_steps:
            assert any(want in s for s in steps), f'no line for "{want}" in {steps}'


# ---------------------------------------------------------------------------
# the helper itself
# ---------------------------------------------------------------------------

def test_log_counts_fields_and_file():
    with tempfile.TemporaryDirectory() as root:
        ref = make_reference_dir(root)
        results = os.path.join(root, 'results')
        vt.setup_value_tracking(results, ref, run_info={'seed': 7})
        try:
            with captured() as lines:
                vt.log_counts('step A', toy_rvas())
                vt.log_counts('nothing to count', None)          # e.g. --fdr-only
                vt.log_counts('exact', toy_rvas(), n_nbhds=12)
                with vt.paused():
                    vt.log_counts('hidden', toy_rvas())
        finally:
            vt.setup_value_tracking(None)   # detach the file handler
        assert [s for s, _ in lines] == ['step A', 'exact'], lines
        c = lines[0][1]
        assert c == {'vars': 13, 'res': 12, 'nbhds_max': 230, 'prots': 3}, c
        assert lines[1][1]['nbhds'] == 12 and 'nbhds_max' not in lines[1][1]

        text = open(os.path.join(results, vt.LOG_FILENAME)).read()
        assert '[run]' in text and 'seed=7' in text
        assert '[counts] step A | vars=13 res=12 nbhds<=230 prots=3' in text
        assert 'hidden' not in text


# ---------------------------------------------------------------------------
# per mode
# ---------------------------------------------------------------------------

def test_map_and_filter_funnel():
    import run
    with tempfile.TemporaryDirectory() as root:
        ref = make_reference_dir(root)
        vt.setup_value_tracking(None, ref)
        with patched(run, map_to_protein=lambda *a, **k: toy_rvas()), captured() as lines:
            df, _ = run.map_and_filter_rvas(
                'input.tsv', None, 'Variant ID', 'ac_case', 'ac_control', ref,
                'P1,P2', 'hg38', None, 100, False, False, None)
    assert_funnel(lines, ['Filter by total AC', 'Remove common variants',
                          'low complexity', 'Restrict to requested proteins'])
    # Removed: P1:99 (AC), P1:10 (common), P2:5 (LCR), P3 (not requested).
    assert lines[-1][1]['prots'] == 2 and lines[-1][1]['vars'] == len(df) == 9


def test_map_to_protein_logs_input_and_join():
    """The join to the missense reference is usually the largest drop of the run."""
    import read_data as rd
    inp = pd.DataFrame({'Variant ID': ['chr1-100-A-G', 'chr1-200-C-T', 'chr1-300-G-A',
                                       'chr2-50-T-C'],
                        'ac_case': [1, 2, 3, 4], 'ac_control': [1, 1, 1, 1]})
    # chr1-100 maps to two proteins, chr1-200 to one, chr1-300 to none; chr2 is absent.
    ref = pd.DataFrame({'uniprot_id': ['P1', 'P2', 'P1'], 'aa_pos': [10, 4, 20],
                        'aa_ref': 'A', 'aa_alt': 'V', 'pdb_filename': 'f',
                        'aa_pos_file': [10, 4, 20]},
                       index=['chr1-100-A-G', 'chr1-100-A-G', 'chr1-200-C-T'])
    load = lambda path, chrom, pos: ref if chrom == 'chr1' else None
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, 'in.tsv')
        inp.to_csv(path, sep='\t', index=False)
        vt.setup_value_tracking(None, make_reference_dir(root))
        with patched(rd, load_ref_for_chrom=load), captured() as lines:
            out = rd.map_to_protein(path, 'Variant ID', 'ac_case', 'ac_control', root)
    steps = dict(lines)
    assert steps['Input RVAS data']['vars'] == 4
    joined = steps['Join to missense reference (rows are variant x protein)']
    assert joined['vars'] == len(out) == 3 and joined['n_variant_ids'] == 2, joined
    assert joined['prots'] == 2


def test_scan_test_funnel():
    import scan_test as st
    df = toy_rvas()
    fdr = pd.DataFrame({'uniprot_id': ['P1'] * 100 + ['P2'] * 50, 'aa_pos': list(range(1, 101)) + list(range(1, 51))})
    with tempfile.TemporaryDirectory() as root:
        ref = make_reference_dir(root)
        vt.setup_value_tracking(None, ref)
        # P2's position 45 lies outside a 40-residue structure.
        batch = lambda *a, **k: ([('P2', 40, [45])], {'P1', 'P2'})
        with patched(st, _process_proteins_batch=batch, compute_fdr=lambda *a, **k: fdr), \
                captured() as lines:
            st.scan_test(df, ref, 15, 15, root, 10, False, False, 0.05, None, False,
                         'fdr.tsv', 'p.h5', None, min_alleles=5)
    assert_funnel(lines, ['case and control alleles', 'Scanned', 'Neighborhoods tested'])
    scanned = dict(lines)['Scanned (skipped genes and out-of-structure variants removed)']
    assert scanned['prots'] == 2 and scanned['vars'] == 11, scanned   # P2 pos 45 dropped
    assert lines[-1][1]['nbhds'] == 150


def test_cmh_funnel():
    import scan_test_cmh as cmh
    df = toy_rvas().assign(stratum='s1')
    fdr = pd.DataFrame({'uniprot_id': ['P1'] * 100, 'aa_pos': range(1, 101)})

    def fake_pvals(df, uniprot_id, **k):
        if uniprot_id == 'P2':
            raise ValueError('aa_pos outside structure')     # skipped gene
        return pd.DataFrame(), None

    with tempfile.TemporaryDirectory() as root:
        ref = make_reference_dir(root)
        vt.setup_value_tracking(None, ref)
        # --remove-nbhd on P1 at residue 10: with an identity adjacency only residue
        # 10 is dropped, which holds one variant.
        with patched(cmh, compute_all_cmh_pvals=fake_pvals,
                     write_cmh_pvals=lambda *a, **k: None,
                     get_adjacency_matrix=lambda *a, **k: np.eye(100, dtype=int),
                     compute_fdr=lambda *a, **k: fdr), captured() as lines:
            cmh.scan_test_cmh(df, ref, 15, 15, root, 10, False, False, 0.05,
                              pd.DataFrame({'uniprot_id': ['P1', 'P2']}), False,
                              'fdr.tsv', 'p.h5', {'P1': [(10, 15)]}, 'stratum',
                              min_alleles=5)
    assert_funnel(lines, ['case and control alleles', '--df-filter', 'Scanned',
                          'Neighborhoods tested'])
    scanned = dict(lines)['Scanned (skipped genes and --remove-nbhd variants removed)']
    assert scanned['prots'] == 1 and scanned['skipped_invalid_data'] == 1, scanned
    assert scanned['vars'] == 6 and scanned['variants_conditioned_out'] == 1, scanned
    assert lines[-1][1]['nbhds'] == 100


def _mu_inputs():
    df = toy_rvas()[['uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt', 'ac_case']].copy()
    df['ac_control'] = 0
    df['mu'] = 1e-8
    return df


def test_mutation_rate_funnel():
    import mutation_rate_test as mrt
    df = _mu_inputs()

    def fake_pvals(df, pdb_guide, pdb_dir, pae_dir, uniprot_id, *a, **k):
        if uniprot_id == 'P2':   # a 40-residue structure: position 45 falls outside
            k['out_of_structure'].append(('P2', 40, [45]))
            df = df[df.aa_pos <= 40]
        out = pd.DataFrame({'original_case': [int(df.ac_case.sum())],
                            'original_mu': [float(df.mu.sum())]})
        return out, out, None, 1.0

    def fake_fdrs(results_dir, fdr_cutoff, fdr_filter, *a, **k):
        uids = list(fdr_filter.uniprot_id)
        return pd.DataFrame({'uniprot_id': np.repeat(uids, 3), 'aa_pos': [1, 2, 3] * len(uids),
                             'fdr': 0.5, 'fdr_poisson': 0.5, 'fdr_binomial': 0.5,
                             'sig_fwer_both': False})

    with tempfile.TemporaryDirectory() as root:
        ref = make_reference_dir(root)
        vt.setup_value_tracking(None, ref)
        with patched(mrt, compute_all_pvals_mu=fake_pvals,
                     write_df_pvals_mu=lambda *a, **k: None,
                     estimate_lambda=lambda *a, **k: 1.0,
                     _compute_both_fdrs=fake_fdrs), captured() as lines:
            mrt.mutation_rate_scan_test(
                df, ref, 15, 15, root, 10, False, False, 0.05, None, 'fdr.tsv', 'p.h5',
                'none', None, 5, 100, seed=1, mu_from_input=True, max_residues=10000)
    assert_funnel(lines, ['de novos', 'rate table', '--max-residues', 'Scanned',
                          'Neighborhoods tested'])
    for step, c in lines:
        assert 'n_denovo' in c and 'sum_mu' in c, step
    # P1 and P2 clear --min-denovo 5; P3 has one de novo.
    assert lines[0][1]['prots'] == 2
    scanned = dict(lines)['Scanned (skipped genes and out-of-structure variants removed)']
    assert scanned['vars'] == 11 and scanned['n_denovo'] == 5016 + 8, scanned
    assert lines[-1][1]['nbhds'] == 6 and lines[-1][1]['vars'] == 11


def test_annotation_funnel():
    import annotation_test_permutation as atp
    df = toy_rvas()
    ann = pd.DataFrame({'uniprot_id': ['P1', 'P1', 'P3'], 'aa_pos': [10, 20, 1],
                        'concept_type': ['Domain'] * 3, 'concept_value': ['a', 'b', 'c'],
                        'radius': 0.0})

    def fake_protein(uniprot_id, prot_rvas_df, prot_ann_df, *a, **k):
        # 'a' and 'b' cover the same positions and are merged into one unit; their
        # radius > 0 expansion is skipped as if the protein had no structure.
        k['stats'].setdefault('not_expanded', {})[uniprot_id] = 2
        return pd.DataFrame({'uniprot_id': [uniprot_id], 'annotation_id': ['Domain_a|Domain_b']})

    def fake_summary(results, *a, **k):
        return pd.concat(list(results), ignore_index=True)

    with tempfile.TemporaryDirectory() as root:
        ref = make_reference_dir(root)
        vt.setup_value_tracking(None, ref)
        with patched(atp, process_protein=fake_protein,
                     summarize_annotation_test=fake_summary), captured() as lines:
            atp.run_annotation_test(df, ann, reference_dir=ref, results_dir=root,
                                    df_filter=pd.DataFrame({'uniprot_id': ['P2']}),
                                    min_alleles=5, n_perm=10)
    assert_funnel(lines, ['remove excluded', 'case and control alleles',
                          'proteins with annotations', 'annotations tested'])
    assert all('nbhds_max' not in c for _, c in lines), 'annotations have no residue bound'
    built = dict(lines)['Annotation test: proteins with at least one annotation built']
    assert built['n_annotation_ids'] == 2 and built['no_annotations_left'] == 0, built
    assert lines[-1][1]['nbhds'] == 1 and lines[-1][1]['prots'] == 1
    assert lines[-1][1]['prots_not_expanded_no_structure'] == 1
    assert lines[-1][1]['annotations_not_expanded_no_structure'] == 2


def test_skip_reasons_are_collected():
    """Every way the basic scan's batch can drop a gene lands under its own reason."""
    import scan_test as st

    def one_protein(df, guide, pdb, pae, results, uniprot_id, *a, **k):
        if uniprot_id == 'P1':
            raise FileNotFoundError('no pdb')
        if uniprot_id == 'P2':
            raise vt.GeneSkipped('alleles outside structure', 'too_few_alleles_in_structure')

    stats = {}
    with patched(st, scan_test_one_protein=one_protein):
        _, scanned = st._process_proteins_batch(
            toy_rvas(), ['P1', 'P2', 'P3'], 'ref', 15, 15, 'out', 3, None, 'p.h5',
            min_alleles=5, stats=stats)
    assert scanned == set()
    assert vt.stats_fields(stats) == {
        'skipped': 3,
        'skipped_missing_file': 1,
        'skipped_too_few_alleles_after_conditioning': 1,   # P3: 1 case, 0 control
        'skipped_too_few_alleles_in_structure': 1,
    }, stats


def test_neutralized_centres_are_counted():
    """Masked centres stay in the FDR pool at p = 1, so they are not visible in the
    neighborhood count; the scan reports them separately."""
    import scan_test as st
    adj = np.eye(10, dtype=int)                    # each residue is its own neighborhood
    keep = np.ones(10, bool)
    keep[:2] = False                               # residues 1 and 2 conditioned out
    df = pd.DataFrame({'aa_pos': range(3, 11), 'ac_case': 1, 'ac_control': 1})
    stats = {}
    with patched(st, get_adjacency_matrix=lambda *a, **k: adj):
        st.compute_all_pvals(df, 'g', 'p', 'pa', 'P1', n_sims=3, radius=15,
                             pae_cutoff=15, seed=0, keep_residues=keep, stats=stats)
    assert stats['neutralized'] == {'P1': 2}, stats
    assert vt.stats_fields(stats, ['P1']) == {'centres_neutralized': 2}
    assert vt.stats_fields(stats, ['P2']) == {'centres_neutralized': 0}


def test_conditional_rescans_do_not_relog_the_funnel():
    """The re-scans re-run gene selection on the full input; only the round lines
    should reach the log."""
    import inspect
    import conditional_scan as cs
    src = inspect.getsource(cs.conditional_scan)
    assert src.count('with paused():') == 2, 'both re-scan calls must be paused'
    with captured() as lines:
        cs._track_round(2, 'genes re-scanned', ['P1', 'P2'], masked_residues=7)
    assert lines == [('Conditional round 2: genes re-scanned',
                      {'prots': 2, 'round': 2, 'masked_residues': 7})], lines


if __name__ == '__main__':
    n_run = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            print(f'  {name}')
            fn()
            n_run += 1
    assert n_run > 0, 'the test_* scan matched nothing'
    print(f'ok ({n_run} tests)')
