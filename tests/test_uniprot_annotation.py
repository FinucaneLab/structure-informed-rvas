"""
The annotation pipeline, exercised without the network.

This file exists because splitting uniprot_annotation.py out of
visualize_and_interpret.py dropped two names the moved code needs -- `re` and
positions_to_pymol_notation -- and nothing caught it. Every run that was supposed to
check the split found a cached *_features.tsv and printed "Loading existing features",
so the moved code never executed; and in run_all the failure is swallowed by a broad
`except Exception`, which turns a NameError into the much milder-looking line
"Warning: annotation failed" once per protein.

So these tests call the pure functions directly, with no cache to short-circuit them
and no `except` to soften them. The network-facing functions are left alone; what
broke was never the network.

Run: python tests/test_uniprot_annotation.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uniprot_annotation as ua

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
def test_family_name_parsing():
    """extract_family_name uses re, which the split left behind."""
    print('\n1. Family name parsing')
    cases = [
        ('Belongs to the Kv1 potassium channel family', 'Kv1 potassium channel'),
        ('Belongs to the protein kinase superfamily', 'protein kinase'),
        ('Contains no family statement at all', None),
        (None, None),
    ]
    for text, want in cases:
        got = ua.extract_family_name(text)
        check(f'{str(text)[:42]!r} -> {want!r}', got == want, f'got {got!r}')


# ---------------------------------------------------------------------------
def test_feature_table():
    """generate_simple_feature_table uses positions_to_pymol_notation, likewise."""
    print('\n2. Feature table')
    rows = []
    for pos in (10, 11, 12, 20):
        rows.append({'query_position': pos, 'source_protein': 'P2',
                     'feature_type': 'Binding site', 'feature_description': 'Ligand: ATP',
                     'alignment_confidence': 0.95})
    rows.append({'query_position': 50, 'source_protein': 'P3', 'feature_type': 'Domain',
                 'feature_description': 'Kinase', 'alignment_confidence': 0.5})

    out = ua.generate_simple_feature_table(pd.DataFrame(rows), confidence_threshold=0.8)
    check('low-confidence transfers are filtered out', len(out) == 1, f'{len(out)} rows')

    row = out.iloc[0]
    check('positions are collapsed into PyMOL ranges',
          row.residues_pymol == '10-12+20', row.residues_pymol)
    check('num_positions counts residues', row.num_positions == 4, str(row.num_positions))
    check('the source protein is recorded', row.source_uniprot_ids == 'P2')
    check('binding sites are flagged as major', bool(row.is_major_functional_site))

    empty = ua.generate_simple_feature_table(pd.DataFrame(rows), confidence_threshold=0.99)
    check('an empty result keeps its columns rather than raising',
          len(empty) == 0 and 'residues_pymol' in empty.columns, str(list(empty.columns)))


# ---------------------------------------------------------------------------
def test_alignment_transfer():
    """Transfer between two sequences, scored by local alignment quality."""
    print('\n3. Annotation transfer')
    query = 'MKVLAAGIVGLNLGGS'
    target = 'MKVLAAGIVGLNLGGS'          # identical, so position 5 maps to position 5
    annotations = {'5': [{'position': 5, 'feature_type': 'Active site',
                          'description': 'Proton acceptor'}]}

    out = ua.transfer_all_annotations(query, target, annotations)
    check('the annotation transfers', len(out) == 1, f'{len(out)} transfers')
    if out:
        t = out[0]
        check('onto the matching position', t['query_position'] == 5, str(t['query_position']))
        check('with full confidence on an identical sequence',
              t['alignment_confidence'] > 0.99, f"{t['alignment_confidence']:.3f}")

    # An unrelated target should still align, but score the transfer poorly.
    far = ua.transfer_all_annotations(query, 'WWWWWWWWWWWWWWWW', annotations)
    worst = max((t['alignment_confidence'] for t in far), default=0.0)
    check('an unrelated target transfers nothing confident', worst < 0.8, f'{worst:.3f}')


# ---------------------------------------------------------------------------
def test_module_imports_cleanly():
    """
    Every name the module body uses must resolve.

    The split broke two of these, and both failures only appear when the function runs.
    """
    print('\n4. Every global each function references resolves')
    import ast
    import builtins

    tree = ast.parse(open(ua.__file__).read())
    module_names = set(dir(ua)) | set(dir(builtins))

    def bound_in(node):
        """Names a function binds locally: args, assignments, imports, loops, with, except."""
        names = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
                names.add(arg.arg)
            for extra in (a.vararg, a.kwarg):
                if extra:
                    names.add(extra.arg)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                names.add(sub.id)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for alias in sub.names:
                    names.add((alias.asname or alias.name).split('.')[0])
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                names.add(sub.name)
            elif isinstance(sub, ast.comprehension):
                for t in ast.walk(sub.target):
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not node:
                names.add(sub.name)
                names |= bound_in(sub)
            elif isinstance(sub, ast.Lambda) and sub is not node:
                names |= bound_in(sub)
        return names

    unresolved = {}
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        local = bound_in(fn)
        loaded = {n.id for n in ast.walk(fn)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        missing = sorted(loaded - local - module_names)
        if missing:
            unresolved[fn.name] = missing

    # This is the check that would have caught the split: extract_family_name loads
    # `re` and generate_simple_feature_table loads positions_to_pymol_notation, neither
    # of which came across, and neither fails until the function is called.
    check('every global referenced by every function resolves', not unresolved,
          '; '.join(f'{k} needs {v}' for k, v in unresolved.items()) or
          f'{len(list(ast.walk(tree)))} nodes checked')
    for name in ('re', 'os', 'json', 'time', 'pd', 'positions_to_pymol_notation'):
        check(f'{name} resolves in the module', hasattr(ua, name))


if __name__ == '__main__':
    test_family_name_parsing()
    test_feature_table()
    test_alignment_transfer()
    test_module_imports_cleanly()
    print('\n' + ('ALL TESTS PASSED' if not FAILURES else f'FAILURES: {FAILURES}'))
    sys.exit(1 if FAILURES else 0)
