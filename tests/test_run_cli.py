"""
The command-line surface of run.py, and the argument rules behind it.

This exists because run.py was restructured from one 870-line `if __name__` block into
build_parser() / validate_args() / main(), while two other branches were adding features
to the old shape -- the CMH test and the annotation permutation test. Reshaping someone
else's argparse block is exactly where a flag or a validation rule goes missing without
anything failing loudly: the tool still starts, still runs, and just quietly no longer
supports the thing.

So test 1 asserts every flag is present by name. It is a list, not a clever check, and
that is the point -- if a refactor drops one, this fails and names it. Test 2 does the
same for the validation rules, which are even easier to lose because nothing references
them.

Run: python tests/test_run_cli.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import build_parser, validate_args, main

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def _args(argv):
    """Parse and validate, as main() does."""
    return validate_args(build_parser().parse_args(argv))


def _rejects(argv, needle):
    """True when validation refuses this combination for the stated reason."""
    try:
        _args(argv)
        return False, 'accepted'
    except (ValueError, SystemExit) as e:
        return (needle.lower() in str(e).lower()), str(e)[:70]


def _dispatch_branch(args):
    """
    Which branch of main()'s if/elif chain this namespace selects.

    Mirrors the conditions in main() rather than calling it, so the test can assert
    which analysis would run without running one. Keep in step with main().
    """
    mutation_rate_mode = (args.mu_col is not None) or (args.mu_file is not None)
    if args.fdr_only and not args.run_3dnt:
        return 'fdr_only'
    if args.run_3dnt and args.conditional and args.test_method != 'cmh':
        return 'conditional'
    if args.run_3dnt and mutation_rate_mode:
        return 'mutation_rate'
    if args.run_3dnt and args.test_method == 'cmh':
        return 'cmh'
    if args.run_3dnt:
        return 'scan'
    return 'none'


# Enough to get past the required-argument checks without touching the filesystem.
BASE = ['--results-dir', '/tmp', '--reference-dir', '/tmp', '--fdr-file', 'x.fdr.tsv']


# ---------------------------------------------------------------------------
def test_flags_present():
    """Every flag either branch contributes must still be reachable."""
    print('\n1. CLI surface')
    options = {o for a in build_parser()._actions for o in a.option_strings}

    groups = {
        'core scan': ['--run-3dnt', '--fdr-only', '--no-fdr', '--n-sims', '--seed',
                      '--neighborhood-radius', '--pae-cutoff', '--fdr-cutoff',
                      '--min-alleles', '--include-depletion'],
        'mutation rate': ['--mu-file', '--mu-col', '--mu-residue-file', '--min-denovo',
                          '--rate-calibration', '--n-trios', '--simulate-null-from-mu'],
        'conditional': ['--conditional', '--no-conditional', '--conditional-criterion',
                        '--conditional-gene-criterion', '--conditional-max-rounds',
                        '--conditional-min-retained', '--remove-nbhd'],
        'CMH (PR #18)': ['--test-method', '--stratum-col'],
        'annotation (PR #19)': ['--annotation-file', '--annotation-format',
                                '--annotation-expand', '--annotation-id'],
        'io': ['--rvas-data-to-map', '--pre-mapped-rvas', '--save-df-rvas',
               '--df-filter', '--results-dir', '--reference-dir', '--combine-pval-files'],
    }
    for group, flags in groups.items():
        missing = [f for f in flags if f not in options]
        check(f'{group} flags present', not missing, f'missing {missing}' if missing else
              f'{len(flags)} flags')

    # Deliberately removed. If one comes back, someone has reverted a decision rather
    # than a line of code, and should know.
    for gone, why in (('--permute', 'moved to the manuscript scripts'),
                      ('--make_movie', 'superseded by pse_to_png'),
                      ('--test-direction', 'the mutation-rate null is enrichment-only')):
        check(f'{gone} stays removed ({why})', gone not in options)


# ---------------------------------------------------------------------------
def test_cmh_validation():
    """The CMH rules must survive having been moved into validate_args."""
    print('\n2. CMH argument rules (PR #18)')
    cmh = BASE + ['--run-3dnt', '--test-method', 'cmh', '--pre-mapped-rvas', '/dev/null']

    ok, msg = _rejects(cmh, 'requires --stratum-col')
    check('cmh without --stratum-col is refused', ok, msg)

    ok, msg = _rejects(cmh + ['--stratum-col', 's', '--neighborhood-radius', 'multiple-big'],
                       'one numeric')
    check('cmh with a multi-radius sweep is refused', ok, msg)

    ok, msg = _rejects(cmh + ['--stratum-col', 's', '--ignore-ac'], 'ignore-ac')
    check('cmh with --ignore-ac is refused', ok, msg)

    ok, msg = _rejects(cmh + ['--stratum-col', 's', '--mu-file'], 'mu-col/--mu-file')
    check('cmh with a mutation-rate null is refused', ok, msg)

    # Accepting the arguments is not the same as running the test, and asserting only
    # the former is how the dispatch bug got past this file: conditional analysis
    # became the default, its branch precedes the CMH branch, and `--test-method cmh`
    # silently ran the unstratified Fisher scan while this test stayed green.
    a = _args(cmh + ['--stratum-col', 'ancestry'])
    check('cmh with --stratum-col is accepted', a.test_method == 'cmh')
    check('cmh turns conditional analysis off rather than being swallowed by it',
          a.conditional is False, f'conditional={a.conditional}')
    check('so the CMH dispatch branch is the one that matches',
          _dispatch_branch(a) == 'cmh', _dispatch_branch(a))

    # and asking for both explicitly is a contradiction, not a silent downgrade
    ok, msg = _rejects(cmh + ['--stratum-col', 's', '--conditional'], 'cmh')
    check('explicit --conditional with cmh is refused', ok, msg)


# ---------------------------------------------------------------------------
def test_conditional_defaulting():
    """
    Conditional analysis is on by default but must soft-disable, not fail.

    It became a default after commands already existed that combine --fdr-only or
    --remove-nbhd with a scan. Those have to keep working; only an explicit
    --conditional alongside them is a contradiction worth refusing.
    """
    print('\n3. Conditional analysis defaulting')
    scan = BASE + ['--run-3dnt', '--pre-mapped-rvas', '/dev/null']

    check('on by default with --run-3dnt', _args(scan).conditional is True)
    check('--no-conditional turns it off', _args(scan + ['--no-conditional']).conditional is False)

    quiet = BASE + ['--fdr-only']
    check('silently off with --fdr-only', _args(quiet).conditional is False)

    ok, msg = _rejects(quiet + ['--conditional'], 'cannot be used here')
    check('explicit --conditional with --fdr-only is refused', ok, msg)

    # --no-fdr leaves conditional analysis nothing to choose hits from
    nofdr = _args(scan + ['--no-fdr'])
    check('--no-fdr turns conditional analysis off', nofdr.conditional is False)
    check('and routes to the plain scan', _dispatch_branch(nofdr) == 'scan',
          _dispatch_branch(nofdr))
    ok, msg = _rejects(scan + ['--no-fdr', '--conditional'], 'no-fdr')
    check('explicit --conditional with --no-fdr is refused', ok, msg)

    # each remaining branch routes where it should
    for argv, want in (
        (scan, 'conditional'),
        (scan + ['--no-conditional'], 'scan'),
        # --mu-col sets the mutation-rate mode without needing a rate file on disk
        (scan + ['--no-conditional', '--mu-col', 'mu'], 'mutation_rate'),
        (BASE + ['--fdr-only'], 'fdr_only'),
    ):
        got = _dispatch_branch(_args(argv))
        check(f'{" ".join(a for a in argv if a.startswith("--"))[:46]} -> {want}',
              got == want, f'got {got}')


# ---------------------------------------------------------------------------
def test_gene_filters_inclusive():
    """--min-alleles and --min-denovo default to 5 and mean "at least"."""
    print('\n4. Gene filter defaults')
    a = _args(BASE + ['--run-3dnt', '--pre-mapped-rvas', '/dev/null'])
    check('--min-alleles defaults to 5', a.min_alleles == 5, str(a.min_alleles))
    check('--min-denovo defaults to 5', a.min_denovo == 5, str(a.min_denovo))

    # The semantics live in the filters, not the parser; assert them where they are.
    import pandas as pd
    from scan_test import _filter_proteins_by_allele_count
    df = pd.DataFrame({'uniprot_id': ['exactly5', 'exactly5', 'only4', 'only4'],
                       'ac_case': [3, 2, 2, 2], 'ac_control': [3, 2, 2, 2]})
    kept = set(_filter_proteins_by_allele_count(df, None, 5))
    check('a gene with exactly 5 of each is admitted', 'exactly5' in kept, str(sorted(kept)))
    check('a gene with 4 of each is not', 'only4' not in kept, str(sorted(kept)))


# ---------------------------------------------------------------------------
def test_main_is_callable():
    """main(argv) composes the other two, so a driver script need not shell out."""
    print('\n5. Programmatic entry point')
    check('build_parser returns a parser', hasattr(build_parser(), 'parse_args'))
    check('validate_args returns the namespace',
          _args(BASE + ['--fdr-only']) is not None)

    try:
        main(['--results-dir', '/tmp'])
        check('main refuses to do nothing', False, 'returned instead of raising')
    except ValueError as e:
        check('main refuses to do nothing', 'no analysis' in str(e).lower(), str(e)[:60])
    except Exception as e:
        check('main refuses to do nothing', False, f'{type(e).__name__}: {e}')


# ---------------------------------------------------------------------------
def test_bounds_are_checked_on_the_paths_that_read_them():
    """
    A flag has to be validated wherever it is consumed, not wherever it looks at home.

    --remove-nbhd sets args.conditional False and then goes on to use two of the
    --conditional-* flags. --conditional-radius was moved out of the `if
    args.conditional:` block for exactly that reason; --conditional-min-retained was
    left behind, and it is the one that fails silently: it is not rejected downstream,
    it is obeyed.
    """
    print('\n6. Bounds on the flags --remove-nbhd still reads')
    scan = BASE + ['--run-3dnt', '--pre-mapped-rvas', '/dev/null']
    removed = scan + ['--uniprot-id', 'P1', '--remove-nbhd', '5']

    check('--remove-nbhd turns conditional analysis off',
          _args(removed).conditional is False, 'otherwise this test proves nothing')

    for bad in ('5', '-0.5'):
        ok, msg = _rejects(removed + ['--conditional-min-retained', bad],
                           'min-retained')
        check(f'--conditional-min-retained {bad} is refused with --remove-nbhd',
              ok, msg)
        ok, msg = _rejects(scan + ['--conditional-min-retained', bad], 'min-retained')
        check(f'--conditional-min-retained {bad} is refused with --conditional too',
              ok, msg)

    check('a value in range is still accepted',
          _args(removed + ['--conditional-min-retained', '0.5']).conditional_min_retained
          == 0.5)
    # The sibling that was already moved out, so a future tidy-up cannot put both back.
    ok, msg = _rejects(removed + ['--conditional-radius', '-1'], 'positive')
    check('--conditional-radius is still checked here as well', ok, msg)

    # --min-alleles 0 admits a gene with no case alleles, which read_p_values divides by.
    ok, msg = _rejects(scan + ['--min-alleles', '0'], 'min-alleles')
    check('--min-alleles 0 is refused', ok, msg)
    ok, msg = _rejects(scan + ['--min-alleles', '-1'], 'min-alleles')
    check('--min-alleles -1 is refused', ok, msg)
    check('--min-alleles 1 is accepted',
          _args(scan + ['--min-alleles', '1']).min_alleles == 1)


if __name__ == '__main__':
    test_flags_present()
    test_cmh_validation()
    test_conditional_defaulting()
    test_gene_filters_inclusive()
    test_main_is_callable()
    test_bounds_are_checked_on_the_paths_that_read_them()
    print('\n' + ('ALL TESTS PASSED' if not FAILURES else f'FAILURES: {FAILURES}'))
    sys.exit(1 if FAILURES else 0)
