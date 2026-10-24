"""
Put the repository root on sys.path for the test suite.

The tests are written in two styles. Some insert the root themselves and run as
scripts; others import the modules directly and expect to be run with the root already
importable, which works under `python -m pytest` from the root but fails under
`python tests/test_x.py` or `pytest tests/test_x.py` with a bare ModuleNotFoundError
that looks like a broken test rather than a broken invocation.

This makes both styles work from anywhere, so `pytest tests/` needs no PYTHONPATH.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Make check()-style failures actually fail under pytest.
#
# Every test module here records failures by appending to a module-level FAILURES
# list; nothing asserts, and only the `if __name__ == '__main__'` block inspects the
# list and sets an exit code. So `python tests/test_x.py` reports a real failure
# while `pytest tests/` reports the same run as passing -- the suite was inert in
# exactly the invocation conftest.py exists to support.
#
# Done here rather than by making check() assert, so that a standalone run still
# prints every check before summarizing instead of stopping at the first failure.
# ---------------------------------------------------------------------------
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    failures = getattr(getattr(item, 'module', None), 'FAILURES', None)
    before = len(failures) if isinstance(failures, list) else 0
    outcome = yield
    if not isinstance(failures, list):
        return
    new = failures[before:]
    # Leave a real exception alone: it is the more informative error.
    if new and outcome.excinfo is None:
        outcome.force_exception(AssertionError(
            f'{len(new)} check(s) failed: ' + '; '.join(str(n) for n in new)
        ))
