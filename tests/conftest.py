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
