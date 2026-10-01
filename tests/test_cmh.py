import os
import sys
import unittest

import numpy as np
from scipy.stats import fisher_exact

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cmh import exact_cmh_distribution, exact_cmh_pvalue
from scan_test_cmh import _draw_stratified_null_cases


class TestExactCMH(unittest.TestCase):
    def test_one_stratum_matches_fisher(self):
        tables = [
            np.array([[1, 9], [11, 3]]),
            np.array([[5, 1], [2, 8]]),
            np.array([[0, 5], [3, 7]]),
            np.array([[2, 4], [6, 8]]),
        ]
        for tab in tables:
            a, b = tab[0]
            c, d = tab[1]
            p_cmh = exact_cmh_pvalue(
                int(a),
                [int(a + b)],
                [int(a + c)],
                [int(tab.sum())],
            )
            p_fisher = fisher_exact(tab).pvalue
            self.assertTrue(np.isclose(p_cmh, p_fisher, rtol=1e-12, atol=1e-15))

    def test_r_rabbits_example(self):
        # R stats::mantelhaen.test documentation example.
        tables = [
            np.array([[0, 6], [0, 5]]),
            np.array([[3, 3], [0, 6]]),
            np.array([[6, 0], [2, 4]]),
            np.array([[5, 1], [6, 0]]),
            np.array([[2, 0], [5, 0]]),
        ]
        observed = sum(int(t[0, 0]) for t in tables)
        inside = [int(t[0, :].sum()) for t in tables]
        cases = [int(t[:, 0].sum()) for t in tables]
        totals = [int(t.sum()) for t in tables]

        p = exact_cmh_pvalue(observed, inside, cases, totals)
        self.assertTrue(np.isclose(p, 0.03994490358126722, rtol=1e-12))

    def test_degenerate_strata_are_handled(self):
        support, pmf = exact_cmh_distribution(
            inside_totals=[0, 4, 2],
            case_totals=[0, 5, 10],
            stratum_totals=[5, 5, 10],
        )
        self.assertTrue(np.isclose(pmf.sum(), 1.0))
        self.assertEqual(len(support), 1)
        self.assertEqual(int(support[0]), 6)


class TestStratifiedNull(unittest.TestCase):
    def test_multivariate_hypergeom_preserves_margins(self):
        total = np.array([2, 1, 3, 4], dtype=int)
        n_case = 5
        n_sims = 100
        rng = np.random.default_rng(123)
        draws = _draw_stratified_null_cases(total, n_case, n_sims, rng)

        self.assertEqual(draws.shape, (len(total), n_sims))
        self.assertTrue(np.all(draws.sum(axis=0) == n_case))
        self.assertTrue(np.all(draws >= 0))
        self.assertTrue(np.all(draws <= total[:, None]))


if __name__ == '__main__':
    unittest.main()
