import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from read_data import filter_by_total_ac, map_to_protein


class TestStratifiedMapping(unittest.TestCase):
    def test_map_to_protein_preserves_stratum(self):
        input_df = pd.DataFrame({
            'Variant ID': ['chr1-100-A-G', 'chr1-100-A-G'],
            'stratum': ['A', 'B'],
            'ac_case': [2, 0],
            'ac_control': [0, 1],
        })
        ref = pd.DataFrame({
            'ref': ['A'],
            'alt': ['G'],
            'aa_ref': ['A'],
            'aa_alt': ['V'],
            'pdb_filename': ['AF-test.pdb.gz'],
            'uniprot_id': ['PTEST'],
            'pos': [100],
            'aa_pos': [10],
            'aa_pos_file': [10],
        }, index=['chr1-100-A-G'])

        with tempfile.NamedTemporaryFile(suffix='.tsv', mode='w', delete=False) as fh:
            input_df.to_csv(fh.name, sep='\t', index=False)
            path = fh.name

        try:
            with patch('read_data.load_ref_for_chrom', return_value=ref):
                out = map_to_protein(
                    path,
                    variant_id_col=None,
                    ac_case_col=None,
                    ac_control_col=None,
                    reference_directory='unused',
                    stratum_col='stratum',
                )
            self.assertIn('stratum', out.columns)
            self.assertEqual(set(out['stratum']), {'A', 'B'})
            self.assertEqual(len(out), 2)
        finally:
            os.unlink(path)

    def test_stratified_ac_filter_sums_across_strata_not_mappings(self):
        # The same variant x stratum rows are duplicated because the genomic
        # variant maps to two proteins.  The filter must count each stratum once.
        df = pd.DataFrame({
            'Variant ID': ['v1', 'v1', 'v1', 'v1', 'v2', 'v2'],
            'stratum': ['A', 'A', 'B', 'B', 'A', 'B'],
            'uniprot_id': ['P1', 'P2', 'P1', 'P2', 'P1', 'P1'],
            'ac_case': [2, 2, 2, 2, 1, 1],
            'ac_control': [2, 2, 2, 2, 0, 0],
        })
        # v1 total AC = 4 + 4 = 8 -> removed by total AC < 6.
        # v2 total AC = 1 + 1 = 2 -> retained.
        out = filter_by_total_ac(df, ac_filter=6, stratum_col='stratum')
        self.assertEqual(set(out['Variant ID']), {'v2'})

    def test_conflicting_variant_stratum_rows_are_rejected(self):
        df = pd.DataFrame({
            'Variant ID': ['v1', 'v1'],
            'stratum': ['A', 'A'],
            'ac_case': [1, 2],
            'ac_control': [0, 0],
        })
        with self.assertRaises(ValueError):
            filter_by_total_ac(df, ac_filter=6, stratum_col='stratum')


if __name__ == '__main__':
    unittest.main()
