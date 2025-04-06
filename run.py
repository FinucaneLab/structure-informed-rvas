import argparse
import pandas as pd
from scan_test import scan_test
from read_data import map_to_protein
# from annotation_test import annotation_test


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--rvas-data-to-map',
        type=str,
        default=None,
        help='''
            .tsv.gz file with columns chr, pos, ref, alt, ac_case, ac_control.
            include exactly one of --rvas-data-to-map or --rvas-data-mapped
        ''',
    )
    parser.add_argument(
        '--rvas-data-mapped',
        type=str,
        default='None',
        help='''
            data frame that already includes uniprot canonical coordinates
            include exactly one of --rvas-data-to-map or --rvas-data-mapped.
        '''
    )
    parser.add_argument(
        '--ac-case-col',
        type=str,
        help='column with allele count in cases',
    )
    parser.add_argument(
        '--ac-control-col',
        type=str,
        help='column with allele count in controls',
    )
    parser.add_argument(
        '--pdb-filename',
        type=str,
        help='if the analysis only uses one pdb file, you can put it here instead of having a column in the input'
    )
    parser.add_argument(
        '--scan-test',
        action='store_true',
        default=False,
        help = 'perform scan test',
    )
    parser.add_argument(
        '--clinvar-test',
        action='store_true',
        default=False,
        help='perform clinvar test',
    )
    parser.add_argument(
        '--annotation_file',
        type=str,
        default=None,
        help='perform a burden test using this annotation',
    )
    parser.add_argument(
        '--neighborhood-radius',
        type=float,
        default=15.0,
        help='neighborhood radius for clinvar or annotation tests',
    )
    parser.add_argument(
        '--filter-file',
        type=str,
        default=None,
        help='file to filter variants after expanding neighborhood'
    )
    parser.add_argument(
        '--genome-build',
        type=str,
        default='hg38',
        help='genome build. must be hg38 or hg37',
    )
    parser.add_argument(
        '--which-proteins',
        type=str,
        default='all',
        help='name of a protein or file with list of proteins'
    )
    parser.add_argument(
        '--reference-dir',
        type=str,
        help='directory with reference files'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        help='directory to write results',
    )
    args = parser.parse_args()

    if args.rvas_data_to_map is not None:
        # map rvas results onto protein coordinates, linked to pdb files
        df_rvas = map_to_protein(args.rvas_data_to_map, args.which_proteins, args.genome_build)
    elif args.rvas_data_mapped is not None:
        df_rvas = pd.read_csv(args.rvas_data_mapped, sep='\t')
        if args.pdb_filename is not None:
            df_rvas['pdb_filename'] = args.pdb_filename
    else:
        raise Exception('either --rvas-data-to-map or --rvas-data-mapped must be defined')
    
    df_rvas = df_rvas.rename(columns = {
        args.ac_case_col: 'ac_case',
        args.ac_control_col: 'ac_control',
        'Uniprot_ID': 'uniprot_id',
    })    

    if args.scan_test:
        scan_test(df_rvas, args.reference_dir, args.neighborhood_radius, args.results_dir)
    
    elif args.clinvar_test:
        print('Performing ClinVar test.')
        annotation_file = f'{args.reference_dir}/ClinVar_PLP_uniprot_canonical.tsv.gz'
        filter_file = f'{args.reference_dir}/AlphaMissense_gt_0.9.tsv.gz'
        annotation_test(
            df_rvas,
            annotation_file,
            args.reference_dir,
            args.neighborhood_radius,
            filter_file,
        )

    elif args.annotation_file is not None:
        print('Performing annotation test')
        annotation_test(
            df_rvas,
            args.annotation_file,
            args.reference_dir,
            args.neighborhood_radius,
            args.filter_file,
        )
    
    else:
        raise Exception('no analysis specified')