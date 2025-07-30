import argparse
import pandas as pd
import os
from scan_test import scan_test
from annotation_test import annotation_test
from read_data import map_to_protein
from pymol_code import run_all
from pymol_code import make_movie_from_pse
from logger_config import get_logger

logger = get_logger(__name__)


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
        '--variant-id-col',
        type=str,
        default=None,
        help='name of the column that has variant ID in chr:pos:ref:alt or chr-pos-ref-alt format.'
    )
    parser.add_argument(
        '--rvas-data-mapped',
        type=str,
        default=None,
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
        '--pae-cutoff',
        type=float,
        default=15.0,
        help='maximum PAE value for clinvar or annotation tests',
    )
    parser.add_argument(
        '--n-sims',
        type=int,
        default=1000,
        help='how many null simulations to do',
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
    parser.add_argument(
        '--ac-filter',
        type=int,
        default=5,
        help='filter out AC greater than this.'
    )
    parser.add_argument(
        '--df-fdr-filter',
        type=str,
        default=None,
        help='tsv to filter to during fdr computation. must have uniprot_id and can also have aa_pos column.'
    )
    parser.add_argument(
        '--no-fdr',
        action='store_true',
        default=False,
        help='skip fdr computation for scan test.'
    )
    parser.add_argument(
        '--fdr-only',
        action='store_true',
        default=False,
        help='only compute the scan test fdr from a directory of results'
    )
    parser.add_argument(
        '--fdr-cutoff',
        type=float,
        default=0.05,
        help='fdr cutoff for summarizing results'
    )
    parser.add_argument(
        '--ignore-ac',
        action='store_true',
        default=False,
        help='count every variant only once',
    )
    parser.add_argument(
        '--visualization',
        action='store_true',
        default=False,
        help='Run visualization tools on a specific UniProt ID'
    )
    parser.add_argument(
        '--uniprot_id',
        type=str,
        default=None,
        help='UniProt ID for visualization'
    )
    parser.add_argument(
        '--make_movie',
        action='store_true',
        default=False,
        help='make movie from a Pymol session file',
    )
    parser.add_argument(
        '--pse',
        type=str,
        default=None,
        help='Pymol session to make a movie from'
    )

    parser.add_argument(
        '--fdr-file',
        type=str,
        default='all_proteins.fdr.tsv',
        help='file in the results directory to write the fdrs to'
    )
    
    args = parser.parse_args()

    # Input validation
    if args.genome_build not in ['hg37', 'hg38']:
        raise ValueError(f"Invalid genome build: {args.genome_build}. Must be 'hg37' or 'hg38'")
    
    if args.neighborhood_radius <= 0:
        raise ValueError(f"Neighborhood radius must be positive, got {args.neighborhood_radius}")
    
    if args.pae_cutoff <= 0:
        raise ValueError(f"PAE cutoff must be positive, got {args.pae_cutoff}")
    
    if args.n_sims <= 0:
        raise ValueError(f"Number of simulations must be positive, got {args.n_sims}")
    
    if args.ac_filter <= 0:
        raise ValueError(f"AC filter must be positive, got {args.ac_filter}")
    
    if not (0 < args.fdr_cutoff < 1):
        raise ValueError(f"FDR cutoff must be between 0 and 1, got {args.fdr_cutoff}")
    
    # Check required directories exist
    if args.reference_dir and not os.path.exists(args.reference_dir):
        raise FileNotFoundError(f"Reference directory not found: {args.reference_dir}")
    
    if args.results_dir and not os.path.exists(args.results_dir):
        logger.info(f"Creating results directory: {args.results_dir}")
        os.makedirs(args.results_dir, exist_ok=True)

    if args.rvas_data_to_map is not None:
        # map rvas results onto protein coordinates, linked to pdb files
        df_rvas = map_to_protein(
            args.rvas_data_to_map,
            args.variant_id_col,
            args.ac_case_col,
            args.ac_control_col,
            args.reference_dir,
            args.which_proteins,
            args.genome_build
        )
    elif args.rvas_data_mapped is not None:
        df_rvas = pd.read_csv(args.rvas_data_mapped, sep='\t')
        df_rvas = df_rvas.rename(columns = {
            args.ac_case_col: 'ac_case',
            args.ac_control_col: 'ac_control',
            args.variant_id_col: 'Variant ID',
            'Uniprot_ID': 'uniprot_id',
        })
    else:
        df_rvas = None
    
    # Only require data input if not doing FDR-only analysis
    if df_rvas is None and not args.fdr_only:
        raise ValueError("Must provide either --rvas-data-to-map or --rvas-data-mapped")
    
    if args.pdb_filename is not None and df_rvas is not None:
        df_rvas['pdb_filename'] = args.pdb_filename

    if not args.which_proteins=='all':
        if os.path.exists(args.which_proteins):
            which_proteins = [x.rstrip() for x in open(args.which_proteins).readlines()]
        else:
            which_proteins = args.which_proteins.split(',')
        df_rvas = df_rvas[df_rvas.uniprot_id.isin(which_proteins)]

    if df_rvas is not None:
        df_rvas = df_rvas[df_rvas.ac_case + df_rvas.ac_control < args.ac_filter]

    if args.df_fdr_filter is not None:
        filter_files = args.df_fdr_filter.split(',')
        def read_filter_file(f):
            df_fdr_filter = pd.read_csv(f, sep='\t')
            if 'aa_pos' in df_fdr_filter.columns:
                df_fdr_filter = df_fdr_filter[['uniprot_id', 'aa_pos']]
            else:
                df_fdr_filter = df_fdr_filter[['uniprot_id']]
            df_fdr_filter = df_fdr_filter.drop_duplicates()
            return df_fdr_filter
        df_fdr_filter = read_filter_file(filter_files[0])
        if len(filter_files) > 1:
            for f in filter_files[1:]:
                next_fdr_filter = read_filter_file(f)
                df_fdr_filter = pd.merge(df_fdr_filter, next_fdr_filter)
    else:
        df_fdr_filter = None

    if args.scan_test: 
        logger.info("Starting scan test analysis")
        scan_test(
            df_rvas,
            args.reference_dir,
            args.neighborhood_radius,
            args.pae_cutoff,
            args.results_dir,
            args.n_sims,
            args.no_fdr,
            args.fdr_only,
            args.fdr_cutoff,
            df_fdr_filter,
            args.ignore_ac,
            args.fdr_file,
        )

    elif args.annotation_file is not None:
        logger.info('Starting annotation test analysis')
        annotation_test(
            df_rvas,
            args.annotation_file,
            args.reference_dir,
            args.neighborhood_radius,
            args.pae_cutoff,
            args.results_dir,
            args.filter_file,
        )
    
    elif args.visualization:
        if not (args.uniprot_id and args.reference_dir and args.results_dir):
            raise ValueError("For visualization, you must provide --uniprot_id, --reference_dir and --results_dir")
        run_all(args.uniprot_id, args.results_dir, args.reference_dir)
    
    elif args.make_movie:
        if not (args.pse and args.results_dir):
            raise ValueError("For making a movie, you must provide --pse and --results_dir")
        make_movie_from_pse(args.results_dir, args.pse)

    else:
        raise Exception('no analysis specified')
