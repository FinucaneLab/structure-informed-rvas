import argparse
import pandas as pd
import numpy as np
import os
import h5py
import glob
from scan_test import scan_test
from read_data import map_to_protein
from pymol_code import run_all
from pymol_code import make_movie_from_pse
from logger_config import get_logger
from utils import get_nbhd_info

logger = get_logger(__name__)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--rvas-data-to-map',
        type=str,
        default=None,
        help='''
            .tsv.gz file with columns chr, pos, ref, alt, ac_case, ac_control.
            If chr/pos/ref/alt are not available, can also provide a column with 
            variant ID in chr:pos:ref:alt or chr-pos-ref-alt format with the 
            --variant-id-col flag. Can also use --ac-case-col and --ac-control-col
            flags to specify different column names for allele counts in cases
            and controls.
        ''',
    )
    parser.add_argument(
        '--variant-id-col',
        type=str,
        default=None,
        help='name of the column that has variant ID in chr:pos:ref:alt or chr-pos-ref-alt format.'
    )
    parser.add_argument(
        '--ac-case-col',
        type=str,
        help='name of the column that has allele count in cases',
    )
    parser.add_argument(
        '--ac-control-col',
        type=str,
        help='name of the column that has allele count in controls',
    )
    parser.add_argument(
        '--run-3dnt',
        action='store_true',
        default=False,
        help = 'perform the 3D neighborhood test',
    )
    parser.add_argument(
        '--neighborhood-radius',
        type=float,
        default=15.0,
        help='neighborhood radius (in Angstroms)',
    )
    parser.add_argument(
        '--pae-cutoff',
        type=float,
        default=15.0,
        help='''
        maximum PAE value for clinvar or annotation tests; argument of 0 will
        result in no PAE filtering used
        '''
    )
    parser.add_argument(
        '--n-sims',
        type=int,
        default=1000,
        help='how many null simulations to do',
    )
    parser.add_argument(
        '--genome-build',
        type=str,
        default='hg38',
        help='genome build. must be hg38 or hg37',
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
        help='filter out variants with AC greater than this.'
    )
    parser.add_argument(
        '--df-fdr-filter',
        type=str,
        default=None,
        help='''
        To consider only a subset of results and compute FDR for this subset, 
        use this flag to specify the path to a tsv with the proteins or specific
        amino acids to filter to during fdr computation. The tsv must have a 
        column called uniprot_id and can also have an aa_pos column.
        '''
    )
    parser.add_argument(
        '--no-fdr',
        action='store_true',
        default=False,
        help='skip fdr computation.'
    )
    parser.add_argument(
        '--fdr-only',
        action='store_true',
        default=False,
        help='''
        Skip everything except fdr computation. Requires that results
        directory already exists and has scan test results.
        '''
    )
    parser.add_argument(
        '--fdr-file',
        type=str,
        default='all_proteins.fdr.tsv',
        help='file in the results directory to write the fdrs to'
    )
    parser.add_argument(
        '--pval-file',
        type=str,
        default='p_values.h5',
        help='p-value file name to save p-values to.'
    )
    parser.add_argument(
        '--combine-pval-files',
        type=str,
        default=None,
        help='''
        comma-delimited list of p-value files to combine. the output file will be
        the one given by --pval-file. this is particularly useful for parallelization.
        '''
    )
    parser.add_argument(
        '--fdr-cutoff',
        type=float,
        default=0.05,
        help='fdr cutoff for summarizing results'
    )
    parser.add_argument(
        '--remove-nbhd',
        type=str,
        default=None,
        help='''
        Analogous to a conditional analysis: remove all case and control mutations in 
        these neighborhood(s) in --uniprot-id. Can be a comma-separated list of positions
        or a single position. This flag will automatically analyze only the protein 
        in --uniprot-id.
        '''
    )
    parser.add_argument(
        '--get-nbhd',
        action='store_true',
        default=False,
        help=
        '''Get list of residues and variants in neighborhood centered at --aa-pos in 
        protein --uniprot-id. Also requires --rvas-data-to-map and --reference-dir.
        '''
    )
    parser.add_argument(
        '--save-df-rvas',
        type=str,
        default=None,
        help='''
        Save the mapped RVAS dataframe. This can be run with only --rvas-data-to-map and
        --reference-dir and will perform the mapping with no additional analysis.
        '''
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
        '--uniprot-id',
        type=str,
        default=None,
        help='''
        Can be a uniprot ID, a comma-separated list of uniprot IDs,
        or a file with a list of uniprot IDs (one per line). When used with
        --3dnt, only these proteins will be analyzed. Also used with
        --visualization, --get-nbhd, --remove-nbhd, and optionall
        --save-df-rvas.
        '''
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
        '--aa-pos',
        type=str,
        default=None,
        help='Amino acid residue position in --uniprot-id for center of desired neighborhood'
    )
    args = parser.parse_args()

    # Input validation
    
    if args.genome_build not in ['hg37', 'hg38']:
        raise ValueError(f"Invalid genome build: {args.genome_build}. Must be 'hg37' or 'hg38'")
    
    if args.neighborhood_radius < 0:
        raise ValueError(f"Neighborhood radius must be non-negative, got {args.neighborhood_radius}")
    
    if args.pae_cutoff < 0:
        raise ValueError(f"PAE cutoff must be non-negative, got {args.pae_cutoff}")
    
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

    # Map and filter RVAS data
    
    if args.rvas_data_to_map is not None:
        df_rvas = map_to_protein(
            args.rvas_data_to_map,
            args.variant_id_col,
            args.ac_case_col,
            args.ac_control_col,
            args.reference_dir,
            args.uniprot_id,
            args.genome_build
        )
    else:
        df_rvas = None

    if args.uniprot_id is not None:
        if os.path.exists(args.uniprot_id):
            uniprot_id = [x.rstrip() for x in open(args.uniprot_id).readlines()]
        else:
            uniprot_id = args.uniprot_id.split(',')
    else:
        uniprot_id = None

    if df_rvas is not None:
        df_rvas = df_rvas[df_rvas.ac_case + df_rvas.ac_control < args.ac_filter]
        if uniprot_id is not None:
            df_rvas = df_rvas[df_rvas.uniprot_id.isin(uniprot_id)]

    # Load FDR filter if provided

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
    if uniprot_id is not None:
        df_fdr_filter_uniprot = pd.DataFrame({'uniprot_id': uniprot_id})
        if args.df_fdr_filter is not None:
            df_fdr_filter = pd.merge(df_fdr_filter, df_fdr_filter_uniprot)
        else:
            df_fdr_filter = df_fdr_filter_uniprot
    else:
        df_fdr_filter = None

    print('df_fdr_filter', df_fdr_filter)

    # Run analysis

    did_nothing = True

    if args.save_df_rvas is not None:
        logger.info(f"Saving mapped RVAS dataframe to {args.save_df_rvas}")
        df_rvas.to_csv(args.save_df_rvas, sep='\t', index=False)
        did_nothing = False

    if args.run_3dnt: 
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
            args.pval_file,
            args.remove_nbhd,
        )
        did_nothing = False

    elif args.visualization:
        if not (args.uniprot_id and args.reference_dir and args.results_dir):
            raise ValueError("For visualization, you must provide --uniprot_id, --reference_dir and --results_dir")
        run_all(args.uniprot_id, args.results_dir, args.reference_dir)
        did_nothing = False
    
    elif args.make_movie:
        if not (args.pse and args.results_dir):
            raise ValueError("For making a movie, you must provide --pse and --results_dir")
        make_movie_from_pse(args.results_dir, args.pse)
        did_nothing = False

    elif args.get_nbhd:
        if not (args.uniprot_id and args.reference_dir and args.aa_pos):
            raise ValueError("For neighborhood residue lists, you must provide --uniprot_id, --reference_dir and --aa_pos")
        nbhd, cases, cntrls = get_nbhd_info(df_rvas, args.uniprot_id, args.aa_pos, args.reference_dir, args.neighborhood_radius, args.pae_cutoff)
        print('Residues in neighborhood:')
        print(nbhd)
        print('Case Variants in neighborhood:')
        print(cases)
        print('Control Variants in neighborhood:')
        print(cntrls)
        did_nothing = False
    
    elif args.combine_pval_files is not None:
        if ',' in args.combine_pval_files:
            pval_files_to_combine = [os.path.join(args.results_dir,f.strip()) for f in args.combine_pval_files.split(',')]
        else:
            pattern = os.path.join(args.results_dir, args.combine_pval_files)
            pval_files_to_combine = glob.glob(pattern)

        with h5py.File(os.path.join(args.results_dir, args.pval_file), 'w') as fid_out:
            for file in pval_files_to_combine:
                with h5py.File(file, 'r') as fid_in:
                    for key in fid_in.keys():
                        if key not in fid_out:
                            fid_in.copy(key, fid_out)
                        else:
                            existing = fid_out[key][:]
                            new_data = fid_in[key][:]
                            combined = np.concatenate([existing, new_data], axis=0)
                            del fid_out[key]
                            fid_out.create_dataset(key, data=combined)
        did_nothing=False

    if did_nothing:
        raise Exception('no analysis specified')
