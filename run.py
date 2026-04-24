import argparse
import pandas as pd
import numpy as np
import os
import h5py
import glob
from scan_test import scan_test
from read_data import map_to_protein
from pymol_code import make_movie_from_pse
from logger_config import get_logger
from utils import get_nbhd_info

logger = get_logger(__name__)

def map_and_filter_rvas(
        rvas_data_to_map,
        pre_mapped_rvas,
        variant_id_col,
        ac_case_col,
        ac_control_col,
        reference_dir,
        uniprot_id,
        genome_build,
        df_filter,
        ac_filter,
        dont_remove_common,
        include_lcr,
):

    if pre_mapped_rvas is not None:
        logger.info(f"Loading pre-mapped RVAS dataframe from {pre_mapped_rvas}")
        df_rvas = pd.read_csv(pre_mapped_rvas, sep='\t')
    elif rvas_data_to_map is not None:
        df_rvas = map_to_protein(
            rvas_data_to_map,
            variant_id_col,
            ac_case_col,
            ac_control_col,
            reference_dir,
            uniprot_id,
            genome_build
        )
    else:
        df_rvas = None


    if df_rvas is not None and pre_mapped_rvas is None:
        df_rvas = df_rvas[df_rvas.ac_case + df_rvas.ac_control < ac_filter]
        if not dont_remove_common:
            logger.info("Removing common variants from RVAS data")
            keys = ['uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt']
            df_common_var = pd.read_csv(
                f'{reference_dir}/common_variants_uniprot.tsv',
                sep='\t',
                usecols = keys,
            )
            to_remove = df_common_var.set_index(keys).index
            df_rvas = df_rvas.set_index(keys)
            df_rvas = df_rvas[~df_rvas.index.isin(to_remove)]
            df_rvas = df_rvas.reset_index()

        if not include_lcr:
            lcr_path = f'{reference_dir}/lcr_positions_uniprot.tsv'
            if not os.path.exists(lcr_path):
                logger.warning(
                    f"LCR positions file not found at {lcr_path}. "
                    "Skipping LCR filter. To suppress this warning, pass --include-lcr."
                )
            else:
                logger.info("Removing variants in low complexity regions")
                keys = ['uniprot_id', 'aa_pos']
                df_lcr = pd.read_csv(lcr_path, sep='\t', usecols=keys)
                lcr_index = df_lcr.set_index(keys).index
                df_rvas = df_rvas.set_index(keys)
                df_rvas = df_rvas[~df_rvas.index.isin(lcr_index)]
                df_rvas = df_rvas.reset_index()

    # Load FDR filter if provided

    if df_filter is not None:
        filter_files = df_filter.split(',')
        def read_filter_file(f):
            df_filter = pd.read_csv(f, sep='\t')
            if 'aa_pos' in df_filter.columns:
                df_filter = df_filter[['uniprot_id', 'aa_pos']]
            else:
                df_filter = df_filter[['uniprot_id']]
            df_filter = df_filter.drop_duplicates()
            return df_filter
        df_filter = read_filter_file(filter_files[0])
        if len(filter_files) > 1:
            for f in filter_files[1:]:
                next_fdr_filter = read_filter_file(f)
                df_filter = pd.merge(df_filter, next_fdr_filter)
        uniprots_from_fdr_filter = list(df_filter[['uniprot_id']].drop_duplicates().values.flatten())
    else:
        df_filter = None

    if uniprot_id is not None:
        if os.path.exists(uniprot_id):
            uniprot_list = [x.rstrip() for x in open(uniprot_id).readlines()]
        else:
            uniprot_list = uniprot_id.split(',')
        if df_filter is not None:
            uniprot_list = list(set(uniprots_from_fdr_filter) & set(uniprot_list))
    elif df_filter is not None:
        uniprot_list = uniprots_from_fdr_filter
    else:
        uniprot_list = None
    
    if uniprot_list is not None and df_rvas is not None and pre_mapped_rvas is None:
        df_filter_uniprot = pd.DataFrame({'uniprot_id': uniprot_list})
        df_rvas = pd.merge(df_rvas, df_filter_uniprot, on='uniprot_id', how='inner')
    
    return df_rvas, df_filter


def _neighborhood_radius_type(value):
    if value in ('multiple-small', 'multiple-big'):
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid neighborhood radius: '{value}'. Must be a number, 'multiple-small', or 'multiple-big'."
        )

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
        type=_neighborhood_radius_type,
        default=15.0,
        help="neighborhood radius in Angstroms, 'multiple-small' to test radii 6,9,12,15,18,21, or 'multiple-big' to test radii 10,15,20,30,40; both combine p-values with harmonic mean",
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
        help='filter to variants with AC less than this.'
    )
    parser.add_argument(
        '--df-filter',
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
        default=None,
        help='p-value file name to save p-values to. Defaults to the fdr file name with .h5 extension.'
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
        '--pre-mapped-rvas',
        type=str,
        default=None,
        help='''
        Path to an already-mapped RVAS dataframe (TSV) produced by a previous run with
        --save-df-rvas. Skips map_to_protein and all variant-level filters (AC, common
        variants, LCR); df_filter is still processed if --df-filter is provided (for FDR).
        '''
    )
    parser.add_argument(
        '--permute',
        action='store_true',
        default=False,
        help='''
        Permute case/control allele counts within each gene (uniprot_id) before saving
        with --save-df-rvas. Within each gene the total case and control allele counts are
        preserved; individual variant counts are redrawn from the hypergeometric distribution
        (equivalent to shuffling allele labels at random). Requires --save-df-rvas.
        '''
    )
    parser.add_argument(
        '--permute-gene-filter',
        type=str,
        default=None,
        help='''
        Comma-separated list of filter files (same format as --df-filter) whose uniprot_id
        columns are intersected to define the set of genes to permute. Only variants in those
        genes are written to the --save-df-rvas output; all variants per gene are retained
        (no position-level filtering). Requires --permute.
        '''
    )
    parser.add_argument(
        '--ignore-ac',
        action='store_true',
        default=False,
        help='count every variant only once',
    )
    parser.add_argument(
        '--dont-remove-common',
        action='store_true',
        default=False,
        help='do not remove common variants from RVAS data',
    )
    parser.add_argument(
        '--include-lcr',
        action='store_true',
        default=False,
        help='include variants in low complexity regions (LCRs); by default LCR positions are excluded',
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

    # Derive pval file name from fdr file name if not explicitly set
    if args.pval_file is None:
        if args.fdr_file.endswith('.fdr.tsv'):
            base = args.fdr_file[:-len('.fdr.tsv')]
        elif args.fdr_file.endswith('.tsv'):
            base = args.fdr_file[:-len('.tsv')]
        else:
            base = args.fdr_file
        args.pval_file = base + '.pvals.h5'

    # Input validation
    
    if args.genome_build not in ['hg37', 'hg38']:
        raise ValueError(f"Invalid genome build: {args.genome_build}. Must be 'hg37' or 'hg38'")
    
    if args.neighborhood_radius not in ('multiple-small', 'multiple-big') and args.neighborhood_radius < 0:
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

    if args.permute and args.save_df_rvas is None:
        raise ValueError("--permute requires --save-df-rvas")
    if args.permute_gene_filter is not None and not args.permute:
        raise ValueError("--permute-gene-filter requires --permute")
    if args.pre_mapped_rvas is not None and args.rvas_data_to_map is not None:
        raise ValueError("--pre-mapped-rvas and --rvas-data-to-map are mutually exclusive")

    df_rvas, df_filter = map_and_filter_rvas(
        args.rvas_data_to_map,
        args.pre_mapped_rvas,
        args.variant_id_col,
        args.ac_case_col,
        args.ac_control_col,
        args.reference_dir,
        args.uniprot_id,
        args.genome_build,
        args.df_filter,
        args.ac_filter,
        args.dont_remove_common,
        args.include_lcr,
    )


    did_nothing = True

    if args.save_df_rvas is not None:
        if args.permute:
            logger.info("Permuting case/control labels within each gene (hypergeometric)")
            if args.permute_gene_filter is not None:
                filter_files = args.permute_gene_filter.split(',')
                gene_sets = [
                    set(pd.read_csv(f, sep='\t')['uniprot_id'].unique())
                    for f in filter_files
                ]
                permute_uniprot_ids = gene_sets[0].intersection(*gene_sets[1:])
                df_rvas = df_rvas[df_rvas['uniprot_id'].isin(permute_uniprot_ids)].reset_index(drop=True)
                logger.info(f"Restricting permutation to {len(permute_uniprot_ids)} genes from --permute-gene-filter")
            rng = np.random.default_rng()
            for uniprot, idx in df_rvas.groupby('uniprot_id').groups.items():
                ac_case = df_rvas.loc[idx, 'ac_case'].values.astype(int)
                ac_control = df_rvas.loc[idx, 'ac_control'].values.astype(int)
                n_each = ac_case + ac_control
                # Explode: one entry per allele across all variants, 1=case 0=control
                labels = np.repeat([1, 0], [ac_case.sum(), ac_control.sum()])
                rng.shuffle(labels)
                # Group back: count case alleles per variant
                ends = np.cumsum(n_each)
                starts = np.concatenate([[0], ends[:-1]])
                new_ac_case = np.array([labels[s:e].sum() for s, e in zip(starts, ends)])
                df_rvas.loc[idx, 'ac_case'] = new_ac_case
                df_rvas.loc[idx, 'ac_control'] = n_each - new_ac_case
            # Remove any variants that appear in more than one gene
            counts = df_rvas['Variant ID'].value_counts()
            df_rvas = df_rvas[df_rvas['Variant ID'].isin(counts[counts == 1].index)].reset_index(drop=True)
            logger.info(f"After removing multi-gene variants: {len(df_rvas)} variants")
        logger.info(f"Saving mapped RVAS dataframe to {args.save_df_rvas}")
        df_rvas.to_csv(args.save_df_rvas, sep='\t', index=False)
        did_nothing = False

    if args.fdr_only and not args.run_3dnt:
        from empirical_fdr import compute_fdr
        df_results = compute_fdr(args.results_dir, args.fdr_cutoff, df_filter, args.reference_dir, args.pval_file)
        df_results.to_csv(f'{args.results_dir}/{args.fdr_file}', sep='\t', index=False)
        did_nothing = False

    elif args.run_3dnt:
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
            df_filter,
            args.ignore_ac,
            args.fdr_file,
            args.pval_file,
            args.remove_nbhd,
        )
        did_nothing = False

    elif args.make_movie:
        if not (args.pse and args.results_dir):
            raise ValueError("For making a movie, you must provide --pse and --results_dir")
        make_movie_from_pse(args.results_dir, args.pse)
        did_nothing = False

    elif args.get_nbhd:
        if not (args.uniprot_id and args.reference_dir and args.aa_pos):
            raise ValueError("For neighborhood residue lists, you must provide --uniprot_id, --reference_dir and --aa_pos")
        if args.neighborhood_radius in ('multiple-small', 'multiple-big'):
            # Look up the per-position best radius recorded in the FDR file
            fdr_path = os.path.join(args.results_dir, args.fdr_file)
            nbhd_radius = 15.0  # fallback
            if os.path.exists(fdr_path):
                df_fdr_lookup = pd.read_csv(fdr_path, sep='\t')
                if 'radius' in df_fdr_lookup.columns:
                    row_match = df_fdr_lookup[
                        (df_fdr_lookup['uniprot_id'] == args.uniprot_id) &
                        (df_fdr_lookup['aa_pos'] == args.aa_pos)
                    ]
                    if len(row_match) > 0 and not pd.isna(row_match.iloc[0]['radius']):
                        nbhd_radius = float(row_match.iloc[0]['radius'])
        else:
            nbhd_radius = args.neighborhood_radius
        nbhd, cases, cntrls = get_nbhd_info(df_rvas, args.uniprot_id, args.aa_pos, args.reference_dir, nbhd_radius, args.pae_cutoff)
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
