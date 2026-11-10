import argparse
import pandas as pd
import numpy as np
import os
import h5py
import glob
from pathlib import Path
from scan_test import scan_test
from scan_test_cmh import scan_test_cmh, DEFAULT_SCREEN_CHI2
from read_data import map_to_protein, filter_by_total_ac
from logger_config import get_logger
from value_tracking import log_counts, setup_value_tracking
from scan_test import DEFAULT_MIN_RETAINED
from utils import (DEFAULT_CONDITIONING_RADIUS, DEFAULT_MIN_ALLELES,
                   best_radius_for_hit, get_nbhd_info)
from annotation_test_permutation import (
    add_result_fn_argument,
    normalize_result_fn,
    read_table,
    run_annotation_test,
)

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
        stratum_col,
        mu_col=None,
        mu_file=None,
        delimiter='\t',
):
    mutation_rate_mode = (mu_col is not None) or (mu_file is not None)

    def _track(step, df):
        # In mutation-rate mode the variant table's ac_case is the de novo count.
        if df is not None and mutation_rate_mode and 'ac_case' in df.columns:
            log_counts(step, df, n_denovo=int(df['ac_case'].sum()))
        else:
            log_counts(step, df)

    if pre_mapped_rvas is not None:
        logger.info(f"Loading pre-mapped RVAS dataframe from {pre_mapped_rvas}")
        df_rvas = read_table(Path(pre_mapped_rvas))
    elif rvas_data_to_map is not None:
        df_rvas = map_to_protein(
            rvas_data_to_map,
            variant_id_col,
            ac_case_col,
            ac_control_col,
            reference_dir,
            uniprot_id,
            genome_build,
            stratum_col=stratum_col,
            mu_col=mu_col,
            mu_file=mu_file,
            delimiter=delimiter,
        )
    else:
        df_rvas = None

    if pre_mapped_rvas is not None:
        _track('Load pre-mapped RVAS data', df_rvas)
    # (map_to_protein logs the input and mapping steps itself.)

    if df_rvas is not None and pre_mapped_rvas is None:
        # With a mutation-rate null there is no ac_control, so the max-AC filter
        # applies to case allele counts alone.
        if mutation_rate_mode:
            df_rvas = df_rvas[df_rvas.ac_case < ac_filter]
        else:
            df_rvas = filter_by_total_ac(
                df_rvas,
                ac_filter,
                stratum_col=stratum_col,
            )
        _track(f'Remove variants with AC >= {ac_filter} in input' if mutation_rate_mode
               else f'Remove variants with AC >= {ac_filter} in input', df_rvas)

        if not dont_remove_common:
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
            _track('Remove variants common in gnomAD', df_rvas)

        if not include_lcr:
            lcr_path = f'{reference_dir}/lcr_positions_uniprot.tsv'
            if not os.path.exists(lcr_path):
                logger.warning(
                    f"LCR positions file not found at {lcr_path}. "
                    "Skipping LCR filter. To suppress this warning, pass --include-lcr."
                )
            else:
                keys = ['uniprot_id', 'aa_pos']
                df_lcr = pd.read_csv(lcr_path, sep='\t', usecols=keys)
                lcr_index = df_lcr.set_index(keys).index
                df_rvas = df_rvas.set_index(keys)
                df_rvas = df_rvas[~df_rvas.index.isin(lcr_index)]
                df_rvas = df_rvas.reset_index()
                _track('Remove variants in low complexity regions', df_rvas)

    # Load FDR filter if provided

    if df_filter is not None:
        filter_files = df_filter.split(',')
        def read_filter_file(f):
            df_filter = pd.read_csv(f, sep='\t')
            df_filter = df_filter[[c for c in ['uniprot_id', 'aa_pos', 'aa_ref', 'aa_alt'] if c in df_filter.columns]]
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
        _track('Restrict to requested proteins (--uniprot-id/--df-filter)', df_rvas)

    return df_rvas, df_filter


def build_components_file(args, logger):
    """
    Group each gene's significant neighborhoods into components and write them.

    Reads the results table and p-value file already in the results directory, so it
    runs after a scan or on its own via --components. Returns the output path.
    """
    from components import components_for_genes, FDR_COLUMN, FWER_COLUMN
    from utils import read_original_mutation_data, resolve_pval_node

    fdr_path = os.path.join(args.results_dir, args.fdr_file)
    pval_path = os.path.join(args.results_dir, args.pval_file)
    for path in (fdr_path, pval_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'building components needs an existing results file; {path} not found'
            )
    df_results = pd.read_csv(fdr_path, sep='\t')

    # The mutation-rate results table carries per-test p-values and no p_value
    # column; the case/control table is the other way round.
    mu_results = 'p_binomial' in df_results.columns
    p_col = 'p_binomial' if mu_results else 'p_value'
    # Under multiple-* each test picks its own best radius. Test B is the test of
    # interest, so its radius defines the neighborhood.
    radius_cols = ('radius_binomial', 'radius_poisson') if mu_results else ('radius',)
    gate_col = FWER_COLUMN if args.component_gene_criterion == 'fwer' else FDR_COLUMN
    for col in (p_col, gate_col, FDR_COLUMN, FWER_COLUMN):
        if col not in df_results.columns:
            raise ValueError(
                f"Column '{col}' not found in {fdr_path}. "
                f'Available: {list(df_results.columns)}'
            )

    with h5py.File(pval_path, 'r') as fid:
        node, mu_pvals = resolve_pval_node(fid, 'binomial')
        # Mismatched pair: a mutation-rate results table read against a case/control
        # p-value file would silently take ac_control as mu, or the reverse. Both
        # produce a plausible table from the wrong column.
        if mu_pvals != mu_results:
            raise ValueError(
                f'{fdr_path} looks like a '
                f'{"mutation-rate" if mu_results else "case/control"} results table '
                f'but {pval_path} is a '
                f'{"mutation-rate" if mu_pvals else "case/control"} p-value file'
            )

        # The CMH scan writes a {uid}_cmh_chisq dataset per protein; nothing else does.
        if any(k.endswith('_cmh_chisq') for k in fid.keys()):
            logger.warning(
                'These results came from a stratified (CMH) scan. The direction filter '
                'that decides which neighborhoods may join a component compares the '
                'observed case count to its expectation under the POOLED margins, but '
                'CMH significance comes from per-stratum comparisons. Where strata '
                'differ in case fraction -- the reason to use CMH at all -- a '
                'neighborhood enriched within every stratum can fall below the pooled '
                'expectation and be dropped as depleted. Treat a thin or empty '
                'component table for a CMH run with suspicion. This goes away when '
                'the scan becomes one-sided: there is then nothing depleted to drop '
                'and the filter stops running.'
            )

        # The neighborhoods are rebuilt here, not read back: only the radius is
        # recorded in the results table, so the PAE cutoff comes from this invocation.
        # Rebuilding at a different cutoff than the scan used silently changes which
        # residues a component contains.
        logger.info(f'Rebuilding neighborhoods at --pae-cutoff {args.pae_cutoff}; '
                    'this must match the scan that produced these results.')

        def read_mut(uniprot_id):
            return read_original_mutation_data(node, uniprot_id, mu_mode=mu_pvals)

        passes_gate = df_results[gate_col] < args.component_gene_cutoff
        gated = sorted(set(df_results.loc[passes_gate, 'uniprot_id']))
        logger.info(f'{len(gated)} genes pass the gene gate '
                    f'({gate_col} < {args.component_gene_cutoff})')

        # A gene qualifies on strong evidence and is then described by everything that
        # evidence is spread across -- every FDR-significant neighborhood, UNIONED with
        # whatever cleared the gate. The union is not redundant: the two corrections are
        # not nested, so a neighborhood can clear the gate and miss --fdr-cutoff, and
        # without this the gene that qualified would be absent from its own table.
        member_mask = (df_results[FDR_COLUMN] < args.fdr_cutoff) | passes_gate

        df_components = components_for_genes(
            df_results, gated, args.reference_dir, args.pae_cutoff, read_mut,
            p_col, member_mask, radius_cols, mu_pvals, gate_col=gate_col,
        )

    stem = args.fdr_file[:-4] if args.fdr_file.endswith('.tsv') else args.fdr_file
    out_path = os.path.join(args.results_dir, f'{stem}.components.tsv')
    if not len(df_components):
        # An empty DataFrame has no columns, so to_csv writes a single newline and
        # pd.read_csv raises EmptyDataError on the way back. Write nothing instead:
        # no components is the ordinary outcome for a scan with no significant gene,
        # and an absent file says that more clearly than an unreadable one.
        logger.info('No components found; no component table written')
        return None
    df_components.to_csv(out_path, sep='\t', index=False)
    logger.info(f'Wrote {len(df_components)} components to {out_path}')
    return out_path


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
        '--mu-file',
        type=str,
        nargs='?',
        const='__reference_default__',
        default=None,
        help='''
        Path to a mutation-rate reference file (parquet) with columns chrom, pos, ref,
        alt and a rate column, inner-joined to the variant data on chr-pos-ref-alt.
        Pass the flag with no argument to use
        <reference-dir>/gnomad_mu_snp_by_variant.parquet. Switches on the
        mutation-rate null. Mutually exclusive with --mu-col.
        '''
    )
    parser.add_argument(
        '--mu-col',
        type=str,
        default=None,
        help='''
        Name of a mutation-rate column already present in --rvas-data-to-map, used
        instead of a rate reference file. Switches on the mutation-rate null.
        Mutually exclusive with --mu-file.
        '''
    )
    parser.add_argument(
        '--mu-residue-file',
        type=str,
        default=None,
        help='''
        Per-residue mutation rate table built by precompute_mu_per_residue.py, giving the
        summed rate over ALL possible missense variants at each residue. Required with
        --mu-file; defaults to <reference-dir>/mu_per_residue_gnomad.parquet. This is what
        makes the denominator the gene's real mutational opportunity rather than only the
        variants present in the de novo file.
        '''
    )
    parser.add_argument(
        '--max-residues',
        type=int,
        default=10000,
        help='''
        Skip proteins longer than this. The pairwise distance matrix scales as n_res^2,
        so titin (34,350 residues) alone would need 9.4 GB. Default 10000 skips 2 of the
        750 ASD analysis genes.
        '''
    )
    parser.add_argument(
        '--simulate-null-from-mu',
        action='store_true',
        default=False,
        help='''
        Replace the observed de novo counts with draws from the mutation rate model and
        run the full pipeline on them, as a null calibration check. P-values should be
        uniform and FDR should come out at its nominal level.
        '''
    )
    parser.add_argument(
        '--test-direction',
        type=str,
        default='enrichment',
        choices=['enrichment', 'depletion'],
        help='''
        Which tail to test. 'enrichment' (default) asks whether a neighborhood holds more
        variants than the mutation rate predicts; 'depletion' asks whether it holds fewer,
        which is the direction selection acts in and the relevant one for variant classes
        that have already been filtered by it.
        '''
    )
    parser.add_argument(
        '--min-mu-coverage',
        type=float,
        default=0.0,
        help='''
        Skip genes where the rate model covers less than this fraction of their possible
        missense variants. Default 0 (report mu_coverage but exclude nothing): coverage
        gaps cost power rather than biasing the tests, because a variant's de novo count
        and its rate are dropped together. Raise it to exclude poorly covered genes.
        '''
    )
    parser.add_argument(
        '--rate-calibration',
        type=str,
        default='global',
        help='''
        How to convert relative mutation rates into expected de novo counts for Test A
        (the absolute test). 'global' (default) estimates lambda_hat = sum(x)/sum(mu)
        over the calibration set; 'none' uses lambda_hat = 1, appropriate only when the
        rate column already holds absolute expected counts; 'fixed:<value>' supplies it
        directly. lambda_hat is always estimated AFTER variant-level filters and BEFORE
        any gene-level selection.
        '''
    )
    parser.add_argument(
        '--rate-calibration-genes',
        type=str,
        default=None,
        help='''
        Optional file (same format as --df-filter) restricting the gene set used to
        estimate lambda_hat. Defaults to the whole input file after variant-level
        filters.
        '''
    )
    parser.add_argument(
        '--min-alleles',
        type=int,
        default=DEFAULT_MIN_ALLELES,
        help='''
        Minimum case AND control alleles per gene, applied inclusively (so the
        default of 5 keeps a gene with 5 of each). Used to choose which genes to
        scan; again, after any --remove-nbhd conditioning, to decide whether enough
        is left to scan; again after out-of-structure positions are dropped; and by
        the annotation test. --min-denovo is the same floor for the mutation-rate
        test, which counts de novos rather than case and control alleles.
        '''
    )
    parser.add_argument(
        '--min-denovo',
        type=int,
        default=5,
        help='''
        Minimum de novo count per gene for the mutation-rate test, applied
        inclusively (so the default of 5 keeps a gene with 5), matching
        --min-alleles in the case/control scan.
        '''
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='''
        Random seed for the null simulations, making a run reproducible. Applies to the
        standard, mutation-rate, and CMH tests. Each protein is given its own derived
        stream, so the null draws stay independent across proteins. The stream is derived
        from the protein's position in the gene list, so reproducing a run means re-running
        the same gene set in the same order; splitting the genes across jobs differently
        changes the draws (the results remain valid, they are just not identical). If not
        given, a seed is generated and logged so that any run can be reproduced after
        the fact.
        '''
    )
    parser.add_argument(
        '--n-trios',
        type=int,
        default=None,
        help='''
        Optional number of trios, used only to report lambda_hat / (2 * n_trios) as a
        units sanity check on the mutation rate model.
        '''
    )
    parser.add_argument(
        '--run-3dnt',
        action='store_true',
        default=False,
        help = 'perform the 3D neighborhood test',
    )
    parser.add_argument(
        '--test-method',
        choices=['fisher','cmh'],
        default='fisher',
        help = 'association test used by 3DNT. Default: fisher',
    )
    parser.add_argument(
        '--stratum-col',
        type=str,
        default=None,
        help = 'categorical stratum column used by CMH',
    )
    parser.add_argument(
        '--alternative',
        choices=['two-sided', 'greater'],
        default='greater',
        help = "scan/annotation test: 'greater' = one-sided case enrichment (applied to the p-value before FWER/FDR)",
    )
    parser.add_argument(
        '--cmh-screen-threshold',
        type=float,
        default=DEFAULT_SCREEN_CHI2,
        help = 'CMH chi-square threshold for cursory screening',
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
        column called uniprot_id and can also have aa_pos, aa_ref and aa_alt
        columns. With --annotation-file, the RVAS data is restricted to these
        proteins/positions/variants before the test.
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
        '--conditional',
        action='store_true',
        default=False,
        help='''
        After the scan, repeat it with each gene's called signal masked out, to ask
        whether the gene carries more than one independent 3D signal. The unit
        conditioned on is a component, not a single neighborhood: one signal is
        reported by every window overlapping it, so masking the best window alone
        leaves the rest in place for the next round to re-find.
        '''
    )
    parser.add_argument(
        '--conditional-max-rounds',
        type=int,
        default=5,
        help='maximum conditional rounds; stops early when no gene still has signal'
    )
    parser.add_argument(
        '--conditional-criterion',
        choices=('fdr', 'fwer'),
        default='fdr',
        help='correction the MEMBER gate uses (which neighborhoods form a component)'
    )
    parser.add_argument(
        '--conditional-cutoff',
        type=float,
        default=None,
        help='member gate cutoff; defaults to --fdr-cutoff'
    )
    parser.add_argument(
        '--conditional-gene-criterion',
        choices=('fwer', 'fdr'),
        default='fwer',
        help='correction the GENE gate uses (which genes are considered at all)'
    )
    parser.add_argument(
        '--conditional-gene-cutoff',
        type=float,
        default=0.05,
        help='gene gate cutoff'
    )
    parser.add_argument(
        '--conditional-min-retained',
        type=float,
        default=DEFAULT_MIN_RETAINED,
        help='''
        A conditioned centre is not re-tested when its neighborhood keeps less than
        this fraction of itself: what is left is a different neighborhood from the
        one the scan reported. 0 still refuses a neighborhood with nothing left.
        '''
    )
    parser.add_argument(
        '--keep-round-pvals',
        action='store_true',
        default=False,
        help='''
        Keep each conditional round's p-value file. By default only round 0's is
        kept -- the round files are copies of the whole thing, tens of GB apiece on
        an exome-wide scan. The per-round results tables are kept either way.
        '''
    )
    parser.add_argument(
        '--components',
        action='store_true',
        default=False,
        help='''
        Group a gene's significant neighborhoods into components and write one row
        per component to <fdr-file stem>.components.tsv. Reads an existing results
        file and its p-value file; runs no scan. The scan centres a neighborhood on
        every residue, so one 3D signal is reported by every window that overlaps
        it; a component is the unit that is actually one finding.
        '''
    )
    parser.add_argument(
        '--no-components',
        action='store_true',
        default=False,
        help='''
        Skip the component table that a scan writes alongside its results file.
        '''
    )
    parser.add_argument(
        '--component-gene-cutoff',
        type=float,
        default=0.05,
        help='''
        A gene is clustered if it has a neighborhood below this cutoff on the
        criterion named by --component-gene-criterion (default FWER). Its components
        are then built from every neighborhood below --fdr-cutoff, so a gene
        qualifies on strong evidence but is described by everything that evidence is
        spread across.
        '''
    )
    parser.add_argument(
        '--component-gene-criterion',
        choices=('fwer', 'fdr'),
        default='fwer',
        help='correction the gene gate for --components is judged on'
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
        Analogous to a conditional analysis: mask all case and control mutations in
        these neighborhood(s) of --uniprot-id out of the scan. Masked, not deleted:
        the residues stay in place with zeroed counts (and, under the mutation-rate
        null, zeroed opportunity), so every per-residue array keeps its positions.
        Other genes in the input are still scanned, unconditioned -- this does NOT
        restrict the analysis to --uniprot-id. A comma-separated list of positions,
        or a single position; --uniprot-id must name exactly one protein, since
        residue numbering is per protein.
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
    parser.add_argument(
        '--annotation-file',
        type=str,
        default=None,
        help='perform a burden test using this annotation',
    )
    parser.add_argument(
        '--annotation-format',
        choices=['long', 'wide'],
        default='long',
        help='long = one row per annotation-position/interval (concept_type, concept_value, start, end, or aa_pos); wide = one row per residue',
    )
    parser.add_argument(
        '--annotation-expand',
        default=None,
        help="comma-separated annotation-id prefixes to expand to their 3D neighborhood using --neighborhood-radius/--pae-cutoff (e.g. 'Binding site,Active site'), or 'all'. Without it there is no expansion.",
    )
    parser.add_argument(
        '--annotation-id',
        type=str,
        default=None,
        help='''
        Used with --annotation-file. Long format: the columns joined into
        the annotation id (default concept_type,concept_value). Wide format:
        the columns used as ids (default all).
        '''
    )
    add_result_fn_argument(parser)
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

    # A seed is always chosen and recorded, so that a run with no --seed is still
    # reproducible after the fact from the logged value.
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
        logger.info(f'No --seed given; using generated seed {args.seed}')
    else:
        logger.info(f'Using seed {args.seed}')

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
    
    if args.cmh_screen_threshold < 0:
        raise ValueError(f"CMH screen threshold must be non-negative, got {args.cmh_screen_threshold}")
            
    if not (0 < args.fdr_cutoff < 1):
        raise ValueError(f"FDR cutoff must be between 0 and 1, got {args.fdr_cutoff}")

    annotation_cmh = args.annotation_file is not None and args.test_method == 'cmh'
    if annotation_cmh and args.stratum_col is None:
        raise ValueError("--test-method cmh requires --stratum-col")

    if args.run_3dnt and args.test_method == 'cmh' and not args.fdr_only:
        if args.stratum_col is None:
            raise ValueError("--test-method cmh requires --stratum-col")
        if args.neighborhood_radius in ('multiple-small', 'multiple-big'):
            raise ValueError(
                "--test-method cmh currently supports one numeric "
                "--neighborhood-radius"
            )
        if args.ignore_ac:
            raise ValueError(
                "--ignore-ac is not currently supported with --test-method cmh"
            )
        if args.mu_col is not None or args.mu_file is not None:
            raise ValueError(
                "--test-method cmh cannot be combined with --mu-col/--mu-file"
            )

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

    # Checked unconditionally: --min-alleles governs the case/control scan, so
    # validating it inside the mutation-rate block would check it in the one mode
    # that does not use it and leave it unchecked everywhere it does.
    # At least 1, not 0: the guard is `sum(ac_case) < min_alleles`, so 0 admits a
    # gene with no case alleles at all, and read_p_values then divides by that count
    # and fills the ratio column with inf and nan rather than failing.
    # --components is the head of the dispatch chain, so combining it with a scan
    # would take the components branch and silently run no scan at all -- building a
    # table from whatever stale results file happened to be in --results-dir. A scan
    # already writes components by default, so the combination is never what anyone
    # wants; refuse it rather than quietly doing the wrong one.
    if args.components and (args.run_3dnt or args.fdr_only):
        raise ValueError(
            '--components builds a component table from results that already exist, '
            'so it cannot be combined with --run-3dnt or --fdr-only. A scan writes '
            'one by default; use --no-components to suppress that.'
        )

    if args.conditional:
        # CMH has no conditional backend, --fdr-only runs no scan to condition, and
        # --no-fdr leaves the corrections the rounds choose on. Refused rather than
        # quietly downgraded: a run that silently did something else would be read as
        # a conditional analysis that found nothing.
        for flag, why in (('test_method', None), ('fdr_only', '--fdr-only runs no scan'),
                          ('no_fdr', '--no-fdr leaves no corrections to gate on')):
            if flag == 'test_method':
                if args.test_method == 'cmh':
                    raise ValueError('--conditional has no CMH backend; use the '
                                     'default --test-method fisher')
            elif getattr(args, flag):
                raise ValueError(f'--conditional cannot be combined with --{flag.replace("_", "-")}: {why}')
        if not args.run_3dnt:
            raise ValueError('--conditional needs --run-3dnt')
        if args.no_components:
            # conditional_scan needs components to decide what to mask, so it builds
            # them every round whether or not the table is wanted. Honouring the flag
            # would save nothing and silently drop the record of what was conditioned
            # on; ignoring it would contradict its own help text.
            raise ValueError(
                '--conditional cannot be combined with --no-components: components '
                'are what it conditions on, so the table is a record of the analysis '
                'rather than an optional extra.'
            )
        if args.remove_nbhd is not None:
            # conditional_scan runs its own round 0 unconditioned and derives every
            # later mask from the components it finds. Passing --remove-nbhd through
            # would be ignored, so the user would get an unconditioned round 0 that
            # re-discovers the neighborhood they asked to mask and reports it as
            # signal, with nothing to say so.
            raise ValueError(
                '--conditional cannot be combined with --remove-nbhd: it chooses what '
                'to mask itself, from the components each round finds.'
            )
        # Checked here, not inside a mode branch: --conditional-min-retained out of
        # range is not rejected downstream, it is obeyed -- 5 kills every centre and
        # the results come out empty with no error.
        if not 0 <= args.conditional_min_retained <= 1:
            raise ValueError('--conditional-min-retained must be between 0 and 1, got '
                             f'{args.conditional_min_retained}')
        for name in ('conditional_gene_cutoff', 'conditional_cutoff'):
            v = getattr(args, name)
            if v is not None and not 0 < v <= 1:
                raise ValueError(f'--{name.replace("_", "-")} must be in (0, 1], got {v}')
        if args.conditional_max_rounds < 1:
            raise ValueError('--conditional-max-rounds must be at least 1, got '
                             f'{args.conditional_max_rounds}')

    if args.min_alleles < 1:
        raise ValueError(f"--min-alleles must be at least 1, got {args.min_alleles}")

    if args.mu_col is not None and args.mu_file is not None:
        raise ValueError("--mu-col and --mu-file are mutually exclusive")

    mutation_rate_mode = (args.mu_col is not None) or (args.mu_file is not None)

    if mutation_rate_mode:
        if args.ac_control_col is not None:
            raise ValueError("--mu-col/--mu-file cannot be combined with --ac-control-col")
        if args.ignore_ac:
            raise ValueError(
                "--ignore-ac is not supported with a mutation-rate null: collapsing de novo "
                "counts to 0/1 discards recurrence, which is signal here."
            )
        if args.mu_file == '__reference_default__':
            if not args.reference_dir:
                raise ValueError("--mu-file with no argument requires --reference-dir")
            args.mu_file = os.path.join(args.reference_dir, 'gnomad_mu_snp_by_variant.parquet')
        if args.mu_file is not None and not os.path.exists(args.mu_file):
            raise FileNotFoundError(f"Mutation rate file not found: {args.mu_file}")
        if args.rate_calibration not in ('global', 'none') and not args.rate_calibration.startswith('fixed:'):
            raise ValueError(
                f"Invalid --rate-calibration: {args.rate_calibration}. "
                "Must be 'global', 'none', or 'fixed:<value>'."
            )
        if args.rate_calibration.startswith('fixed:'):
            try:
                float(args.rate_calibration.split(':', 1)[1])
            except ValueError:
                raise ValueError(f"Could not parse a number from --rate-calibration {args.rate_calibration}")
        # At least 1, like --min-alleles. Under the old exclusive rule 0 meant "at
        # least 1 de novo"; now that the comparison is inclusive, 0 is vacuously
        # true and admits every gene in the table, including genes with no observed
        # de novos, padding the FDR and FWER pools.
        if args.min_denovo < 1:
            raise ValueError(f"--min-denovo must be at least 1, got {args.min_denovo}")
        if args.mu_file is not None:
            if args.mu_residue_file is None:
                if not args.reference_dir:
                    raise ValueError("--mu-file requires --reference-dir or --mu-residue-file")
                args.mu_residue_file = os.path.join(args.reference_dir, 'mu_per_residue_gnomad.parquet')
            if not os.path.exists(args.mu_residue_file):
                raise FileNotFoundError(
                    f"Per-residue mutation rate table not found: {args.mu_residue_file}\n"
                    f"Build it with: python precompute_mu_per_residue.py "
                    f"--reference-dir {args.reference_dir}"
                )

    # Counts after every filtering step go to <results_dir>/value_tracking.log.
    setup_value_tracking(args.results_dir, args.reference_dir, run_info={'seed': args.seed})

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
        args.stratum_col,
        args.mu_col,
        args.mu_file,
    )

    if (
        (args.run_3dnt and args.test_method == 'cmh' and not args.fdr_only or annotation_cmh)
        and df_rvas is not None
    ):
        if args.stratum_col not in df_rvas.columns:
            raise KeyError(
                f"Mapped RVAS data do not contain stratum column '{args.stratum_col}'"
            )
        if df_rvas[args.stratum_col].isna().any():
            raise ValueError(
                f"Mapped RVAS data contain missing values in '{args.stratum_col}'"
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
                log_counts('Restrict permutation to --permute-gene-filter genes', df_rvas)
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
            log_counts('Remove variants in more than one gene (after permutation)', df_rvas)
        logger.info(f"Saving mapped RVAS dataframe to {args.save_df_rvas}")
        df_rvas.to_csv(args.save_df_rvas, sep='\t', index=False)
        did_nothing = False

    # --remove-nbhd names positions in ONE protein (--uniprot-id). Keyed by gene so
    # the scan backends cannot apply it to every protein in the file: with
    # --pre-mapped-rvas, --uniprot-id does not restrict df_rvas, so an unkeyed string
    # silently conditioned every gene on the same position.
    conditioning = None
    if args.remove_nbhd is not None:
        if not args.uniprot_id:
            raise ValueError('--remove-nbhd requires --uniprot-id: it names positions '
                             'in one protein, not a position shared by every protein')
        # --uniprot-id is an accession, a comma-separated list, OR a path to a file of
        # accessions -- see map_and_filter_rvas, which parses it the same way. Using
        # the raw string as the dict key means a list or a path matches no gene and
        # conditioning silently does nothing.
        if os.path.exists(args.uniprot_id):
            with open(args.uniprot_id) as fh:
                targets = [x.strip() for x in fh if x.strip()]
        else:
            targets = [x.strip() for x in args.uniprot_id.split(',') if x.strip()]
        if not targets:
            raise ValueError(f'--uniprot-id named no accessions: {args.uniprot_id!r}')
        if len(targets) > 1:
            # Residue numbering is per protein, so one position means different things
            # in different genes -- and a gene shorter than the position raises, is
            # skipped, and then drops out of the results entirely.
            raise ValueError(
                f'--remove-nbhd names positions in one protein, but --uniprot-id gave '
                f'{len(targets)} accessions ({targets[:3]}...). Condition one gene at '
                'a time.'
            )

        positions = [int(x) for x in
                     (p.strip() for p in args.remove_nbhd.split(',')) if x]
        if not positions:
            raise ValueError(f'--remove-nbhd parsed to no positions: {args.remove_nbhd!r}')

        # Under multiple-* each position was called at its own best radius, recorded in
        # the results table. Conditioning every one at a flat 15 A would leave most of a
        # 30 A or 40 A neighborhood unmasked, so the re-scan can re-call the region it
        # was supposed to have conditioned out and report it as a new signal.
        multi = isinstance(args.neighborhood_radius, str)
        df_hits = None
        if multi:
            hit_path = os.path.join(args.results_dir, args.fdr_file)
            if os.path.exists(hit_path):
                df_hits = pd.read_csv(hit_path, sep='\t')
            else:
                logger.warning(
                    f'--neighborhood-radius {args.neighborhood_radius} with no '
                    f'{hit_path} to read each hit\'s chosen radius from; conditioning '
                    f'falls back to {DEFAULT_CONDITIONING_RADIUS} A, which may leave '
                    'most of a larger neighborhood unmasked.'
                )

        conditioning = {}
        for target in targets:
            centers = []
            for pos in positions:
                if multi:
                    # Order is deliberate. A case/control table has `radius`; a
                    # mutation-rate table has none, only the per-test columns, and
                    # Test B is the test of interest (see the mutation_rate_test
                    # module docstring), so its radius defines the neighborhood.
                    radius = best_radius_for_hit(
                        df_hits, target, pos, default=DEFAULT_CONDITIONING_RADIUS,
                        radius_cols=('radius', 'radius_binomial', 'radius_poisson'))
                else:
                    radius = float(args.neighborhood_radius)
                centers.append((pos, radius))
            conditioning[target] = centers
        logger.info(f'Conditioning {len(conditioning)} gene(s) on {len(positions)} '
                    f'position(s): { {k: v for k, v in list(conditioning.items())[:3]} }')

    # Conditioning is only consumed by the scan backends. Flagged rather than left
    # silent: --remove-nbhd hard-errors on a missing or multiple --uniprot-id, so a
    # third configuration that quietly ignores it is an inconsistent contract.
    if conditioning and not (args.run_3dnt or args.fdr_only):
        logger.warning(
            '--remove-nbhd was given but this invocation runs no scan, so nothing is '
            'conditioned. It applies to --run-3dnt and --fdr-only only.'
        )

    wrote_results = False

    if args.components:
        build_components_file(args, logger)
        did_nothing = False

    elif args.fdr_only and not args.run_3dnt and mutation_rate_mode:
        wrote_results = True
        from mutation_rate_test import mutation_rate_scan_test
        mutation_rate_scan_test(
            df_rvas, args.reference_dir, args.neighborhood_radius, args.pae_cutoff,
            args.results_dir, args.n_sims, args.no_fdr, True, args.fdr_cutoff,
            df_filter, args.fdr_file, args.pval_file, args.rate_calibration,
            args.rate_calibration_genes, args.min_denovo, args.n_trios, args.seed,
            args.mu_residue_file, args.mu_col is not None, args.min_mu_coverage,
            args.max_residues, args.simulate_null_from_mu, args.test_direction,
            conditioning,
        )
        did_nothing = False

    elif args.fdr_only and not args.run_3dnt:
        wrote_results = True
        from empirical_fdr import compute_fdr
        df_results = compute_fdr(args.results_dir, args.fdr_cutoff, df_filter, args.reference_dir, args.pval_file)
        df_results.to_csv(f'{args.results_dir}/{args.fdr_file}', sep='\t', index=False)
        did_nothing = False

    elif args.conditional:
        from conditional_scan import conditional_scan
        wrote_results = True
        logger.info('Starting conditional scan analysis')
        mu_kwargs = {}
        if mutation_rate_mode:
            mu_kwargs = dict(
                rate_calibration=args.rate_calibration,
                rate_calibration_genes=args.rate_calibration_genes,
                min_denovo=args.min_denovo, n_trios=args.n_trios, seed=args.seed,
                mu_residue_file=args.mu_residue_file,
                mu_from_input=args.mu_col is not None,
                min_mu_coverage=args.min_mu_coverage, max_residues=args.max_residues,
                simulate_null=args.simulate_null_from_mu, direction=args.test_direction,
            )
        conditional_scan(
            df_rvas, args.reference_dir, args.neighborhood_radius, args.pae_cutoff,
            args.results_dir, args.n_sims, args.fdr_cutoff, df_filter, args.fdr_file,
            args.pval_file, args.seed, mutation_rate_mode, mu_kwargs=mu_kwargs,
            max_rounds=args.conditional_max_rounds,
            criterion=args.conditional_criterion, cutoff=args.conditional_cutoff,
            min_retained=args.conditional_min_retained, ignore_ac=args.ignore_ac,
            alternative=args.alternative, min_alleles=args.min_alleles,
            gene_criterion=args.conditional_gene_criterion,
            gene_cutoff=args.conditional_gene_cutoff, min_denovo=args.min_denovo,
            keep_round_pvals=args.keep_round_pvals,
        )
        did_nothing = False

    elif args.run_3dnt and mutation_rate_mode:
        wrote_results = True
        from mutation_rate_test import mutation_rate_scan_test
        mutation_rate_scan_test(
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
            args.fdr_file,
            args.pval_file,
            args.rate_calibration,
            args.rate_calibration_genes,
            args.min_denovo,
            args.n_trios,
            args.seed,
            args.mu_residue_file,
            args.mu_col is not None,
            args.min_mu_coverage,
            args.max_residues,
            args.simulate_null_from_mu,
            args.test_direction,
            conditioning,
        )
        did_nothing = False

    elif args.run_3dnt and args.test_method == 'cmh':
        wrote_results = True
        logger.info("Starting scan test analysis with CMH")
        scan_test_cmh(
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
            conditioning,
            args.stratum_col,
            args.cmh_screen_threshold,
            args.seed,
            args.alternative,
            args.min_alleles,
        )
        did_nothing = False

    elif args.run_3dnt:
        wrote_results = True
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
            conditioning,
            args.seed,
            args.alternative,
            args.min_alleles,
        )
        did_nothing = False
    
    elif args.annotation_file is not None:
        logger.info('Starting annotation test analysis')
        if args.ignore_ac:
            raise ValueError("--ignore-ac is not supported with --annotation-file")
        
        if args.annotation_id:
            ann_id_columns = tuple(args.annotation_id.split(','))
        else:
            ann_id_columns = ('*',) if args.annotation_format == 'wide' else ('concept_type', 'concept_value')

        expand = args.annotation_expand
        # option to use "radius" column in annotation file:
        ## if radius column exists, otherwise, use neighborhood argument
        normalize_result_fn(args.annotation_result_fn)  # fail early on a bad name
        df_ann = read_table(Path(args.annotation_file))
        if expand is None:
            # No --annotation-expand: never expand, whatever radius is in the file/args.
            if 'radius' in df_ann.columns:
                logger.warning("Ignoring annotation file radius column: no --annotation-expand given")
            df_ann['radius'] = 0.0
            logger.info("No --annotation-expand; annotations are tested without neighborhood expansion")
        elif 'radius' in df_ann.columns:
            if args.annotation_format == 'long':
                # ensure each annotation type only has one radius
                radius_nunique = df_ann.groupby(list(ann_id_columns))['radius'].nunique(dropna=False)
                incohesive = radius_nunique[radius_nunique > 1]
                if len(incohesive) > 0:
                    raise ValueError(
                        "Annotation file has a radius that is not the same for every aa_pos "
                        f"within an annotation (grouped by {ann_id_columns}): "
                        f"{list(incohesive.index)}"
                    )
            # wide format: the converter already splits rows by (value, radius)
            logger.info("Using annotation file radius column as neighborhood radii")
        else:
            if isinstance(args.neighborhood_radius, str):
                raise ValueError('--annotation-expand needs a numeric --neighborhood-radius when the annotation file has no radius column')
            logger.info(
                f"Annotation file has no radius column; using --neighborhood-radius "
                f"({args.neighborhood_radius}) for every annotation"
            )
            df_ann['radius'] = args.neighborhood_radius

        mu_kwargs = {}
        if mutation_rate_mode:
            if args.stratum_col is not None or args.test_method == 'cmh':
                raise ValueError(
                    "--stratum-col/--test-method cmh are not supported with --mu-file/--mu-col in the "
                    "annotation test: all strata share one rate vector, so stratified results equal unstratified."
                )
            if args.alternative == 'two-sided':
                raise ValueError("--alternative two-sided is not supported with a mutation-rate null; use --test-direction")
            from mutation_rate_test import (build_mu_lookup, estimate_lambda, load_mu_per_residue,
                                            simulate_null_input)
            if args.mu_col is not None:
                logger.warning(
                    'Using --mu-col, so mutational opportunity is taken from the input file. Only valid if '
                    'it lists ALL possible missense variants, including those with no observed de novos.'
                )
                df_mu = df_rvas.groupby(['uniprot_id', 'aa_pos'], as_index=False).agg(mu=('mu', 'sum'))
                df_mu['n_with_mu'] = np.nan
                df_mu['n_possible'] = np.nan
            else:
                df_mu = load_mu_per_residue(args.mu_residue_file)
            mu_lookup = build_mu_lookup(df_mu)
            cal_genes = None
            if args.rate_calibration_genes is not None:
                cal_genes = pd.read_csv(args.rate_calibration_genes, sep='\t')['uniprot_id'].unique()
            # lambda_hat BEFORE gene selection (--min-denovo), see estimate_lambda
            lambda_hat = estimate_lambda(df_rvas, df_mu, args.rate_calibration, cal_genes, args.n_trios)
            if args.simulate_null_from_mu:
                df_rvas = simulate_null_input(df_rvas, mu_lookup, lambda_hat, args.seed)
                lambda_hat = estimate_lambda(df_rvas, df_mu, args.rate_calibration, cal_genes, args.n_trios)
                log_counts('Replace counts with a simulated null (--simulate-null-from-mu)',
                           df_rvas, n_denovo=int(df_rvas['ac_case'].sum()))
            logger.info('Annotation test with a mutation-rate null; use a fresh results dir (shards of case/control runs would be reused)')
            mu_kwargs = dict(mu_lookup=mu_lookup, lambda_hat=lambda_hat,
                             min_denovo=args.min_denovo, direction=args.test_direction)

        run_annotation_test(
            df_rvas,
            df_ann,
            ann_format=args.annotation_format,
            ann_id_columns=ann_id_columns,
            expand_include=[] if expand is None else (None if expand == 'all' else expand.split(',')),
            pae_cutoff=args.pae_cutoff,
            reference_dir=args.reference_dir,
            df_include=df_filter,
            n_perm=args.n_sims,
            seed=args.seed,
            results_dir=Path(args.results_dir) / 'annotation_test',
            cleanup_per_protein=True,
            stratum_col=args.stratum_col if annotation_cmh else None,
            alternative=args.alternative,
            result_fn=args.annotation_result_fn,
            min_alleles=args.min_alleles,
            **mu_kwargs,
        )
        did_nothing = False

    elif args.make_movie:
        if not (args.pse and args.results_dir):
            raise ValueError("For making a movie, you must provide --pse and --results_dir")
        from pymol_code import make_movie_from_pse  # pymol is only needed here
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

        def _merge_node(node_in, node_out):
            """
            Copy datasets across, recursing into groups. Mutation-rate results are stored
            per test in the 'poisson' and 'binomial' groups, so those keys collide on
            every file after the first and must be descended into rather than read as
            datasets.
            """
            for key in node_in.keys():
                item = node_in[key]
                if isinstance(item, h5py.Group):
                    _merge_node(item, node_out.require_group(key))
                elif key not in node_out:
                    node_in.copy(key, node_out, name=key)
                else:
                    combined = np.concatenate([node_out[key][:], item[:]], axis=0)
                    del node_out[key]
                    node_out.create_dataset(key, data=combined)

        with h5py.File(os.path.join(args.results_dir, args.pval_file), 'w') as fid_out:
            for file in pval_files_to_combine:
                with h5py.File(file, 'r') as fid_in:
                    _merge_node(fid_in, fid_out)
                    for name, value in fid_in.attrs.items():
                        if name in fid_out.attrs and fid_out.attrs[name] != value:
                            logger.warning(
                                f"{os.path.basename(file)}: attribute '{name}' is {value}, "
                                f"but {fid_out.attrs[name]} was recorded from an earlier "
                                "file. For a mutation-rate run split across jobs, pass an "
                                "exome-wide --rate-calibration fixed:<value> so every chunk "
                                "shares one lambda_hat."
                            )
                        else:
                            fid_out.attrs[name] = value
        did_nothing=False

    # A scan's results table is the input to a component table, so build one by
    # default: the scan centres a neighborhood on every residue, and the components
    # are the units that are actually distinct findings. Guarded, because the scan
    # results are already on disk by this point and a missing structure or an
    # unreadable p-value file must not turn a completed scan into a failed run.
    # Not after a conditional run: conditional_scan has already written
    # <stem>.components.tsv covering every round, and this would rebuild it from
    # round 0 alone -- under --component-gene-cutoff rather than
    # --conditional-gene-cutoff -- and overwrite it at the same path. --no-components
    # is refused alongside --conditional for the same reason, so the two cannot
    # disagree about whether the file should exist.
    if wrote_results and not args.no_components and not args.no_fdr and not args.conditional:
        try:
            build_components_file(args, logger)
        except Exception as e:
            logger.warning(
                f'Scan results were written, but the component table could not be '
                f'built: {e}. Re-run with --components to retry, or --no-components '
                f'to skip it.'
            )

    if did_nothing:
        raise Exception('no analysis specified')
