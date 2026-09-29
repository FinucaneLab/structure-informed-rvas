"""
Conditional 3DNT: repeat the scan with the neighborhoods already called masked out.

The question is whether a gene carries more than one independent 3D signal. A single
strong neighborhood drags down the p-values of every neighborhood that overlaps it, so
the second-best hit in a gene is usually the best hit seen through a different window.
Masking the called neighborhood and re-scanning asks what is left.

Per protein. Each gene conditions on its own hits and is re-scanned on its own; genes
without a hit are never re-scanned and keep their round-0 results. A round therefore
costs one scan per gene still carrying signal, not a full exome scan.

Masking, not deletion. The called neighborhood's residues are zeroed in place -- counts
and, under the mutation-rate null, mutational opportunity together. See the indexing
contract in utils.py for why rows are never dropped, and compute_all_pvals_mu for why m
has to go with the counts.

Multiple testing stays exome-wide. Each round writes a p-value file that is the previous
round's file with the re-scanned genes overwritten, so the empirical null pool still
spans every gene in the analysis -- unmasked nulls for the genes that were not touched,
freshly drawn masked nulls for the genes that were. FDR and FWER in round k are therefore
on the same footing as in round 0, which is what makes the rounds comparable.

What this does NOT correct for is the sequential testing itself: each round is controlled
at the nominal level, and the rounds are not jointly corrected. This matches the usual
convention in conditional association analysis, and means the later rounds should be read
as "there is additional signal here" rather than as independently calibrated discoveries.
"""

import os
import shutil

import numpy as np
import pandas as pd

from logger_config import get_logger
from utils import best_radius_for_hit
from empirical_fdr import compute_fdr
from scan_test import scan_test, DEFAULT_MIN_RETAINED

logger = get_logger(__name__)


def _round_path(filename, round_idx):
    """results.h5 -> results.round2.h5, keeping any multi-part extension intact."""
    stem, ext = os.path.splitext(filename)
    return f'{stem}.round{round_idx}{ext}'


def _criterion_column(criterion, mutation_rate_mode):
    """The column a hit is judged on, in each mode."""
    if mutation_rate_mode:
        # fdr_max / fwer_max: a neighborhood passes only if it passes both tests.
        return 'fdr_max' if criterion == 'fdr' else 'fwer_max'
    return 'fdr' if criterion == 'fdr' else 'fwer'


def _radius_columns(mutation_rate_mode):
    """Where the per-position chosen radius lives, per mode.

    Under the mutation-rate null the two tests pick their best radius independently;
    Test B is the test of interest (see the mutation_rate_test module docstring), so its
    radius defines the neighborhood to condition out.
    """
    return ('radius_binomial', 'radius_poisson') if mutation_rate_mode else ('radius',)


def _top_hits(df_results, active_genes, crit_col, cutoff):
    """The best remaining position in each active gene, if it clears the cutoff."""
    df = df_results[df_results['uniprot_id'].isin(active_genes)]
    df = df[df[crit_col].notna()]
    if len(df) == 0:
        return df.head(0)
    best = df.loc[df.groupby('uniprot_id')[crit_col].idxmin()]
    return best[best[crit_col] < cutoff].sort_values(by=crit_col)


def conditional_scan(
    df_rvas,
    reference_dir,
    radius,
    pae_cutoff,
    results_dir,
    n_sims,
    fdr_cutoff,
    df_filter,
    fdr_file,
    pval_file,
    seed,
    mutation_rate_mode,
    mu_kwargs=None,
    max_rounds=5,
    criterion='fdr',
    cutoff=None,
    min_retained=DEFAULT_MIN_RETAINED,
    conditioning_radius='best',
    ignore_ac=False,
):
    """
    Run the base scan, then repeat it with each gene's called neighborhoods masked out.

    Writes, in results_dir:
      <fdr_file>, <pval_file>              round 0, byte-for-byte what a plain run gives
      <stem>.roundK.tsv / .roundK.h5       the full exome-wide table for each later round
      <stem>.conditional.tsv               one row per independent signal, all rounds

    Returns the conditional summary table.
    """
    from mutation_rate_test import mutation_rate_scan_test, _compute_both_fdrs

    mu_kwargs = dict(mu_kwargs or {})
    cutoff = fdr_cutoff if cutoff is None else cutoff
    crit_col = _criterion_column(criterion, mutation_rate_mode)
    radius_cols = _radius_columns(mutation_rate_mode)

    # ---------------- round 0: the ordinary scan ----------------
    logger.info('Conditional analysis, round 0: unconditioned scan')
    if mutation_rate_mode:
        mutation_rate_scan_test(
            df_rvas, reference_dir, radius, pae_cutoff, results_dir, n_sims,
            False, False, fdr_cutoff, df_filter, fdr_file, pval_file, **mu_kwargs
        )
    else:
        scan_test(
            df_rvas, reference_dir, radius, pae_cutoff, results_dir, n_sims,
            False, False, fdr_cutoff, df_filter, ignore_ac, fdr_file, pval_file,
            conditioning=None, seed=seed, min_retained=min_retained,
        )

    df_results = pd.read_csv(os.path.join(results_dir, fdr_file), sep='\t')
    if crit_col not in df_results.columns:
        raise ValueError(
            f'{fdr_file} has no column {crit_col}; --conditional-criterion '
            f'{criterion} is not available for this test'
        )

    # Every later round is corrected against this same gene set, so that the rounds are
    # comparable and a gene does not enter or leave the multiple-testing burden mid-way.
    analysis_genes = sorted(df_results['uniprot_id'].unique())
    logger.info(f'Round 0 covers {len(analysis_genes)} genes')
    round_filter = pd.DataFrame({'uniprot_id': analysis_genes})

    conditioning = {}           # uniprot_id -> [(aa_pos, radius), ...]
    accepted_rows = []
    current_pval_file = pval_file

    hits = _top_hits(df_results, analysis_genes, crit_col, cutoff)
    logger.info(f'Round 0: {len(hits)} genes with a hit at {crit_col} < {cutoff}')
    for _, row in hits.iterrows():
        rec = row.to_dict()
        rec.update({'round': 0, 'conditioned_on': '', 'n_conditioned_nbhds': 0})
        accepted_rows.append(rec)

    active = list(hits['uniprot_id'])

    # ---------------- rounds 1..max_rounds ----------------
    for round_idx in range(1, max_rounds + 1):
        if not active:
            logger.info(f'No genes left with signal; stopping after round {round_idx - 1}')
            break

        # Extend each active gene's mask with the hit it just produced.
        for _, row in hits.iterrows():
            uid, aa_pos = row['uniprot_id'], int(row['aa_pos'])
            r = (best_radius_for_hit(df_results, uid, aa_pos, radius_cols=radius_cols)
                 if conditioning_radius == 'best' else float(conditioning_radius))
            conditioning.setdefault(uid, []).append((aa_pos, r))

        prev_pval_file = current_pval_file
        current_pval_file = _round_path(pval_file, round_idx)
        round_fdr_file = _round_path(fdr_file, round_idx)

        # Start from the previous round's file so the untouched genes keep their nulls;
        # the re-scanned genes' datasets are then overwritten in place by write_dataset.
        shutil.copyfile(os.path.join(results_dir, prev_pval_file),
                        os.path.join(results_dir, current_pval_file))

        active = sorted(active)   # the per-protein seed derives from list position
        logger.info(f'Conditional analysis, round {round_idx}: re-scanning '
                    f'{len(active)} genes with {sum(len(conditioning[g]) for g in active)} '
                    f'conditioned neighborhoods')

        if mutation_rate_mode:
            mutation_rate_scan_test(
                df_rvas, reference_dir, radius, pae_cutoff, results_dir, n_sims,
                True, False, fdr_cutoff, df_filter, round_fdr_file, current_pval_file,
                conditioning=conditioning, min_retained=min_retained,
                seed_salt=round_idx, gene_list=active, **mu_kwargs
            )
            df_results = _compute_both_fdrs(results_dir, fdr_cutoff, round_filter,
                                            reference_dir, current_pval_file)
        else:
            scan_test(
                df_rvas, reference_dir, radius, pae_cutoff, results_dir, n_sims,
                True, False, fdr_cutoff, df_filter, ignore_ac, round_fdr_file,
                current_pval_file, conditioning=conditioning, seed=seed,
                min_retained=min_retained, seed_salt=round_idx,
                uniprot_id_list=active,
            )
            df_results = compute_fdr(results_dir, fdr_cutoff, round_filter,
                                     reference_dir, current_pval_file)

        df_results.to_csv(os.path.join(results_dir, round_fdr_file), sep='\t', index=False)

        hits = _top_hits(df_results, active, crit_col, cutoff)

        # A gene whose re-scan failed keeps the previous round's datasets in the copied
        # file, which would re-report the hit we just masked. Catch that rather than
        # letting it look like a second independent signal.
        kept = []
        for _, row in hits.iterrows():
            uid, aa_pos = row['uniprot_id'], int(row['aa_pos'])
            already = {p for p, _ in conditioning.get(uid, [])}
            if aa_pos in already:
                logger.error(
                    f'{uid}: round {round_idx} again picked residue {aa_pos}, which is '
                    'already conditioned out. The re-scan did not take effect for this '
                    'gene (check the errors above); dropping it from the analysis.'
                )
                continue
            rec = row.to_dict()
            rec.update({
                'round': round_idx,
                'conditioned_on': ','.join(str(p) for p, _ in conditioning[uid]),
                'n_conditioned_nbhds': len(conditioning[uid]),
            })
            accepted_rows.append(rec)
            kept.append(uid)

        hits = hits[hits['uniprot_id'].isin(kept)]
        logger.info(f'Round {round_idx}: {len(kept)} genes with a further independent '
                    f'hit at {crit_col} < {cutoff}')
        active = kept
    else:
        if active:
            logger.warning(
                f'Stopped at --conditional-max-rounds {max_rounds} with {len(active)} '
                'genes still showing signal; raise the limit to go further.'
            )

    # ---------------- summary ----------------
    df_conditional = pd.DataFrame(accepted_rows)
    stem, _ = os.path.splitext(fdr_file)
    out_file = f'{stem}.conditional.tsv'
    if len(df_conditional):
        front = [c for c in ('round', 'uniprot_id', 'gene_name', 'aa_pos', crit_col)
                 if c in df_conditional.columns]
        rest = [c for c in df_conditional.columns if c not in front]
        df_conditional = df_conditional[front + rest].sort_values(
            by=['uniprot_id', 'round']).reset_index(drop=True)
    df_conditional.to_csv(os.path.join(results_dir, out_file), sep='\t', index=False)

    n_multi = 0
    if len(df_conditional):
        per_gene = df_conditional.groupby('uniprot_id').size()
        n_multi = int((per_gene > 1).sum())
        logger.info('')
        logger.info(f'{len(df_conditional)} independent signals across '
                    f'{df_conditional["uniprot_id"].nunique()} genes; '
                    f'{n_multi} genes carry more than one')
        extra = df_conditional[df_conditional['round'] > 0]
        if len(extra):
            logger.info(f'Signals found only after conditioning:\n{extra.to_string()}')
    else:
        logger.info(f'No genes had a hit at {crit_col} < {cutoff}')
    logger.info(f'Wrote {out_file}')

    return df_conditional
