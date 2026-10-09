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

The unit conditioned on is a component, not a single neighborhood. See components.py:
one signal is reported by every window that overlaps it, so masking only the best window
leaves the rest of the same signal in place and the next round re-finds it. A gene
qualifies on having an FWER-significant neighborhood and its components are built from
all of its FDR-significant ones; the top component -- the one holding the gene's best
neighborhood -- is masked in full.

Conditioning on a whole component removes much more than one neighborhood did. Measured
on the ASD analysis it takes out a median 13% of a gene's residues but 73% of its de
novos, and for half the genes it leaves fewer de novos than --min-denovo required to
enter the analysis at all. Those genes are dropped before the re-scan rather than
re-scanned to produce a table of p = 1: Test B conditions on the gene's remaining de novo
count, so with one or two left there is no power for any neighborhood and the empty
result is arithmetic, not evidence of absence.
"""

import os
import shutil

import numpy as np
import pandas as pd

from logger_config import get_logger
from utils import (best_radius_for_hit, parse_pymol_positions,
                   positions_to_pymol_notation)
from components import components_for_genes, top_component
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


def _gene_gate_column(criterion, mutation_rate_mode):
    """The column the GENE gate is judged on (default FWER), as opposed to members."""
    if mutation_rate_mode:
        return 'fwer_max' if criterion == 'fwer' else 'fdr_max'
    return 'fwer' if criterion == 'fwer' else 'fdr'


def _effect_column(mutation_rate_mode, test='binomial'):
    """The observed-vs-expected column, used to keep reporting to enriched hits."""
    return f'obs_exp_{test}' if mutation_rate_mode else 'ratio'


def _genes_passing_gate(df_results, active_genes, gate_col, gate_cutoff, effect_col=None):
    """
    Genes with at least one ENRICHED neighborhood clearing the gene gate.

    The standard scan is two-sided, so without the effect filter a gene could qualify on
    a strongly depleted neighborhood and then be conditioned on a cluster of case
    variants it had nothing to do with.
    """
    df = df_results[df_results['uniprot_id'].isin(active_genes)]
    df = df[df[gate_col].notna() & (df[gate_col] < gate_cutoff)]
    if effect_col and effect_col in df.columns:
        df = df[df[effect_col] > 1]
    return sorted(df['uniprot_id'].unique())


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
    gene_criterion='fwer',
    gene_cutoff=0.05,
    min_denovo=5,
    default_radius=15.0,
):
    """
    Run the base scan, then repeat it with each gene's called neighborhoods masked out.

    Writes, in results_dir:
      <fdr_file>, <pval_file>              round 0, byte-for-byte what a plain run gives
      <stem>.roundK.tsv / .roundK.h5       the full exome-wide table for each later round
      <stem>.components.tsv                every component found, all rounds, with its
                                           residues and mutations in PyMOL notation
      <stem>.conditional.tsv               one row per independent signal, all rounds

    `criterion`/`cutoff` select the MEMBER gate (which neighborhoods make up a
    component, FDR by default); `gene_criterion`/`gene_cutoff` select the GENE gate
    (which genes are considered at all, FWER by default). `min_denovo` matches the scan's
    own gene filter and is reused to stop a gene once conditioning has taken its de novos
    below the bar it had to clear to be analysed.

    Returns the conditional summary table.
    """
    from mutation_rate_test import mutation_rate_scan_test, _compute_both_fdrs

    mu_kwargs = dict(mu_kwargs or {})
    cutoff = fdr_cutoff if cutoff is None else cutoff
    crit_col = _criterion_column(criterion, mutation_rate_mode)
    gate_col = _gene_gate_column(gene_criterion, mutation_rate_mode)
    gate_cutoff = gene_cutoff
    effect_col = _effect_column(mutation_rate_mode)
    p_col = 'p_binomial' if mutation_rate_mode else 'p_value'
    radius_cols = _radius_columns(mutation_rate_mode)

    def read_mut(uniprot_id):
        """Per-residue counts for one gene, from whichever p-value file is current."""
        import h5py
        import hdf5plugin  # noqa: F401
        from utils import resolve_pval_node, read_original_mutation_data
        test = 'binomial' if mutation_rate_mode else None
        with h5py.File(os.path.join(results_dir, current_pval_file), 'r') as fid:
            node, mu_mode = resolve_pval_node(fid, test or 'binomial')
            return read_original_mutation_data(node, uniprot_id, mu_mode=mu_mode)

    def remaining_denovo(uniprot_id, masked):
        """De novos left in a gene once `masked` residues are conditioned out."""
        try:
            df_mut = read_mut(uniprot_id)
        except Exception as e:
            logger.warning(f'{uniprot_id}: could not count remaining de novos - {e}')
            return None
        return int(df_mut.loc[~df_mut.aa_pos.isin(masked), 'ac_case'].sum())

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

    # Every later round is corrected against the same set of tests as round 0, so that
    # the rounds are comparable and no position enters or leaves the multiple-testing
    # burden mid-way. When --df-filter carries an aa_pos column the filter is
    # position-level, and it has to be carried through verbatim: dropping it would
    # silently correct later rounds over every residue in the protein instead of the
    # filtered subset.
    analysis_genes = sorted(df_results['uniprot_id'].unique())
    if df_filter is not None:
        round_filter = df_filter[df_filter['uniprot_id'].isin(analysis_genes)]
        if 'aa_pos' in df_filter.columns:
            logger.info(f'Round 0 covers {len(analysis_genes)} genes and '
                        f'{len(round_filter)} filtered positions; later rounds are '
                        'corrected over the same positions')
        else:
            logger.info(f'Round 0 covers {len(analysis_genes)} genes')
    else:
        round_filter = pd.DataFrame({'uniprot_id': analysis_genes})
        logger.info(f'Round 0 covers {len(analysis_genes)} genes')

    conditioning = {}           # uniprot_id -> set of 1-based residues already masked
    accepted_rows = []
    component_tables = []
    current_pval_file = pval_file

    def _components(results, genes, round_idx):
        return components_for_genes(
            results, genes, reference_dir, pae_cutoff, read_mut,
            p_col, crit_col, gate_col, radius_cols, cutoff, mutation_rate_mode,
            round_idx=round_idx, default_radius=default_radius,
            effect_col=effect_col,
        )

    # The gene gate is FWER by default and the members are the FDR-significant
    # neighborhoods, so a gene qualifies on strong evidence but conditions on everything
    # that evidence is spread across.
    gated = _genes_passing_gate(df_results, analysis_genes, gate_col, gate_cutoff,
                                effect_col)
    logger.info(f'Round 0: {len(gated)} genes pass the gene gate '
                f'({gate_col} < {gate_cutoff})')
    comps = _components(df_results, gated, 0)
    component_tables.append(comps)

    for uid in gated:
        comp = top_component(comps, uid)
        if comp is None:
            continue
        rec = {'uniprot_id': uid, 'gene_name': comp.gene_name, 'aa_pos': comp.lead_aa_pos,
               'round': 0, 'conditioned_on': '', 'n_conditioned_residues': 0,
               'component_id': comp.component_id, 'n_centers': comp.n_centers,
               'n_residues': comp.n_residues, 'n_case_alleles': comp.n_case_alleles,
               crit_col: comp.lead_fdr, gate_col: comp.lead_fwer, p_col: comp.lead_p}
        accepted_rows.append(rec)

    hits = comps[comps.uniprot_id.isin(gated)] if len(comps) else comps
    active = sorted(set(hits['uniprot_id'])) if len(hits) else []

    # ---------------- rounds 1..max_rounds ----------------
    for round_idx in range(1, max_rounds + 1):
        if not active:
            logger.info(f'No genes left with signal; stopping after round {round_idx - 1}')
            break

        # Extend each active gene's mask with the WHOLE top component it just produced,
        # not just that component's best neighborhood: masking one window of a signal
        # leaves the rest of the same signal in place for the next round to re-find.
        for uid in list(active):
            comp = top_component(hits, uid)
            if comp is None:
                continue
            conditioning.setdefault(uid, set()).update(
                parse_pymol_positions(comp.residues_pymol))

        # A gene with too few de novos left cannot produce a hit -- Test B conditions on
        # the gene total, so with one or two remaining no neighborhood can reach
        # significance. Drop those before paying for a re-scan.
        exhausted = []
        for uid in list(active):
            left = remaining_denovo(uid, conditioning.get(uid, set()))
            if left is not None and left <= min_denovo:
                exhausted.append((uid, left))
        if exhausted:
            logger.info(
                f'Round {round_idx}: {len(exhausted)} genes have {min_denovo} or fewer '
                f'de novos left after conditioning and are not re-scanned '
                f'({", ".join(f"{u}:{n}" for u, n in exhausted[:8])}'
                f'{", ..." if len(exhausted) > 8 else ""})'
            )
            drop = {u for u, _ in exhausted}
            active = [u for u in active if u not in drop]
        if not active:
            logger.info(f'No genes left with testable signal; stopping after round '
                        f'{round_idx - 1}')
            break

        prev_pval_file = current_pval_file
        current_pval_file = _round_path(pval_file, round_idx)
        round_fdr_file = _round_path(fdr_file, round_idx)

        # Start from the previous round's file so the untouched genes keep their nulls;
        # the re-scanned genes' datasets are then overwritten in place by write_dataset.
        shutil.copyfile(os.path.join(results_dir, prev_pval_file),
                        os.path.join(results_dir, current_pval_file))

        active = sorted(active)   # the per-protein seed derives from list position
        logger.info(f'Conditional analysis, round {round_idx}: re-scanning '
                    f'{len(active)} genes, masking '
                    f'{sum(len(conditioning[g]) for g in active)} residues in total')

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

        # Re-apply the gene gate each round: a gene continues only on fresh
        # FWER-strength evidence, not on an FDR-level remnant of what was masked.
        gated = _genes_passing_gate(df_results, active, gate_col, gate_cutoff,
                                    effect_col)
        comps = _components(df_results, gated, round_idx)

        # A gene whose re-scan failed keeps the previous round's datasets in the copied
        # file, which would re-report the component we just masked. Catch that rather
        # than letting it look like a second independent signal.
        kept = []
        rows_this_round = []
        for uid in gated:
            comp = top_component(comps, uid)
            if comp is None:
                continue
            masked = conditioning.get(uid, set())
            if int(comp.lead_aa_pos) in masked:
                logger.error(
                    f'{uid}: round {round_idx} again picked residue {comp.lead_aa_pos}, '
                    'which is already conditioned out. The re-scan did not take effect '
                    'for this gene (check the errors above); dropping it from the '
                    'analysis.'
                )
                continue
            overlap = len(parse_pymol_positions(comp.residues_pymol) & masked)
            rec = {'uniprot_id': uid, 'gene_name': comp.gene_name,
                   'aa_pos': comp.lead_aa_pos, 'round': round_idx,
                   'conditioned_on': positions_to_pymol_notation(sorted(masked)),
                   'n_conditioned_residues': len(masked),
                   'component_id': comp.component_id, 'n_centers': comp.n_centers,
                   'n_residues': comp.n_residues, 'n_case_alleles': comp.n_case_alleles,
                   'n_residues_overlapping_mask': overlap,
                   crit_col: comp.lead_fdr, gate_col: comp.lead_fwer,
                   p_col: comp.lead_p}
            accepted_rows.append(rec)
            rows_this_round.append(rec)
            kept.append(uid)

        if len(comps):
            component_tables.append(comps[comps.uniprot_id.isin(kept)])
        hits = comps[comps.uniprot_id.isin(kept)] if len(comps) else comps
        logger.info(f'Round {round_idx}: {len(kept)} genes with a further independent '
                    f'component at {gate_col} < {gate_cutoff}')
        active = kept
    else:
        if active:
            logger.warning(
                f'Stopped at --conditional-max-rounds {max_rounds} with {len(active)} '
                'genes still showing signal; raise the limit to go further.'
            )

    # ---------------- summary ----------------
    stem, _ = os.path.splitext(fdr_file)

    df_components = pd.concat([t for t in component_tables if len(t)], ignore_index=True) \
        if any(len(t) for t in component_tables) else pd.DataFrame()
    comp_file = f'{stem}.components.tsv'
    df_components.to_csv(os.path.join(results_dir, comp_file), sep='\t', index=False)
    if len(df_components):
        logger.info(f'Wrote {comp_file}: {len(df_components)} components across '
                    f'{df_components.uniprot_id.nunique()} genes '
                    f'(from {int(df_components.n_centers.sum())} significant neighborhoods)')
    else:
        logger.info(f'Wrote {comp_file} (empty: no gene passed the gene gate)')

    df_conditional = pd.DataFrame(accepted_rows)
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
        logger.info(f'No genes passed the gene gate at {gate_col} < {gate_cutoff}')
    logger.info(f'Wrote {out_file}')

    return df_conditional
