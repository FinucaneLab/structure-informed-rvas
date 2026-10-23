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
from collections import namedtuple

import pandas as pd

from logger_config import get_logger
from utils import parse_pymol_positions, positions_to_pymol_notation
from components import components_for_genes, top_component
from empirical_fdr import compute_fdr
from scan_test import scan_test, DEFAULT_MIN_RETAINED

logger = get_logger(__name__)


def _round_path(filename, round_idx):
    """results.h5 -> results.round2.h5, keeping any multi-part extension intact."""
    stem, ext = os.path.splitext(filename)
    return f'{stem}.round{round_idx}{ext}'


# Every column name the conditional rounds read, per null. Both modes name the
# corrections `fdr` and `fwer`, so only the p-value and the radius differ -- which is
# why there is one table here instead of a selector function per column.
#
# Under the mutation-rate null the two tests pick their best radius independently, and
# Test B is the test of interest (see the mutation_rate_test module docstring), so its
# radius comes first and defines the neighborhood to condition out.
ResultColumns = namedtuple('ResultColumns', 'p radius')
RESULT_COLUMNS = {
    False: ResultColumns(p='p_value', radius=('radius',)),
    True:  ResultColumns(p='p_binomial', radius=('radius_binomial', 'radius_poisson')),
}
FDR_COLUMN = 'fdr'
FWER_COLUMN = 'fwer'


def _column_for(criterion):
    """The column a criterion names. Used for both the member gate and the gene gate --
    they are the same choice, made twice."""
    return FDR_COLUMN if criterion == 'fdr' else FWER_COLUMN


TOTAL_COLUMNS = ('n_denovo_gene', 'mu_gene', 'mu_coverage')


def _carry_gene_totals(df_results, gene_totals, df_round0):
    """
    Give a later round's table the per-gene columns round 0 has.

    mutation_rate_scan_test merges n_denovo_gene, mu_gene and mu_coverage on after
    computing the corrections; the conditional rounds call _compute_both_fdrs directly
    and so would omit them, leaving the round tables a different shape from round 0 for
    anything reading them side by side.

    Two sources, because neither alone covers the table. `gene_totals` is THIS round's,
    as returned by mutation_rate_scan_test, and is the only correct source for a gene
    that was re-scanned: conditioning masks residues out of original_case and
    original_mu, and Test B's binomial null uses the masked n_g and m_g, so round 0's
    values would print a 40-de-novo total beside p-values computed against the 10 that
    survived. But a round re-scans only `active`, so gene_totals covers just those;
    every other gene in the table was never masked, which makes round 0's values exactly
    right for it. Taking only one source leaves the other half of the table wrong or
    NaN.
    """
    frames = []
    if gene_totals:
        frames.append(pd.DataFrame(
            [(uid, n_g, m_g, cov) for uid, (n_g, m_g, cov) in gene_totals.items()],
            columns=['uniprot_id'] + list(TOTAL_COLUMNS),
        ))
    have = set(frames[0].uniprot_id) if frames else set()
    if df_round0 is not None and all(c in df_round0.columns for c in TOTAL_COLUMNS):
        carried = df_round0[['uniprot_id'] + list(TOTAL_COLUMNS)].drop_duplicates(
            subset='uniprot_id')
        frames.append(carried[~carried.uniprot_id.isin(have)])
    if not frames:
        return df_results
    df_totals = pd.concat(frames, ignore_index=True)
    keep = [c for c in df_totals.columns
            if c == 'uniprot_id' or c not in df_results.columns]
    if keep == ['uniprot_id']:
        return df_results
    return df_results.merge(df_totals[keep], on='uniprot_id', how='left')


def _drop_failed_rescans(active, scanned, round_filter, round_idx):
    """
    Remove genes whose re-scan did not happen from the round's active set and pool.

    A round copies the previous round's p-value file and overwrites only the genes it
    re-scans, so a gene the scan skipped -- too few alleles left after conditioning, a
    missing structure, any per-gene error -- keeps its UNCONDITIONED p-values in that
    file. Those values are small, by construction: they are why the gene was selected
    for conditioning in the first place. Leaving them in means this round's FDR and
    FWER are computed against a null pool containing a signal the analysis believes it
    masked, which deflates FWER and inflates the false-discovery count for every other
    gene in the round.

    The `lead_aa_pos in masked` guard downstream catches the same genes and stops them
    being REPORTED, but it runs after the corrections are computed, so it cannot undo
    their effect on everyone else. Returns (active, round_filter) with them removed.
    """
    failed = [uid for uid in active if uid not in scanned]
    if not failed:
        return active, round_filter
    logger.warning(
        f'Round {round_idx}: {len(failed)} genes were not re-scanned and keep the '
        f'previous round\'s unconditioned p-values; dropping them from this round so '
        f'they do not enter its FDR and FWER pools '
        f'({", ".join(failed[:8])}{", ..." if len(failed) > 8 else ""})'
    )
    kept = [uid for uid in active if uid in scanned]
    if not kept:
        logger.info(f'Round {round_idx}: no gene was successfully re-scanned; stopping')
    return kept, round_filter[~round_filter['uniprot_id'].isin(failed)]


def _genes_passing_gate(df_results, active_genes, gate_col, gate_cutoff):
    """
    Genes with at least one neighborhood clearing the gene gate.

    Direction is not filtered here. The scan is one-sided by default, so a depleted
    neighborhood has p = 1 and cannot clear any gate; and for an older two-sided results
    file, components.enriched_mask drops the depleted neighborhoods, leaving such a gene
    with no component and top_component returning None, so it falls out below anyway.

    Doing it here as well would mean a second definition of "enriched", and the obvious
    column to use -- `ratio` -- is the regularized one, which disagrees with the test
    near OR = 1.
    """
    df = df_results[df_results['uniprot_id'].isin(active_genes)]
    df = df[df[gate_col].notna() & (df[gate_col] < gate_cutoff)]
    return sorted(df['uniprot_id'].unique())


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
    ignore_ac=False,
    include_depletion=False,
    min_alleles=5,
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
    cols = RESULT_COLUMNS[bool(mutation_rate_mode)]
    p_col, radius_cols = cols.p, cols.radius
    # Which correction each gate is judged on. Rows are always written under
    # FDR_COLUMN and FWER_COLUMN, never under these -- with --conditional-criterion
    # fwer the two selectors name the same column, and keying a row on them would
    # write a table with no FDR column at all.
    member_col = _column_for(criterion)
    gate_col = _column_for(gene_criterion)
    gate_cutoff = gene_cutoff

    def read_mut(uniprot_id):
        """Per-residue counts for one gene, from whichever p-value file is current."""
        import h5py
        import hdf5plugin  # noqa: F401
        from utils import resolve_pval_node, read_original_mutation_data
        test = 'binomial' if mutation_rate_mode else None
        with h5py.File(os.path.join(results_dir, current_pval_file), 'r') as fid:
            node, mu_mode = resolve_pval_node(fid, test or 'binomial')
            return read_original_mutation_data(node, uniprot_id, mu_mode=mu_mode)

    def gene_is_exhausted(uniprot_id, masked):
        """
        Has conditioning taken this gene below the bar it had to clear to be analysed?

        The bar differs by null, so the check has to as well. The mutation-rate scan
        admits a gene on --min-denovo de novos; the case/control scan admits it on
        --min-alleles case AND control alleles. Counting only cases in case/control mode
        would let a gene through while its controls had been masked away, leaving the
        Fisher test nothing to compare against.

        Returns (exhausted, description); exhausted is None when the counts cannot be
        read, which is not the same as "has enough left".
        """
        try:
            df_mut = read_mut(uniprot_id)
        except Exception as e:
            logger.warning(f'{uniprot_id}: could not count remaining alleles - {e}')
            return None, None
        left = df_mut.loc[~df_mut.aa_pos.isin(masked)]
        n_case = int(left['ac_case'].sum())
        if mutation_rate_mode:
            return n_case < min_denovo, f'{n_case} de novos'
        n_control = int(left['ac_control'].sum())
        return (n_case < min_alleles or n_control < min_alleles,
                f'{n_case} case / {n_control} control alleles')

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
            include_depletion=include_depletion, min_alleles=min_alleles,
        )

    df_results = pd.read_csv(os.path.join(results_dir, fdr_file), sep='\t')
    round0_results = df_results
    if member_col not in df_results.columns:
        raise ValueError(
            f'{fdr_file} has no column {member_col}; --conditional-criterion '
            f'{criterion} is not available for this test'
        )

    # Every later round is corrected against the same set of tests as round 0, so that
    # the rounds are comparable and no position enters or leaves the multiple-testing
    # burden mid-way -- except for genes that stop being re-scanned, which have to
    # leave it, since a gene not re-scanned carries stale unconditioned p-values in
    # the copied file. Those exits are permanent: every later round copies this one.
    # When --df-filter carries an aa_pos column the filter is position-level, and it
    # has to be carried through verbatim: dropping it would silently correct later
    # rounds over every residue in the protein instead of the filtered subset.
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
            p_col, member_col, radius_cols, cutoff, mutation_rate_mode,
            round_idx=round_idx, default_radius=default_radius,
            gate_col=gate_col, fdr_name=FDR_COLUMN, fwer_name=FWER_COLUMN,
        )

    # The gene gate is FWER by default and the members are the FDR-significant
    # neighborhoods, so a gene qualifies on strong evidence but conditions on everything
    # that evidence is spread across.
    gated = _genes_passing_gate(df_results, analysis_genes, gate_col, gate_cutoff)
    logger.info(f'Round 0: {len(gated)} genes pass the gene gate '
                f'({gate_col} < {gate_cutoff})')
    comps = _components(df_results, gated, 0)

    # radius_cols[0] is the column this mode's results carry the chosen radius in
    # ('radius', or 'radius_binomial' under the mutation-rate null). Recording it under
    # that name is what lets `run_all --fdr-file <stem>.conditional.tsv` rebuild each
    # neighborhood at the radius the scan picked instead of falling back to --radius.
    accepted = []
    for uid in gated:
        comp = top_component(comps, uid)
        if comp is None:
            continue
        # The gene gate admitted this gene on SOME neighborhood, which need not be one
        # of this component's. With --include-depletion, or over an older two-sided
        # results file, the gate can be cleared by a depleted neighborhood that
        # enriched_mask then drops, leaving an enriched component that never cleared it.
        # Reporting that as an accepted signal -- and masking it -- is what this guard
        # stops; _genes_passing_gate deliberately does not filter direction itself.
        if not float(comp.best_gate) < gate_cutoff:
            logger.info(f'{uid}: best component is {gate_col} = {float(comp.best_gate):.3g}, '
                        f'above the gene gate ({gate_cutoff}); not reported')
            continue
        accepted.append(uid)
        rec = {'uniprot_id': uid, 'gene_name': comp.gene_name, 'aa_pos': comp.lead_aa_pos,
               'round': 0, 'conditioned_on': '', 'n_conditioned_residues': 0,
               'component_id': comp.component_id, 'n_centers': comp.n_centers,
               'n_residues': comp.n_residues, 'n_case_alleles': comp.n_case_alleles,
               radius_cols[0]: float(comp.radius),
               # The component's own best values, not the lead's. The component was
               # SELECTED on best_gate, and under the mutation-rate null the gate and
               # criterion columns are maxima over both tests and so are not monotone in
               # the p_binomial that picks the lead -- so a component can clear the gate
               # through one member while its lead sits above the cutoff. Writing the
               # lead's value made `run_all --fdr-file <stem>.conditional.tsv
               # --all-signals` filter out a signal this analysis accepted and masked.
               # p_col already carries the component minimum, since lead_p is the min.
               FDR_COLUMN: float(comp.best_fdr), FWER_COLUMN: float(comp.best_fwer),
               p_col: comp.lead_p}
        accepted_rows.append(rec)

    # `accepted`, not `gated`: a gene whose component the guard above rejected has no
    # round-0 signal, so masking that component and re-scanning it would pay for a round
    # whose hits would be written as "further independent" signals for a gene that never
    # had a first one. Rounds >= 1 have always filtered to `kept` this way; round 0 did
    # not, which left the guard stopping the report but not the masking it names.
    # An empty frame has no uniprot_id column to filter on, so it passes through.
    hits = comps[comps.uniprot_id.isin(accepted)] if len(comps) else comps
    component_tables.append(hits)
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
            spent, detail = gene_is_exhausted(uid, conditioning.get(uid, set()))
            # None means the counts were unreadable, not that the gene is spent.
            # Re-scan it: the warning is already logged, and dropping a gene on a
            # transient read failure would silently shorten the analysis.
            if spent is True:
                exhausted.append((uid, detail))
        if exhausted:
            logger.info(
                f'Round {round_idx}: {len(exhausted)} genes fall below the bar they had '
                f'to clear to enter the analysis, and are not re-scanned '
                f'({", ".join(f"{u} has {n}" for u, n in exhausted[:8])}'
                f'{", ..." if len(exhausted) > 8 else ""})'
            )
            drop = {u for u, _ in exhausted}
            active = [u for u in active if u not in drop]
            # Out of `round_filter` as well as `active`: not being re-scanned, these
            # genes keep the previous round's UNCONDITIONED p-values in the copied
            # file, and leaving them in the pool is the same corruption of this
            # round's FDR and FWER that _drop_failed_rescans exists to prevent.
            round_filter = round_filter[~round_filter['uniprot_id'].isin(drop)]
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
            round_totals = mutation_rate_scan_test(
                df_rvas, reference_dir, radius, pae_cutoff, results_dir, n_sims,
                True, False, fdr_cutoff, df_filter, round_fdr_file, current_pval_file,
                conditioning=conditioning, min_retained=min_retained,
                seed_salt=round_idx, gene_list=active, **mu_kwargs
            )
            # gene_totals is keyed by the genes actually re-scanned.
            active, round_filter = _drop_failed_rescans(
                active, set(round_totals or {}), round_filter, round_idx)
            if not active:
                break
            df_results = _compute_both_fdrs(results_dir, fdr_cutoff, round_filter,
                                            reference_dir, current_pval_file)
            # Round 0 goes through mutation_rate_scan_test, which merges the per-gene
            # totals on afterwards; calling _compute_both_fdrs directly skips that, so
            # the round tables would be ragged against round 0. These are this round's
            # own totals, computed from the conditioned arrays.
            df_results = _carry_gene_totals(df_results, round_totals, round0_results)
        else:
            scanned = scan_test(
                df_rvas, reference_dir, radius, pae_cutoff, results_dir, n_sims,
                True, False, fdr_cutoff, df_filter, ignore_ac, round_fdr_file,
                current_pval_file, conditioning=conditioning, seed=seed,
                min_retained=min_retained, seed_salt=round_idx,
                uniprot_id_list=active, include_depletion=include_depletion,
                min_alleles=min_alleles,
            )
            active, round_filter = _drop_failed_rescans(
                active, scanned, round_filter, round_idx)
            if not active:
                break
            df_results = compute_fdr(results_dir, fdr_cutoff, round_filter,
                                     reference_dir, current_pval_file)

        df_results.to_csv(os.path.join(results_dir, round_fdr_file), sep='\t', index=False)

        # Re-apply the gene gate each round: a gene continues only on fresh
        # FWER-strength evidence, not on an FDR-level remnant of what was masked.
        gated = _genes_passing_gate(df_results, active, gate_col, gate_cutoff)
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
            if not float(comp.best_gate) < gate_cutoff:
                logger.info(f'{uid}: round {round_idx} component is {gate_col} = '
                            f'{float(comp.best_gate):.3g}, above the gene gate '
                            f'({gate_cutoff}); not reported')
                continue
            overlap = len(parse_pymol_positions(comp.residues_pymol) & masked)
            rec = {'uniprot_id': uid, 'gene_name': comp.gene_name,
                   'aa_pos': comp.lead_aa_pos, 'round': round_idx,
                   'conditioned_on': positions_to_pymol_notation(sorted(masked)),
                   'n_conditioned_residues': len(masked),
                   'component_id': comp.component_id, 'n_centers': comp.n_centers,
                   'n_residues': comp.n_residues, 'n_case_alleles': comp.n_case_alleles,
                   'n_residues_overlapping_mask': overlap,
                   radius_cols[0]: float(comp.radius),
                   FDR_COLUMN: float(comp.best_fdr), FWER_COLUMN: float(comp.best_fwer),
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
        front = [c for c in ('round', 'uniprot_id', 'gene_name', 'aa_pos', member_col)
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
