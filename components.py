"""
Group a gene's significant neighborhoods into components.

The scan tests a neighborhood centred on every residue, so one real 3D signal is
reported many times over: in the ASD analysis the 20 genes with an FWER-significant
neighborhood carry 746 FDR-significant neighborhoods between them. Those are not 746
findings. A component is the unit that is.

Two neighborhoods join the same component when they share at least one CASE-MUTATED
residue. The rule is deliberately about the mutations rather than the geometry. Plain
residue overlap connects any two centres within roughly twice the radius whether or not
a single variant sits in the overlap, which chains distinct sites together across a
domain; requiring a shared mutated residue means the two windows are reporting the same
variants, which is what makes them the same finding.

Membership is transitive, as connected components always are: A and C can land in one
component through B without sharing a mutated residue themselves. That is intended -- a
run of overlapping windows walking along one cluster of de novos is one signal -- but it
is also the direction in which this can over-merge, so component_table reports
seq_span_aa and the caller can see how far a component reaches.

Only ENRICHED neighborhoods are clustered. The standard case/control scan is
two-sided, so its significant set mixes neighborhoods carrying more case variants than
expected with ones carrying fewer. A component is a cluster of case variants, so a
depleted neighborhood does not belong in one: on the ClinVar results, clustering
everything produced 1,440 "components" of which 81% were lone neighborhoods holding no
case variant at all, and most of the rest with one or two were depletion calls. They are
counted and reported, not silently dropped -- clustering depletion would need a rule
based on the controls rather than the cases.

Gene gate and member gate are different by design. A gene qualifies on having an
FWER-significant neighborhood; the components are then built from all of its
FDR-significant neighborhoods. FWER significance implies FDR significance at the same
cutoff, so the gate never admits a gene whose qualifying neighborhood is then left out
of the clustering.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from logger_config import get_logger
from utils import (get_adjacency_matrix, positions_to_pymol_notation,
                   parse_pymol_positions)

logger = get_logger(__name__)


def neighborhood_sets(adj_by_radius, centers):
    """
    Residue sets of each centre's neighborhood, as 1-based positions.

    `centers` is a list of (aa_pos, radius) pairs; each neighborhood is built at its own
    radius, which matters under --neighborhood-radius multiple-*, where every position
    picks its own.
    """
    out = []
    for aa_pos, radius in centers:
        adj = adj_by_radius[radius]
        out.append(set(np.flatnonzero(adj[int(aa_pos) - 1]) + 1))
    return out


def build_components(centers, adj_by_radius, case_positions):
    """
    Label each centre with its component index.

    Returns (labels, nbhd_sets, shared_sets) where labels[i] is the component of
    centers[i], nbhd_sets[i] is that centre's residue set, and shared_sets[i] is the
    case-mutated subset that decides the edges.
    """
    nbhd = neighborhood_sets(adj_by_radius, centers)
    case_positions = set(int(p) for p in case_positions)
    shared = [n & case_positions for n in nbhd]

    k = len(centers)
    if k == 0:
        return np.zeros(0, dtype=int), nbhd, shared

    # Centres x mutated-residues incidence, so the edge test is one boolean matmul
    # rather than k^2 set intersections. The mutated residues are few (tens), so this
    # stays small even for CDK13's 87 centres.
    mutated = sorted(set().union(*shared)) if any(shared) else []
    if not mutated:
        # No centre contains a case mutation: nothing can share one, so every centre is
        # its own component. Reachable when a neighborhood is significant through
        # depletion (--test-direction depletion).
        return np.arange(k), nbhd, shared
    index = {p: j for j, p in enumerate(mutated)}
    B = np.zeros((k, len(mutated)), dtype=bool)
    for i, s in enumerate(shared):
        for p in s:
            B[i, index[p]] = True
    edges = (B.astype(np.int32) @ B.astype(np.int32).T) > 0
    np.fill_diagonal(edges, False)
    _, labels = connected_components(csr_matrix(edges), directed=False)
    return labels, nbhd, shared


def _seq_span(centers, labels, comp):
    """
    Sequence separation between the first and last centre of a component.

    This is a span along the sequence, NOT a spatial diameter. A large value is not a
    defect: it means the component is assembled in 3D from parts of the chain that are
    far apart in sequence, which is the case 3DNT exists to find. It is reported so that
    such components are visible, not so they can be filtered out.
    """
    members = [int(centers[i][0]) for i in range(len(centers)) if labels[i] == comp]
    return (max(members) - min(members)) if members else 0


def component_table(
    df_results,
    uniprot_id,
    gene_name,
    centers,
    adj_by_radius,
    df_mut,
    p_col,
    fdr_col,
    fwer_col,
    mu_mode,
    round_idx=0,
):
    """
    One row per component for a single gene.

    `df_mut` is the per-residue mutation frame from read_original_mutation_data: aa_pos
    and ac_case always, plus mu (mutation-rate mode) or ac_control (case/control).
    `centers` is a list of (aa_pos, radius).
    """
    case_pos = set(df_mut.loc[df_mut.ac_case > 0, 'aa_pos'].astype(int))
    labels, nbhd, shared = build_components(centers, adj_by_radius, case_pos)

    by_pos = df_results.set_index('aa_pos')
    rows = []
    n_caseless = 0
    for comp in sorted(set(labels.tolist())):
        idx = [i for i in range(len(centers)) if labels[i] == comp]
        # A neighborhood holding no case variant cannot share one, so it can never join
        # anything and arrives here as a component of one with nothing in it. That is not
        # a finding: under a two-sided scan these are the depletion calls, and on the
        # ClinVar results they were 81% of the raw output. Count them and move on.
        if not any(shared[i] for i in idx):
            n_caseless += len(idx)
            continue
        member_centers = sorted(int(centers[i][0]) for i in idx)
        residues = set().union(*[nbhd[i] for i in idx])
        in_comp = df_mut[df_mut.aa_pos.isin(residues)]
        case_rows = in_comp[in_comp.ac_case > 0]

        # The lead is the component's best-scoring centre, which is what the conditional
        # rounds report and what the component is named by.
        lead = min(member_centers, key=lambda p: by_pos.loc[p, p_col])
        lead_row = by_pos.loc[lead]

        rec = {
            'uniprot_id': uniprot_id,
            'gene_name': gene_name,
            'component_id': int(comp) + 1,
            'round': round_idx,
            'n_centers': len(idx),
            'lead_aa_pos': int(lead),
            'lead_p': float(lead_row[p_col]),
            'lead_fdr': float(lead_row[fdr_col]),
            'lead_fwer': float(lead_row[fwer_col]),
            'radius': float(dict(centers)[lead]),
            'n_residues': len(residues),
            'seq_span_aa': _seq_span(centers, labels, comp),
            'residues_pymol': positions_to_pymol_notation(residues),
            'centers_pymol': positions_to_pymol_notation(member_centers),
            # pos:radius per member, because under --neighborhood-radius multiple-* each
            # centre picks its own and the single `radius` column only records the lead's.
            # This is what lets a viewer redraw the component as its actual union of
            # spheres rather than one radius applied to every centre.
            'centers_detail': ','.join(
                f'{int(centers[i][0])}:{float(centers[i][1]):g}' for i in sorted(
                    idx, key=lambda j: int(centers[j][0]))),
            'n_case_residues': int(len(case_rows)),
            'n_case_alleles': int(case_rows.ac_case.sum()),
            'case_pymol': positions_to_pymol_notation(case_rows.aa_pos.astype(int)),
            'case_detail': ','.join(f'{int(r.aa_pos)}:{int(r.ac_case)}'
                                    for r in case_rows.sort_values('aa_pos').itertuples()),
        }
        if mu_mode:
            rec['mu_component'] = float(in_comp.mu.sum())
        else:
            ctrl = in_comp[in_comp.ac_control > 0]
            rec['n_control_residues'] = int(len(ctrl))
            rec['n_control_alleles'] = int(ctrl.ac_control.sum())
            rec['control_pymol'] = positions_to_pymol_notation(ctrl.aa_pos.astype(int))
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values('lead_p').reset_index(drop=True)
    out.attrs['n_caseless'] = n_caseless
    return out


def components_for_genes(
    df_results,
    genes,
    reference_dir,
    pae_cutoff,
    read_mut,
    p_col,
    fdr_col,
    fwer_col,
    radius_cols,
    member_cutoff,
    mu_mode,
    round_idx=0,
    default_radius=15.0,
    effect_col=None,
):
    """
    Build the component table for every gene in `genes`.

    `genes` is the set that passed the gene gate; membership within each is every
    neighborhood with `fdr_col` below `member_cutoff`. `read_mut(uniprot_id)` returns
    that gene's per-residue mutation frame.

    Genes whose structure cannot be loaded are skipped with a message rather than
    failing the run: a component table is a reporting artefact, and losing one gene's
    row should not take down a scan that already succeeded.
    """
    pdb_guide = f'{reference_dir}/pdb_pae_file_pos_guide.tsv'
    pdb_dir = f'{reference_dir}/pdb_files/'
    pae_dir = f'{reference_dir}/pae_files/'

    tables = []
    n_caseless = 0
    n_depleted = 0
    for uniprot_id in sorted(genes):
        g = df_results[(df_results.uniprot_id == uniprot_id) &
                       (df_results[fdr_col] < member_cutoff)]
        # Enriched only. A two-sided scan calls depleted neighborhoods too, and a
        # cluster of case variants is not what those are.
        if effect_col and effect_col in g.columns:
            n_all = len(g)
            g = g[g[effect_col] > 1]
            n_depleted += n_all - len(g)
        if len(g) == 0:
            continue
        # One row per position. Results tables written before the gene_name merge was
        # collapsed carry a row per paralogous gene name, which would otherwise put the
        # same centre in the component twice and make the lead lookup ambiguous.
        n_before = len(g)
        g = g.drop_duplicates(subset='aa_pos')
        if len(g) < n_before:
            logger.warning(
                f'{uniprot_id}: {n_before - len(g)} duplicate rows per position in the '
                'results table (several gene names share this accession); using one each'
            )
        gene_name = g.gene_name.iloc[0] if 'gene_name' in g.columns else uniprot_id

        centers = []
        for row in g.itertuples():
            radius = default_radius
            for col in radius_cols:
                val = getattr(row, col, None)
                if val is not None and not pd.isna(val):
                    radius = float(val)
                    break
            centers.append((int(row.aa_pos), radius))

        try:
            adj_by_radius = {
                r: np.asarray(get_adjacency_matrix(pdb_guide, pdb_dir, pae_dir,
                                                   uniprot_id, r, pae_cutoff))
                for r in sorted({r for _, r in centers})
            }
            df_mut = read_mut(uniprot_id)
        except Exception as e:
            logger.warning(f'{uniprot_id}: could not build components - {e}')
            continue

        t = component_table(
            g, uniprot_id, gene_name, centers, adj_by_radius, df_mut,
            p_col, fdr_col, fwer_col, mu_mode, round_idx,
        )
        n_caseless += t.attrs.get('n_caseless', 0)
        if len(t):
            tables.append(t)

    if n_depleted:
        logger.info(
            f'Round {round_idx}: {n_depleted} significant neighborhoods are depleted '
            f'({effect_col} <= 1) and were not clustered; components describe enriched '
            'signal only.'
        )
    if n_caseless:
        logger.info(
            f'Round {round_idx}: {n_caseless} significant neighborhoods contain no case '
            'variant and were not clustered. Under a two-sided scan these are the '
            'depletion calls; clustering them would need a rule based on the controls, '
            'not the cases.'
        )
    if not tables:
        return pd.DataFrame()
    out = pd.concat(tables, ignore_index=True)
    logger.info(
        f'Round {round_idx}: {len(out)} components across {out.uniprot_id.nunique()} '
        f'genes from {int(out.n_centers.sum())} significant neighborhoods'
    )
    return out


def parse_centers_detail(row, default_radius=15.0):
    """
    Member centres of a component as [(aa_pos, radius), ...].

    Prefers the centers_detail column, which carries each centre's own radius. Falls
    back to centers_pymol plus the component's single `radius` for files written before
    centers_detail existed -- exact for a single-radius run, approximate under
    multiple-*. Falls back again to the lead centre alone if neither column is present,
    so a hand-written components file with only the required columns still draws
    something rather than raising.
    """
    detail = getattr(row, 'centers_detail', None)
    if isinstance(detail, str) and detail.strip():
        out = []
        for part in detail.split(','):
            pos, _, rad = part.partition(':')
            out.append((int(pos), float(rad) if rad else default_radius))
        return out

    radius = getattr(row, 'radius', None)
    radius = (float(radius) if radius is not None and not pd.isna(radius)
              else default_radius)

    centers_pymol = getattr(row, 'centers_pymol', None)
    if isinstance(centers_pymol, str) and centers_pymol.strip():
        return [(p, radius) for p in sorted(parse_pymol_positions(centers_pymol))]

    lead = getattr(row, 'lead_aa_pos', None)
    return [(int(lead), radius)] if lead is not None and not pd.isna(lead) else []


def top_component(df_components, uniprot_id):
    """The component holding a gene's best-scoring neighborhood, or None."""
    g = df_components[df_components.uniprot_id == uniprot_id]
    if len(g) == 0:
        return None
    return g.loc[g.lead_p.idxmin()]
