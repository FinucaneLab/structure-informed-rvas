#!/usr/bin/env python3
"""
visualize_and_interpret.py - Protein structure visualization and interpretation tools

Subcommands:
  run_all    Full pipeline (visualize, annotate, neighborhoods) for all significant
             proteins, or for every component with --components
  visualize  Generate PyMOL visualizations for a single protein
  annotate   Transfer functional site annotations from protein family members
  nbhd-features  Cross-reference a neighborhood with functional features
  color      Color a chain in a PyMOL session by per-residue ratio values

Usage:
  python visualize_and_interpret.py run_all --fdr-file <tsv> --results-dir <dir> --reference-dir <dir>
  python visualize_and_interpret.py annotate <uniprot_id> <output_tsv> [options]
  python visualize_and_interpret.py color --tsv <tsv> --input-cif <cif> --output-pse <pse> --chain <chain> --uniprot <id>
"""

import argparse
import csv
import os
import re
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

from uniprot_annotation import (aggregate_annotations_across_family,
                                generate_simple_feature_table, resolve_family_members)
# Re-exported rather than redefined: this module used to carry its own copies, and the
# positions_to_pymol_notation one lacked the None guard and the int() coercion, so it
# raised on a Series or ndarray and emitted "1.0-2.0" for float input. Tests import
# both names from here, which is why they stay in this namespace.
from utils import parse_pymol_positions, positions_to_pymol_notation


# ---------------------------------------------------------------------------
# Alignment utilities
# ---------------------------------------------------------------------------



















# ---------------------------------------------------------------------------
# UniProt API utilities
# ---------------------------------------------------------------------------

















# ---------------------------------------------------------------------------
# Annotation transfer logic
# ---------------------------------------------------------------------------



def safe_name(name):
    """
    Make a gene name safe to put in a filename.

    compute_fdr joins the gene names of a shared accession with '/' -- HBA1/HBA2, and
    the histone clusters -- so the name is no longer a bare identifier. Interpolated
    into a path it reads as a directory separator, and the write fails on a directory
    that was never created. Collapse anything that is not filename-safe.
    """
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('_')
    return cleaned or 'unnamed'


def _load_gene_map(reference_dir):
    """Return {uniprot_id: gene_name} from protein_sequence_guide.tsv."""
    path = os.path.join(reference_dir, 'protein_sequence_guide.tsv')
    try:
        df = pd.read_csv(path, sep='\t')
        # Every name, slash-joined, exactly as empirical_fdr.compute_fdr builds the
        # gene_name column of the results table. 67 accessions carry several (HBA1/HBA2,
        # the histone clusters); taking just one here disagreed with the components file,
        # which carries the table's name, so --components and --fdr-file wrote one
        # protein's PSEs and TSVs under two different stems.
        return (df[['uniprot_id', 'gene_name']].dropna().drop_duplicates()
                  .groupby('uniprot_id')['gene_name']
                  .apply(lambda names: '/'.join(sorted(names)))
                  .to_dict())
    except Exception:
        return {}


def _load_mut_data(pval_path, uniprot_id, test='binomial'):
    """
    Read one protein's per-residue mutation data, from either flavour of p-value file.
    Returns (df_mut, mu_mode); in mutation-rate mode df_mut carries `mu` rather than
    `ac_control`.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import read_original_mutation_data, resolve_pval_node

    with h5py.File(pval_path, 'r') as fid:
        node, mu_mode = resolve_pval_node(fid, test)
        return read_original_mutation_data(node, uniprot_id, mu_mode=mu_mode), mu_mode


# Column layouts of the two kinds of FDR table. The mutation-rate run corrects each test
# separately and reports the max, so it has no plain `fdr` / `p_value` column at all.
MU_FDR_SCHEMA = {
    'significance_column': 'fdr_max',
    'p_column': {'binomial': 'p_binomial', 'poisson': 'p_poisson'},
    'radius_columns': {'binomial': ('radius_binomial', 'radius_poisson'),
                       'poisson': ('radius_poisson', 'radius_binomial')},
}
STD_FDR_SCHEMA = {
    'significance_column': 'fdr',
    'p_column': {'binomial': 'p_value', 'poisson': 'p_value'},
    'radius_columns': {'binomial': ('radius',), 'poisson': ('radius',)},
}


def _fdr_schema(df_fdr):
    """Pick the column layout for an FDR table by looking at what it actually contains."""
    return MU_FDR_SCHEMA if 'fdr_max' in df_fdr.columns else STD_FDR_SCHEMA


def _compute_nbhd(uniprot_id, center, reference_dir, radius, pae_cutoff):
    """Return set of residue positions in the neighborhood centered at `center`."""
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import get_adjacency_matrix
    pdb_pae_guide = os.path.join(reference_dir, 'pdb_pae_file_pos_guide.tsv')
    pdb_dir = os.path.join(reference_dir, 'pdb_files')
    pae_dir = os.path.join(reference_dir, 'pae_files')
    adj_mat = get_adjacency_matrix(pdb_pae_guide, pdb_dir, pae_dir, uniprot_id, radius, pae_cutoff)
    if adj_mat is None:
        raise RuntimeError(f"No structure found for {uniprot_id}")
    return set(int(x) for x in np.where(adj_mat[int(center) - 1, :] == 1)[0] + 1)


def _nbhd_features_table(nbhd, df_mut, features_df, mu_mode=False):
    """
    Cross-reference `nbhd` (set of residue positions) with `features_df`
    (output of generate_simple_feature_table).

    `df_mut` must have columns aa_pos, ac_case, and either ac_control (standard 3DNT)
    or mu (mutation-rate mode).

    For each feature that overlaps the neighborhood, computes:
      - prop_nbhd_in_feature: fraction of neighborhood residues in the feature
      - prop_feature_in_nbhd: fraction of feature residues in the neighborhood

    and then, in standard mode:
      - prop_case_only_in_feature: fraction of case-only alleles (in the nbhd)
        that fall inside the feature
      - case_only_positions: comma-separated list of case-only-mutated positions

    or, in mutation-rate mode, where there is no control cohort to define "case-only":
      - prop_denovo_in_feature: fraction of the neighborhood's de novos inside the feature
      - prop_mu_in_feature: fraction of the neighborhood's mutational opportunity inside
        the feature -- the null for the line above. A feature that simply covers more of
        the neighborhood, or covers a mutational hotspot, carries a high
        prop_denovo_in_feature on its own; the comparison between the two columns is what
        says whether the de novos are concentrated there beyond the rate.
      - denovo_positions: comma-separated list of de novo positions inside the feature

    Returns a DataFrame sorted descending by the relevant proportion column.
    """
    df_nbhd = df_mut[df_mut['aa_pos'].isin(nbhd)]
    if mu_mode:
        df_hit = df_nbhd[df_nbhd['ac_case'] > 0]
        mu_nbhd = float(df_nbhd['mu'].sum())
    else:
        df_hit = df_nbhd[(df_nbhd['ac_case'] > 0) & (df_nbhd['ac_control'] == 0)]
    n_hit_nbhd = int(df_hit['ac_case'].sum())

    rows = []
    for _, row in features_df.iterrows():
        feature_pos = parse_pymol_positions(row['residues_pymol'])
        overlap = nbhd & feature_pos
        if not overlap:
            continue
        df_hit_in_feature = df_hit[df_hit['aa_pos'].isin(overlap)]
        n_hit_in_feature = int(df_hit_in_feature['ac_case'].sum())
        hit_positions = sorted(df_hit_in_feature['aa_pos'].tolist())
        prop_hit = round(n_hit_in_feature / n_hit_nbhd, 3) if n_hit_nbhd > 0 else None
        out = {
            'feature_name': row['feature_name'],
            'functional_site_category': row['functional_site_category'],
            'prop_nbhd_in_feature': round(len(overlap) / len(nbhd), 3),
            'prop_feature_in_nbhd': round(len(overlap) / row['num_positions'], 3),
        }
        if mu_mode:
            mu_in_feature = float(df_nbhd[df_nbhd['aa_pos'].isin(overlap)]['mu'].sum())
            out['prop_denovo_in_feature'] = prop_hit
            out['prop_mu_in_feature'] = round(mu_in_feature / mu_nbhd, 3) if mu_nbhd > 0 else None
            out['denovo_positions'] = ','.join(str(p) for p in hit_positions) if hit_positions else ''
        else:
            out['prop_case_only_in_feature'] = prop_hit
            out['case_only_positions'] = ','.join(str(p) for p in hit_positions) if hit_positions else ''
        rows.append(out)

    result_df = pd.DataFrame(rows)
    if len(result_df) > 0:
        sort_col = 'prop_denovo_in_feature' if mu_mode else 'prop_case_only_in_feature'
        result_df = result_df.sort_values(sort_col, ascending=False)
    return result_df


def _annotate_and_save(uniprot_id, features_path, cache_dir, confidence_threshold=0.8):
    """
    Functional-site features for one protein, cached on disk.

    If `features_path` already exists it is loaded and returned, so re-running over a
    list of proteins costs nothing for the ones already done. Otherwise the full
    pipeline runs: family lookup, annotation transfer, feature table.
    """
    if os.path.exists(features_path):
        print(f"  Loading existing features from {features_path}")
        return pd.read_csv(features_path, sep='\t')

    os.makedirs(os.path.dirname(os.path.abspath(features_path)), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    family_members = resolve_family_members(uniprot_id, cache_dir)
    annotations_df = aggregate_annotations_across_family(
        uniprot_id, family_members, cache_dir, confidence_threshold=0.7
    )
    features_df = generate_simple_feature_table(annotations_df, confidence_threshold)
    features_df.to_csv(features_path, sep='\t', index=False)
    print(f"  Saved features to {features_path}")
    return features_df


# ---------------------------------------------------------------------------
# Subcommand: annotate
# ---------------------------------------------------------------------------

def cmd_annotate(args):
    """
    CLI handler for `annotate`. Transfers functional site annotations from the query's
    family members onto the query via pairwise alignment, and writes a features TSV.
    Falls back to the protein's own annotations when it has no family record. Results
    are cached, so re-runs across a family are fast.
    """
    print(f"Simple functional site annotation for {args.uniprot_id}")
    print(f"Output will be saved to: {args.output_file}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")

    os.makedirs(args.cache_dir, exist_ok=True)

    print("\n=== Fetching Family Members ===")
    family_members = resolve_family_members(args.uniprot_id, args.cache_dir,
                                            max_members=args.max_family_members)

    annotations_df = aggregate_annotations_across_family(
        args.uniprot_id, family_members, args.cache_dir, confidence_threshold=0.7
    )
    features_df = generate_simple_feature_table(annotations_df, args.confidence_threshold)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    features_df.to_csv(args.output_file, sep='\t', index=False)

    print(f"\n{'='*80}")
    print(f"Output saved to: {args.output_file}")
    print(f"{'='*80}")

    if len(features_df):
        print("\nFeature summary by category:")
        counts = features_df.groupby('functional_site_category').size().sort_values(ascending=False)
        for feat_type, count in counts.items():
            print(f"  {feat_type}: {count} unique features")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: nbhd-features
# ---------------------------------------------------------------------------

def cmd_nbhd_features(args):
    """
    CLI handler for `nbhd-features`. Loads a features TSV (from `annotate`),
    computes the structural neighborhood centered at --aa-pos, then cross-references
    the two. Writes two files to {results_dir}/neighborhoods/:
      - {gene}_{uniprot}_{center}_nbhd.tsv: per-residue mutation counts in the neighborhood
      - {gene}_{uniprot}_{center}_nbhd_features.tsv: overlap of each feature with the
        neighborhood, sorted by fraction of case-only mutations in feature
    """
    features_df = pd.read_csv(args.features, sep='\t')

    gene_name = _load_gene_map(args.reference_dir).get(args.uniprot, args.uniprot)
    center = int(args.aa_pos)

    nbhd = _compute_nbhd(args.uniprot, center, args.reference_dir, args.radius, args.pae_cutoff)
    print(f"Neighborhood centered at {gene_name} ({args.uniprot}) aa {center}: {len(nbhd)} residues")

    df_mut, mu_mode = _load_mut_data(args.pval_file, args.uniprot, args.test)
    if mu_mode:
        print(f"Mutation-rate results; using the {args.test} null.")

    nbhd_dir = os.path.join(args.results_dir, 'neighborhoods')
    os.makedirs(nbhd_dir, exist_ok=True)

    # Save neighborhood residue file
    second_col = 'mu' if mu_mode else 'ac_control'
    nbhd_path = os.path.join(nbhd_dir, f'{safe_name(gene_name)}_{args.uniprot}_{center}_nbhd.tsv')
    df_mut[df_mut['aa_pos'].isin(nbhd)][['aa_pos', 'ac_case', second_col]].to_csv(
        nbhd_path, sep='\t', index=False
    )
    print(f"Saved neighborhood residues to {nbhd_path}")

    result_df = _nbhd_features_table(nbhd, df_mut, features_df, mu_mode=mu_mode)

    if len(result_df) == 0:
        print("No features overlap with this neighborhood.")

    out_path = args.output or os.path.join(
        nbhd_dir, f'{safe_name(gene_name)}_{args.uniprot}_{center}_nbhd_features.tsv'
    )
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    result_df.to_csv(out_path, sep='\t', index=False)
    print(f"Saved {len(result_df)} overlapping features to {out_path}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: color
# ---------------------------------------------------------------------------

def _make_mut_pse(pymol_cmd, input_cif, chain, uniprot_id, pval_file, output_pse,
                  test='binomial'):
    """
    Create a _mut-style PSE: grey cartoon with mutation spheres on one chain.

    Standard 3DNT shows the case/control classes; a mutation-rate run has no controls and
    shows de novo positions instead, darkened where a residue carries more than one.
    """
    df_mut, mu_mode = _load_mut_data(pval_file, uniprot_id, test)

    pymol_cmd.reinitialize()
    pymol_cmd.load(input_cif, "complex")
    pymol_cmd.hide("everything", "all")
    pymol_cmd.show("cartoon", "all")
    pymol_cmd.show("spheres", "hetatm and not resn HOH")
    pymol_cmd.color("grey70", "all")
    pymol_cmd.bg_color("white")

    # Extract target chain so it can have independent transparency
    pymol_cmd.extract("target_chain", f"chain {chain}")
    pymol_cmd.set("cartoon_transparency", 0.6, "complex")
    pymol_cmd.set("cartoon_transparency", 0.0, "target_chain")

    pymol_cmd.set("cartoon_loop_radius", 0.4)
    pymol_cmd.set("specular", 0.0)
    pymol_cmd.set("ambient", 0.4)
    pymol_cmd.set("ray_trace_mode", 1)
    pymol_cmd.set("ray_trace_gain", 0.15)

    from pymol_code import show_mutation_spheres
    # The same named selections as the whole-chain sessions, scoped to the extracted
    # chain so the rest of the complex is left alone.
    show_mutation_spheres(df_mut, mu_mode, scope='target_chain and')

    pymol_cmd.save(output_pse)
    print(f"[OK] Saved mutations session to {output_pse}")


def cmd_color(args):
    """
    CLI handler for `color`. Loads a CIF structure, greys out all chains, then
    colors the specified chain by per-residue case/control ratio using a
    blue (low) to red (high) spectrum stored as B-factors. Ratios are read
    from p_values.h5 (preferred — smooth gradient over all residues) or a TSV.
    Saves a PyMOL session file (.pse).
    """
    from pymol import cmd as pymol_cmd

    if not args.tsv and not args.pval_file:
        raise ValueError("Either --tsv or --pval-file must be provided")

    pymol_cmd.reinitialize()
    pymol_cmd.load(args.input_cif, "complex")
    pymol_cmd.hide("everything", "all")
    pymol_cmd.show("cartoon", "all")
    pymol_cmd.show("spheres", "hetatm and not resn HOH")
    pymol_cmd.color("grey70", "all")
    pymol_cmd.bg_color("white")

    # Extract the target chain into its own object so it can have independent transparency
    pymol_cmd.extract("target_chain", f"chain {args.chain}")
    pymol_cmd.set("cartoon_transparency", 0.6, "complex")
    pymol_cmd.set("cartoon_transparency", 0.0, "target_chain")

    residue_ratios = {}
    mu_mode = False

    if args.pval_file:
        import h5py
        import hdf5plugin  # noqa: F401
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils import read_p_values, resolve_pval_node
        pval_file = args.pval_file
        with h5py.File(pval_file, 'r') as fid:
            node, mu_mode = resolve_pval_node(fid, args.test)
            df_eff = read_p_values(node, args.uniprot, mu_mode=mu_mode)
        # obs/exp against the chosen null in mutation-rate mode; the regularized
        # case/control ratio otherwise.
        effect = df_eff['obs_exp'] if mu_mode else df_eff['ratio']
        residue_ratios = {int(pos): float(v)
                          for pos, v in zip(df_eff['aa_pos'], effect)}
    else:
        with open(args.tsv, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["uniprot_id"] == args.uniprot:
                    try:
                        residue_ratios[int(row["aa_pos"])] = float(row["ratio"])
                    except ValueError:
                        continue

    if not residue_ratios:
        raise RuntimeError(f"No ratio data found for UniProt ID {args.uniprot}")

    from pymol_code import apply_effect_coloring
    apply_effect_coloring(list(residue_ratios.keys()), list(residue_ratios.values()),
                          mu_mode, "target_chain")
    pymol_cmd.set("specular", 0.0)
    pymol_cmd.set("ambient", 0.4)
    pymol_cmd.set("ray_trace_mode", 1)
    pymol_cmd.set("ray_trace_gain", 0.15)
    pymol_cmd.set("cartoon_loop_radius", 0.4)
    pymol_cmd.save(args.output_pse)
    print(f"[OK] Saved colored session to {args.output_pse}")

    if args.pval_file:
        stem = args.output_pse[:-len("_ratio.pse")] if args.output_pse.endswith("_ratio.pse") else args.output_pse[:-4]
        mut_pse = stem + "_mut.pse"
        _make_mut_pse(pymol_cmd, args.input_cif, args.chain, args.uniprot, pval_file, mut_pse,
                      test=args.test)

    return 0


# ---------------------------------------------------------------------------
# Subcommand: visualize
# ---------------------------------------------------------------------------

def cmd_visualize(args):
    """CLI handler for `visualize`. Delegates to pymol_code.run_all for one protein."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pymol_code import run_all
    run_all(args.uniprot, args.results_dir, args.reference_dir, pval_file=args.pval_file,
            test=args.test)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------

def _units_from_components(components_file, gene_map):
    """
    One unit per component, read straight from <stem>.components.tsv.

    The residue set is taken from the file rather than recomputed: a component is the
    union of many overlapping neighborhoods, the conditional analysis already masked
    exactly that set, and rebuilding it here from centres and radii would risk
    annotating a different region from the one that was conditioned on. It also means
    this path needs no structures at all.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from components import parse_centers_detail

    try:
        df = pd.read_csv(components_file, sep='\t')
    except pd.errors.EmptyDataError:
        raise ValueError(
            f'{components_file} is empty: the conditional run found no gene passing '
            'the gene gate, so there are no components to annotate.'
        ) from None
    required = {'uniprot_id', 'component_id', 'residues_pymol', 'lead_aa_pos'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f'{components_file} is missing {sorted(missing)}; expected the '
            'components.tsv written by a --conditional run.'
        )

    units = []
    # Sort by whichever of these the file actually has. `round` is written by
    # conditional_scan but a hand-made components file need not carry it, and both
    # branches of the old expression used it unconditionally.
    sort_cols = [c for c in ('round', 'lead_p') if c in df.columns]
    ordered = df.sort_values(sort_cols) if sort_cols else df
    for row in ordered.itertuples():
        uniprot_id = row.uniprot_id
        # `or` is not enough: a missing gene_name arrives as NaN, which is truthy, so
        # it would flow straight into the output filenames.
        gene_name = getattr(row, 'gene_name', None)
        if gene_name is None or pd.isna(gene_name) or str(gene_name).strip() == '':
            gene_name = gene_map.get(uniprot_id, uniprot_id)
        residues = parse_pymol_positions(row.residues_pymol)
        lead = int(row.lead_aa_pos)
        rnd = int(getattr(row, 'round', 0))
        info = (f'component {row.component_id} (round {rnd}), lead {lead}, '
                f'{len(residues)} residues, {getattr(row, "n_centers", "?")} neighborhoods, '
                f'{getattr(row, "n_case_alleles", "?")} case alleles')
        units.append({
            'uniprot_id': uniprot_id,
            'gene_name': gene_name,
            'center': lead,
            'residues': residues,
            # member centres with their own radii, so PyMOL can redraw the component as
            # the union of spheres it actually is
            'centers': parse_centers_detail(row),
            'label': f'c{int(row.component_id)}_{lead}',
            'kind': 'component',
            'info': info,
        })
    return units


def _units_from_fdr(args, df_fdr, sig_col, p_col, radius_cols, gene_map):
    """One unit per called neighborhood (or per protein, without --all-signals)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import best_radius_for_hit

    df_sig = df_fdr[df_fdr[sig_col] < args.significance_cutoff].copy()
    if len(df_sig) == 0:
        return []

    if args.all_signals:
        # One row per independent signal, as written by --conditional. Collapsing to the
        # best per protein would throw away exactly what that analysis found.
        hits = df_sig.sort_values(p_col).reset_index(drop=True)
    else:
        hits = (df_sig.sort_values(p_col)
                      .groupby('uniprot_id', as_index=False)
                      .first())

    units = []
    for row in hits.itertuples():
        uniprot_id = row.uniprot_id
        center = int(row.aa_pos)
        # The scan records which radius it chose for this position under
        # --neighborhood-radius multiple-*; re-deriving the neighborhood at a fixed 15 A
        # would annotate a different neighborhood from the one that was tested.
        radius = best_radius_for_hit(df_fdr, uniprot_id, center,
                                     default=args.radius, radius_cols=radius_cols)
        units.append({
            'uniprot_id': uniprot_id,
            'gene_name': gene_map.get(uniprot_id, uniprot_id),
            'center': center,
            'residues': None,            # computed from the structure in the loop
            'radius': radius,
            'label': str(center),
            'kind': 'neighborhood',
            'info': (f'neighborhood centre {center}, radius {radius}, '
                     f'{sig_col}={getattr(row, sig_col):.4g} {p_col}={getattr(row, p_col):.4g}'),
        })
    return units


def cmd_analyze(args):
    """
    Full pipeline for every called region:
      1. Visualization (PyMOL session, once per protein)
      2. Annotation (functional sites, once per protein)
      3. Per-residue table for the region
      4. Region x features table

    The region is a component when --components is given, and a single neighborhood
    otherwise. A component is the unit the conditional analysis actually masked, so
    annotating it describes the whole signal rather than the one window that scored best.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    gene_map = _load_gene_map(args.reference_dir)

    if args.components:
        units = _units_from_components(args.components, gene_map)
        print(f"Components mode: {len(units)} components from {args.components}")
        if args.significance_column or args.all_signals:
            print("  Note: --significance-column and --all-signals do not apply with "
                  "--components; the file already holds the called components.")
    else:
        # Load FDR file and work out which layout it has. A mutation-rate run corrects
        # each test separately and reports the max, so it has no plain `fdr` or
        # `p_value` column.
        df_fdr = pd.read_csv(args.fdr_file, sep='\t')
        schema = _fdr_schema(df_fdr)
        sig_col = args.significance_column or schema['significance_column']
        p_col = schema['p_column'][args.test]
        radius_cols = schema['radius_columns'][args.test]
        if schema is MU_FDR_SCHEMA:
            print(f"Mutation-rate results detected; ranking by {p_col}, "
                  f"significance from {sig_col}.")
        for col in (sig_col, p_col):
            if col not in df_fdr.columns:
                raise ValueError(
                    f"Column '{col}' not found in {args.fdr_file}. "
                    f"Available: {list(df_fdr.columns)}"
                )
        units = _units_from_fdr(args, df_fdr, sig_col, p_col, radius_cols, gene_map)
        if not units:
            print(f"No proteins with {sig_col} < {args.significance_cutoff} "
                  f"in {args.fdr_file}")
            return 0

    annotations_dir = args.annotations_dir or os.path.join(args.results_dir, 'annotations')
    cache_dir = args.cache_dir or os.path.join(annotations_dir, 'cache')
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    pval_path = args.pval_file
    nbhd_dir = os.path.join(args.results_dir, 'neighborhoods')
    os.makedirs(nbhd_dir, exist_ok=True)

    # The _mut PSE is per protein but a protein can carry several components, so they
    # are collected up front and all drawn into it as separate toggleable objects.
    components_by_protein = defaultdict(list)
    for unit in units:
        if unit['kind'] == 'component' and unit.get('centers'):
            components_by_protein[unit['uniprot_id']].append(
                {'label': unit['label'], 'centers': unit['centers']})

    n = len(units)
    visualized = set()
    annotated = {}
    for i, unit in enumerate(units, 1):
        uniprot_id = unit['uniprot_id']
        gene_name = unit['gene_name']
        center = unit['center']
        kind = unit['kind']

        print(f"\n{'='*60}")
        print(f"[{i}/{n}] {gene_name} ({uniprot_id})")
        print(f"  {unit['info']}")
        print('='*60)

        # 1. Visualization. The whole-chain PSEs are per protein; the component PSE is
        # per component, since that is what it highlights.
        if not args.skip_visualization:
            if uniprot_id not in visualized:
                print("  Running visualization...")
                try:
                    from pymol_code import run_all
                    run_all(uniprot_id, args.results_dir, args.reference_dir,
                            pval_file=pval_path, test=args.test,
                            components=components_by_protein.get(uniprot_id),
                            gene_name=gene_name)
                    print("  Visualization done.")
                except Exception as e:
                    print(f"  Warning: visualization failed: {e}")
                visualized.add(uniprot_id)
            if kind == 'component':
                try:
                    from pymol_code import pymol_component
                    pymol_component(uniprot_id, args.reference_dir, args.results_dir,
                                    pval_path, unit['residues'], unit['label'],
                                    gene_name=gene_name, test=args.test,
                                    centers=unit.get('centers'))
                except Exception as e:
                    print(f"  Warning: component visualization failed: {e}")

        # 2. Annotation (per protein -- several components can share one protein)
        features_path = os.path.join(
            annotations_dir, f'{uniprot_id}_features.tsv'
        )
        if uniprot_id in annotated:
            features_df = annotated[uniprot_id]
        else:
            print(f"  Annotating functional sites -> {features_path}")
            try:
                features_df = _annotate_and_save(
                    uniprot_id, features_path, cache_dir,
                    confidence_threshold=args.confidence_threshold
                )
            except Exception as e:
                print(f"  Warning: annotation failed: {e}")
                features_df = pd.DataFrame()
            annotated[uniprot_id] = features_df

        # 3. The region's residues. A component carries its own set; a neighborhood is
        # rebuilt from the structure at the radius its scan chose.
        if unit['residues'] is not None:
            nbhd = unit['residues']
            print(f"  Component has {len(nbhd)} residues")
        else:
            radius = unit['radius']
            print(f"  Computing neighborhood (center={center}, radius={radius}, "
                  f"pae={args.pae_cutoff})...")
            try:
                nbhd = _compute_nbhd(
                    uniprot_id, center, args.reference_dir,
                    radius, args.pae_cutoff
                )
                print(f"  Neighborhood has {len(nbhd)} residues")
            except Exception as e:
                print(f"  Warning: neighborhood computation failed: {e}")
                continue

        # 4. Load per-residue mutation data
        try:
            df_mut, mu_mode = _load_mut_data(pval_path, uniprot_id, args.test)
        except Exception as e:
            print(f"  Warning: could not read mutation data: {e}")
            continue

        # 5. Per-residue table for the region
        second_col = 'mu' if mu_mode else 'ac_control'
        stem = f'{safe_name(gene_name)}_{uniprot_id}_{unit["label"]}'
        suffix = 'component' if kind == 'component' else 'nbhd'
        res_path = os.path.join(nbhd_dir, f'{stem}_{suffix}.tsv')
        df_mut[df_mut['aa_pos'].isin(nbhd)][['aa_pos', 'ac_case', second_col]].to_csv(
            res_path, sep='\t', index=False
        )
        print(f"  Saved {kind} residues to {res_path}")

        # 6. Features table (skip if no features)
        if len(features_df) == 0:
            print(f"  No features available; skipping {kind}-features table.")
            continue

        result_df = _nbhd_features_table(nbhd, df_mut, features_df, mu_mode=mu_mode)
        feat_path = os.path.join(nbhd_dir, f'{stem}_{suffix}_features.tsv')
        result_df.to_csv(feat_path, sep='\t', index=False)
        print(f"  Saved {len(result_df)} overlapping features to {feat_path}")

    label = ('components' if args.components
             else ('signals' if args.all_signals else 'proteins'))
    print(f"\nDone. Processed {n} {label}.")
    return 0


def pse_to_png(input_pse, ray_trace_gain=0.15):
    from pymol import cmd

    output_png = input_pse.replace(".pse", ".png")
    print(f"[INFO] Loading PSE: {input_pse}")
    print(f"[INFO] Output PNG:  {output_png}")

    cmd.load(input_pse)

    # Rendering settings
    cmd.set("ambient", 0.5)
    cmd.set("ray_shadows", 0)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_gain", ray_trace_gain)
    cmd.set("ray_opaque_background", 1)
    cmd.bg_color("white")

    width  = 2400
    height = 1800
    dpi    = 300

    cmd.ray(width, height)
    cmd.png(output_png, width, height, dpi, 0, 0)

    print(f"[INFO] Saved PNG to: {output_png}")
    cmd.quit(0)


def pse_to_gif(input_pse, n_frames=36, width=640, height=480, fps=15, ray_trace_gain=0.15):
    from pymol import cmd
    import tempfile

    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] Pillow is required: pip install Pillow")
        return

    output_gif = input_pse.replace(".pse", ".gif")
    print(f"[INFO] Loading PSE: {input_pse}")
    print(f"[INFO] Output GIF:  {output_gif} ({n_frames} frames at {fps} fps)")

    cmd.load(input_pse)
    cmd.set("ambient", 0.5)
    cmd.set("ray_shadows", 0)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_gain", ray_trace_gain)
    cmd.set("ray_opaque_background", 1)
    cmd.bg_color("white")

    tmpdir = tempfile.mkdtemp()
    frame_paths = []
    step = 360.0 / n_frames

    try:
        for i in range(n_frames):
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            print(f"[INFO] Rendering frame {i + 1}/{n_frames}...")
            cmd.png(frame_path, width, height, dpi=72, ray=1, quiet=1)
            frame_paths.append(frame_path)
            cmd.turn("y", step)

        print("[INFO] Stitching frames into GIF...")
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        duration_ms = int(1000 / fps)
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=duration_ms,
            optimize=False,
        )
    finally:
        for p in frame_paths:
            if os.path.exists(p):
                os.remove(p)
        if os.path.isdir(tmpdir):
            os.rmdir(tmpdir)

    print(f"[INFO] Saved GIF to: {output_gif}")
    cmd.quit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Protein structure interpretation tools',
        epilog=(
            "Each subcommand has its own options, which plain -h does not show.\n"
            "  %(prog)s --help-all           every subcommand's options\n"
            "  %(prog)s <subcommand> -h      just one subcommand's options"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    class _HelpAll(argparse.Action):
        """
        Print the top-level help and then every subcommand's.

        argparse shows subcommand options only under `prog subcommand -h`, so plain -h
        lists the subcommand names and nothing else -- there is no way to see the whole
        interface at once, which is the first thing anyone wants.
        """

        def __call__(self, parser_, namespace, values, option_string=None):
            parser_.print_help()
            for name, sub in subparsers.choices.items():
                print('\n' + '=' * 72)
                print(f'  {parser_.prog} {name}')
                print('=' * 72)
                print(sub.format_help())
            parser_.exit()

    parser.add_argument('--help-all', action=_HelpAll, nargs=0,
                        help="show this help, plus every subcommand's options, and exit")

    def add_test_arg(p):
        """--test selects the null for a mutation-rate p-value file; ignored otherwise."""
        p.add_argument('--test', choices=['binomial', 'poisson'], default='binomial',
                       help='For mutation-rate results, which null drives the output: '
                            'binomial = Test B, the within-gene comparison and the test '
                            'of interest (default); poisson = Test A, the regional-'
                            'constraint guard, whose standalone significance is not '
                            'calibrated. Ignored for standard 3DNT results.')

    # -- annotate subcommand --
    p_annotate = subparsers.add_parser(
        'annotate',
        help='Transfer functional site annotations from protein family members',
        description='Transfers functional site annotations from protein family members and outputs '
                    'a table showing each unique feature with its positions.'
    )
    p_annotate.add_argument('uniprot_id', help='UniProt ID of protein to annotate')
    p_annotate.add_argument('output_file', help='Output TSV file path')
    p_annotate.add_argument('--cache-dir', default='cache', help='Cache directory (default: cache)')
    p_annotate.add_argument('--confidence-threshold', type=float, default=0.8,
                            help='Minimum alignment confidence to include (default: 0.8)')
    p_annotate.add_argument('--max-family-members', type=int, default=50,
                            help='Cap the family at this many best-annotated members '
                                 '(default: 50). UniProt returns up to 500 for a large '
                                 'family, and each one costs a full-length alignment')

    # -- nbhd-features subcommand --
    p_nbhd = subparsers.add_parser(
        'nbhd-features',
        help='Cross-reference a neighborhood with a functional features TSV',
        description='Identifies which functional features (from interpret.py annotate) '
                    'overlap with a structural neighborhood centered at a given residue.'
    )
    p_nbhd.add_argument('--features', required=True,
                        help='Features TSV from interpret.py annotate')
    p_nbhd.add_argument('--reference-dir', required=True,
                        help='Reference directory (same as for run.py)')
    p_nbhd.add_argument('--results-dir', required=True,
                        help='Results directory (for output files)')
    p_nbhd.add_argument('--pval-file', required=True,
                        help='Full path to p_values.h5 HDF5 file')
    p_nbhd.add_argument('--uniprot', required=True, help='UniProt ID')
    p_nbhd.add_argument('--aa-pos', required=True, type=int,
                        help='Neighborhood center residue position')
    p_nbhd.add_argument('--output', default=None,
                        help='Output TSV file; if omitted, prints to stdout')
    p_nbhd.add_argument('--radius', type=float, default=15.0,
                        help='Neighborhood radius in Angstroms (default: 15.0)')
    p_nbhd.add_argument('--pae-cutoff', type=float, default=15.0,
                        help='Maximum PAE value (default: 15.0)')
    add_test_arg(p_nbhd)

    # -- visualize subcommand --
    p_vis = subparsers.add_parser(
        'visualize',
        help='Generate PyMOL visualizations for a protein',
    )
    p_vis.add_argument('--uniprot', required=True, help='UniProt ID')
    p_vis.add_argument('--results-dir', required=True, help='Results directory')
    p_vis.add_argument('--reference-dir', required=True, help='Reference directory')
    p_vis.add_argument('--pval-file', required=True, help='Full path to p_values.h5 HDF5 file')
    add_test_arg(p_vis)

    # -- run_all subcommand --
    p_analyze = subparsers.add_parser(
        'run_all',
        help='Run the full interpretation pipeline for every significant protein in an FDR file',
        description='For each significant protein: visualization, annotation, top-neighborhood '
                    'residue table, and neighborhood × features table.',
    )
    p_analyze.add_argument('--fdr-file', default=None,
                           help='FDR results TSV (output of run.py --run-3dnt), or the '
                                '.conditional.tsv from a --conditional run. Required '
                                'unless --components is given')
    p_analyze.add_argument('--components', default=None,
                           help='<stem>.components.tsv from a --conditional run. Switches '
                                'the unit of analysis from one neighborhood to one '
                                'component: the whole set of overlapping neighborhoods '
                                'reporting a signal, which is what the conditional '
                                'analysis masked. Residues come from the file, so no '
                                'structure is needed to rebuild them')
    p_analyze.add_argument('--results-dir', required=True,
                           help='Results directory (must contain p_values.h5)')
    p_analyze.add_argument('--reference-dir', required=True,
                           help='Reference directory')
    p_analyze.add_argument('--significance-column', default=None,
                           help='Column to use for significance cutoff. Default depends '
                                'on the file: "fdr" for standard 3DNT results, "fdr_max" '
                                'for mutation-rate results (a neighborhood passes only '
                                'if it passes both tests)')
    p_analyze.add_argument('--all-signals', action='store_true', default=False,
                           help='Process every significant row rather than only the best '
                                'neighborhood per protein. Use with the .conditional.tsv '
                                'from a --conditional run, which lists one row per '
                                'independent signal')
    p_analyze.add_argument('--significance-cutoff', type=float, default=0.05,
                           help='Significance cutoff (default: 0.05)')
    p_analyze.add_argument('--annotations-dir', default=None,
                           help='Directory for annotation files (default: {results-dir}/annotations)')
    p_analyze.add_argument('--cache-dir', default=None,
                           help='Cache directory for UniProt API calls (default: {annotations-dir}/cache)')
    p_analyze.add_argument('--pval-file', required=True,
                           help='Full path to p_values.h5 HDF5 file')
    p_analyze.add_argument('--radius', type=float, default=15.0,
                           help='Fallback neighborhood radius in Angstroms (default: 15.0). '
                                'The radius the scan recorded for each hit is used when '
                                'the results file has one')
    p_analyze.add_argument('--pae-cutoff', type=float, default=15.0,
                           help='Maximum PAE value (default: 15.0)')
    p_analyze.add_argument('--confidence-threshold', type=float, default=0.8,
                           help='Alignment confidence threshold for annotation (default: 0.8)')
    p_analyze.add_argument('--skip-visualization', action='store_true', default=False,
                           help='Skip PyMOL visualization step')
    add_test_arg(p_analyze)

    # -- color subcommand --
    p_color = subparsers.add_parser(
        'color',
        help='Color a chain in a PyMOL session by per-residue ratio values',
        description='Colors a specific chain in an mmCIF structure by per-residue ratio values '
                    'and saves a PyMOL session file.'
    )
    p_color.add_argument('--tsv', default=None,
                         help='TSV file with uniprot_id, aa_pos, ratio columns')
    p_color.add_argument('--pval-file', default=None,
                         help='Full path to p_values.h5 HDF5 file (preferred over --tsv for smooth gradients)')
    p_color.add_argument('--input-cif', required=True, help='Input structure (.cif)')
    p_color.add_argument('--output-pse', required=True, help='Output PyMOL session (.pse)')
    p_color.add_argument('--chain', required=True, help='Target chain ID (e.g. A)')
    p_color.add_argument('--uniprot', required=True, help='Target UniProt ID')
    add_test_arg(p_color)

    # -- pse to png --
    p_pse_to_png = subparsers.add_parser(
        'pse_to_png',
        help='Convert a PyMOL session file (.pse) to a PNG image',
        description='Loads a PyMOL session file and saves a PNG image of the current view.'
    )
    p_pse_to_png.add_argument('input_pse', help='Input PyMOL session file (.pse)')
    p_pse_to_png.add_argument('--ray-trace-gain', type=float, default=0.15,
                               help='Outline thickness (default: 0.15; use smaller for large complexes)')

    p_pse_to_gif = subparsers.add_parser(
        'pse_to_gif',
        help='Convert a PyMOL session file (.pse) to a rotating GIF',
        description='Loads a PyMOL session file and saves a GIF of a full y-axis rotation.'
    )
    p_pse_to_gif.add_argument('input_pse', help='Input PyMOL session file (.pse)')
    p_pse_to_gif.add_argument('--n-frames', type=int, default=36,
                               help='Number of frames for one full rotation (default: 36)')
    p_pse_to_gif.add_argument('--width', type=int, default=640,
                               help='Frame width in pixels (default: 640)')
    p_pse_to_gif.add_argument('--height', type=int, default=480,
                               help='Frame height in pixels (default: 480)')
    p_pse_to_gif.add_argument('--fps', type=int, default=15,
                               help='Frames per second (default: 15)')
    p_pse_to_gif.add_argument('--ray-trace-gain', type=float, default=0.15,
                               help='Outline thickness (default: 0.15; use smaller for large complexes)')

    args = parser.parse_args()

    if args.command == 'run_all' and not (args.fdr_file or args.components):
        parser.error('run_all needs --fdr-file or --components')

    if args.command == 'annotate':
        return cmd_annotate(args)
    elif args.command == 'nbhd-features':
        return cmd_nbhd_features(args)
    elif args.command == 'visualize':
        return cmd_visualize(args)
    elif args.command == 'run_all':
        return cmd_analyze(args)
    elif args.command == 'color':
        return cmd_color(args)
    elif args.command == 'pse_to_png':
        return pse_to_png(args.input_pse, args.ray_trace_gain)
    elif args.command == 'pse_to_gif':
        return pse_to_gif(args.input_pse, args.n_frames, args.width, args.height, args.fps, args.ray_trace_gain)


if __name__ == '__main__':
    sys.exit(main())
