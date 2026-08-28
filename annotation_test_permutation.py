"""Annotation test: is a protein annotation enriched for case vs control rare-variant burden.

Per protein: build a binary (n_annotations, seq_len) matrix from the annotation table
(long or wide format, see the converters below), optionally expand local annotations
(binding site, active site, ...) to their 3D structural neighborhood, then test each
annotation against a per-residue permutation null that fixes the total (case+control)
allele count at each residue and redraws case/control labels (multivariate
hypergeometric). Local FDR/FWER come from that per-protein null; global_fdr/global_fwer
pool the null p-values across all tested proteins.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from empirical_fdr import _compute_false_discoveries, _compute_fwer
from logger_config import get_logger
from scan_test import get_case_control_ac_matrix
from utils import get_adjacency_matrix, get_chi2_pvals_case_control

logger = get_logger(__name__)


def load_protein_lengths(reference_dir: str | Path) -> dict[str, int]:
    """uniprot_id -> sequence length, from a structure-informed-rvas reference dir."""
    guide = pd.read_csv(
        Path(reference_dir) / "protein_sequence_guide.tsv",
        sep="\t",
        usecols=["uniprot_id", "sequence"],
    )
    return dict(zip(guide["uniprot_id"], guide["sequence"].str.len()))


def filter_variants(
    rvas_df: pd.DataFrame, df_filter: pd.DataFrame | None, mode: str = "exclude"
) -> pd.DataFrame:
    """Restrict rvas_df to (mode="include") or drop (mode="exclude") rows matching
    df_filter, joined on whichever of uniprot_id/aa_pos/aa_ref/aa_alt are present in
    both - e.g. exclude a set of variants/positions/proteins, or include only a gene
    panel. None (the default) keeps rvas_df as-is.
    """
    if df_filter is None:
        return rvas_df
    on = [
        c
        for c in ["uniprot_id", "aa_pos", "aa_ref", "aa_alt"]
        if c in rvas_df.columns and c in df_filter.columns
    ]
    if mode == "include":
        return rvas_df.merge(df_filter[on].drop_duplicates(), on=on, how="inner")
    merged = rvas_df.merge(
        df_filter[on].drop_duplicates(), on=on, how="left", indicator=True
    )
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge")


def filter_proteins_by_allele_count(
    rvas_df: pd.DataFrame, min_alleles: int = 5
) -> list[str]:
    """uniprot_ids with more than min_alleles total case AND control alleles.

    Below that, the permutation null still runs, but there's not enough signal for the
    result to mean much - drop these proteins rather than let them pad the total test count.
    """
    totals = rvas_df.groupby("uniprot_id")[["ac_case", "ac_control"]].sum()
    return totals[
        (totals["ac_case"] > min_alleles) & (totals["ac_control"] > min_alleles)
    ].index.tolist()


def convert_long_annotations_to_binary_matrix(
    ann_df: pd.DataFrame,
    seq_len: int | None = None,
    ann_include: list[str] | None = None,
    ann_id_columns: tuple[str, ...] = ("concept_type", "concept_value"),
):
    """Binary matrix from long-format annotations (one row per annotation-position, or
    per annotation-interval), single protein.

    ann_df columns: concept_type, concept_value, start, end, and optionally geometry -
    or, for single-residue annotations (one row per residue, e.g. a long-format cluster
    table with uniprot_id/aa_pos/cluster_col/cluster_id), aa_pos instead of start/end.
    ann_id_columns picks which columns are joined into the annotation id, so a cluster
    table like that is handled by passing ann_id_columns=("cluster_col", "cluster_id") -
    no separate format needed. Positions are 1-based, inclusive. geometry == "pair" (e.g.
    disulfide bonds) marks only start and end; anything else ("range"/"single"/missing/
    point) fills [start, end]. ann_include restricts to rows whose ann_id_columns[0]
    value is in this list (e.g. ann_include=["Domain"] with the default id columns to
    test only Domain annotations). Annotations that end up covering the exact same
    position set (e.g. a cluster that's stable across two clustering resolutions) are
    merged into one row so they're tested once, not twice - their ids are kept, joined
    by "|" in the result's annotation_id.

    Returns:
        label_binary_matrix: shape (n_annotations, seq_len), 1 at annotated positions.
        row_meta: dict mapping row index -> [annotation id, ...] (ids of merged rows).
    """
    if "start" not in ann_df.columns and "aa_pos" in ann_df.columns:
        ann_df = ann_df.assign(start=ann_df["aa_pos"], end=ann_df["aa_pos"])
    ann_df = ann_df.dropna(subset=["start", "end"])
    if ann_include is not None:
        ann_df = ann_df[ann_df[ann_id_columns[0]].isin(ann_include)]
    if "geometry" not in ann_df.columns:
        ann_df = ann_df.assign(geometry="range")

    if ann_df.empty:
        return np.zeros((0, seq_len or 0), dtype=np.uint8), {}

    if seq_len is None:
        seq_len = int(ann_df["end"].max())

    label_binary_matrix = []
    row_meta = {}
    signature_to_row_idx = {}
    for ann_id, grp in ann_df.groupby(list(ann_id_columns), sort=False):
        ann_id = "_".join([str(x) for x in ann_id])
        row = np.zeros(seq_len, dtype=np.uint8)
        for start, end, geometry in zip(grp["start"], grp["end"], grp["geometry"]):
            start, end = int(start), int(end)
            if geometry == "pair":
                endpoints = [p - 1 for p in (start, end) if p <= seq_len]
                row[endpoints] = 1
            else:
                row[start - 1 : end] = 1

        signature = tuple(np.flatnonzero(row).tolist())
        existing_idx = signature_to_row_idx.get(signature)
        if existing_idx is None:
            row_meta[len(label_binary_matrix)] = [ann_id]
            signature_to_row_idx[signature] = len(label_binary_matrix)
            label_binary_matrix.append(row)
        else:
            row_meta[existing_idx].append(ann_id)

    return np.array(label_binary_matrix, dtype=np.uint8), row_meta


def convert_wide_annotations_to_binary_matrix(
    ann_df: pd.DataFrame,
    seq_len: int | None = None,
    ann_id_columns: list[str] | None = None,
):
    """Binary matrix from wide-format residue annotations, single protein.

    One row per residue (aa_pos); each of `ann_id_columns` is a key whose value becomes
    an annotation id (no concatenation - each selected column's values are already
    annotation ids, or become one via the {column}_{value} disambiguation below). Two
    supported layouts: a single generic column (e.g. "annotation") whose values are
    already fully-qualified ids ("Binding site", "Domain_SET", ...) - used as-is; or
    several columns (concept types / clustering layers, one column per method) where the
    id is "{column}_{value}" to disambiguate e.g. the same cluster id across layers.
    Missing values (NaN) mean "not annotated in this column" and are dropped.

    Returns:
        label_binary_matrix: shape (n_annotations, seq_len), 1 at annotated positions.
        row_meta: dict mapping row index -> [id, ...] (annotations that share the exact
            same set of positions are merged into one row).
    """
    label_binary_matrix = []
    row_meta = {}
    signature_to_row_idx = {}

    if seq_len is None:
        seq_len = int(ann_df["aa_pos"].max())

    if ann_id_columns is None:
        ann_id_columns = [c for c in ann_df.columns if c not in ["uniprot_id", "aa_pos"]]
    single_column = len(ann_id_columns) == 1

    aa_pos_0_based = ann_df["aa_pos"].to_numpy(dtype=np.int64) - 1

    for col in ann_id_columns:
        not_na = ann_df[col].notna().to_numpy()
        col_values = ann_df[col].to_numpy()[not_na]
        col_positions = aa_pos_0_based[not_na]
        unique_values, inverse = np.unique(col_values, return_inverse=True)

        for value_idx, value in enumerate(unique_values):
            positions = col_positions[inverse == value_idx]
            if positions.size == 0:
                continue

            signature = tuple(np.unique(positions).tolist())
            existing_idx = signature_to_row_idx.get(signature)
            ann_id = str(value) if single_column else f"{col}_{value}"

            if existing_idx is None:
                row = np.zeros(seq_len, dtype=np.uint8)
                row[list(signature)] = 1
                label_binary_matrix.append(row)
                row_idx = len(label_binary_matrix) - 1
                signature_to_row_idx[signature] = row_idx
                row_meta[row_idx] = [ann_id]
            else:
                row_meta[existing_idx].append(ann_id)

    return np.array(label_binary_matrix, dtype=np.uint8), row_meta


def get_protein_adjacency_matrix(
    uniprot_id: str, reference_dir: str, radius: float, pae_cutoff: float = 0
):
    """CA-CA distance (+ optional PAE) adjacency matrix for one protein.

    reference_dir contains pdb_pae_file_pos_guide.tsv, pdb_files/, pae_files/. Row/col i
    is aa_pos i+1. Returns None if uniprot_id isn't in the guide.
    """
    reference_dir = Path(reference_dir)
    return get_adjacency_matrix(
        reference_dir / "pdb_pae_file_pos_guide.tsv",
        reference_dir / "pdb_files",
        reference_dir / "pae_files",
        uniprot_id,
        radius,
        pae_cutoff,
    )


def expand_annotations_neighborhood(
    label_binary_matrix: np.ndarray,
    adjacency_matrix: np.ndarray,
    rows_to_expand: list[int] | None = None,
):
    """Expand selected annotation rows to their 3D structural neighborhood.

    label_binary_matrix: (n_annotations, seq_len), as returned by the
    convert_*_annotations_to_binary_matrix functions above - radius/pae_cutoff is baked
    into adjacency_matrix, so it applies uniformly to every row picked by rows_to_expand.
    adjacency_matrix: (n_residues, n_residues) from get_protein_adjacency_matrix.
    rows_to_expand: row indices to expand (e.g. only "Binding site"/"Active site" rows,
        picked from row_meta - broad concepts like "Domain" don't make sense to expand
        this way). Default: expand every row.

    Returns an array shaped like label_binary_matrix: rows not selected are returned
    unchanged; selected rows become their seed positions plus any 3D neighbor of a seed.
    """
    seq_len = label_binary_matrix.shape[1]
    n = min(seq_len, adjacency_matrix.shape[0])
    if adjacency_matrix.shape[0] != seq_len:
        warnings.warn(
            f"adjacency_matrix has {adjacency_matrix.shape[0]} residues, label matrix has "
            f"{seq_len} - restricting expansion to the first {n}.",
            UserWarning,
        )

    if rows_to_expand is None:
        rows_to_expand = range(label_binary_matrix.shape[0])

    adjacent = adjacency_matrix[:n, :n].astype(bool)
    expanded = label_binary_matrix.copy()
    for i in rows_to_expand:
        seeds = np.flatnonzero(label_binary_matrix[i, :n])
        expanded[i, :n] = 0
        if seeds.size:
            expanded[i, :n] = adjacent[:, seeds].any(axis=1)

    return expanded


def test_protein_annotations(
    uniprot_id: str,
    prot_rvas_df: pd.DataFrame,
    label_matrix: np.ndarray,
    row_meta: dict,
    seq_len: int,
    n_perm: int,
) -> pd.DataFrame:
    """Permutation test for one protein's annotations.

    Null: for each residue, total (case+control) allele count is fixed and case/control
    labels are redrawn n_perm times (multivariate hypergeometric, see
    scan_test.get_case_control_ac_matrix). p-value per annotation is chi-square
    (Fisher's exact for borderline cases), against the observed count and against each
    permutation, giving local_fdr/local_fwer within this protein.
    """
    case_ac_matrix, ctrl_ac_matrix = get_case_control_ac_matrix(
        prot_rvas_df, n_res=seq_len, n_sim=n_perm
    )
    case_ac_matrix = case_ac_matrix.astype(int)
    ctrl_ac_matrix = ctrl_ac_matrix.astype(int)

    ac_in_case = label_matrix @ case_ac_matrix  # (n_annotations, 1 + n_perm)
    ac_in_ctrl = label_matrix @ ctrl_ac_matrix

    total_ac_case = int(case_ac_matrix[:, 0].sum())
    total_ac_ctrl = int(ctrl_ac_matrix[:, 0].sum())

    pvals_table, potential_significant_table = get_chi2_pvals_case_control(
        ac_in_case, ac_in_ctrl, total_ac_case, total_ac_ctrl
    )
    or_table = np.ones_like(pvals_table)

    # Only run the exact test on (a, b) pairs that are both observed somewhere in this
    # protein's data (avoid computing it for every unobserved cell in the lookup table)
    # and flagged borderline by the chi-square pre-filter.
    observed_table = np.zeros_like(potential_significant_table, dtype=bool)
    a_obs, b_obs = ac_in_case.ravel(), ac_in_ctrl.ravel()
    valid_pairs = (a_obs + b_obs) > 0
    observed_table[a_obs[valid_pairs], b_obs[valid_pairs]] = True
    for a, b in np.argwhere(observed_table & potential_significant_table):
        table = [[a, total_ac_case - a], [b, total_ac_ctrl - b]]
        or_table[a, b], pvals_table[a, b] = fisher_exact(table)

    pvalues = pvals_table[ac_in_case, ac_in_ctrl]
    odds_ratios = or_table[ac_in_case, ac_in_ctrl]

    observed_pvals = pvalues[:, 0]
    null_pvals = pvalues[:, 1:]
    local_fwer = (null_pvals.min(axis=0)[None, :] <= observed_pvals[:, None]).mean(
        axis=1
    )
    local_fdr = (null_pvals <= observed_pvals[:, None]).mean(axis=1)

    pvals_df = pd.DataFrame(
        {
            "uniprot_id": uniprot_id,
            "annotation_id": ["|".join(row_meta[i]) for i in range(len(row_meta))],
            "p_value": observed_pvals,
            "local_fwer": local_fwer,
            "local_fdr": local_fdr,
            "odds_ratio": odds_ratios[:, 0],
            "ac_in_case": ac_in_case[:, 0],
            "ac_in_ctrl": ac_in_ctrl[:, 0],
        }
    )
    pvals_df.loc[:, [f"null_pval_{i}" for i in range(n_perm)]] = null_pvals
    return pvals_df


def summarize_annotation_test(
    per_protein_results: list[pd.DataFrame],
    n_perm: int,
    results_dir: Path,
    reference_dir: str | Path | None = None,
    cleanup_per_protein: bool = False,
) -> pd.DataFrame:
    """Pool per-protein permutation nulls into an empirical global FDR/FWER + a BH-FDR baseline.

    cleanup_per_protein: once results.parquet is written, delete the per-protein
    `{uniprot_id}_annotation_pvals.parquet` shards (results.parquet already carries every
    row from them, plus global FDR/FWER) and remove the now-empty per_protein/ dir.
    """
    all_pvals_df = pd.concat(per_protein_results, ignore_index=True)
    null_cols = [f"null_pval_{i}" for i in range(n_perm)]

    # global_fdr below is a step-down estimate and only valid computed in ascending
    # p_value order (rank i+1 divides the pooled-null count exceeded by observed p-values
    # up to that rank).
    df_pvals = all_pvals_df[
        [c for c in all_pvals_df.columns if c not in null_cols]
    ].sort_values("p_value", ascending=True, ignore_index=True)
    null_pvals_dict = {
        uniprot_id: grp[null_cols].to_numpy(dtype=float)
        for uniprot_id, grp in all_pvals_df.groupby("uniprot_id")
    }
    uniprot_ids = list(null_pvals_dict)
    assert sum(len(v) for v in null_pvals_dict.values()) == len(df_pvals)

    false_discoveries = _compute_false_discoveries(
        df_pvals=df_pvals,
        null_pvals_dict=null_pvals_dict,
        uniprot_ids=uniprot_ids,
        n_sims=n_perm,
    )
    df_pvals["false_discoveries_avg"] = false_discoveries
    df_pvals["global_fdr"] = [x / (i + 1) for i, x in enumerate(false_discoveries)]
    df_pvals["global_fdr"] = df_pvals["global_fdr"][::-1].cummin()[::-1]
    df_pvals["global_fwer"] = _compute_fwer(
        df_pvals=df_pvals,
        null_pvals_dict=null_pvals_dict,
        uniprot_ids=uniprot_ids,
        n_sims=n_perm,
    )
    df_pvals["bh_reject"], df_pvals["bh_fdr"] = fdrcorrection(
        df_pvals["p_value"].to_numpy(), alpha=0.05
    )

    if reference_dir is not None:
        guide = pd.read_csv(
            Path(reference_dir) / "protein_sequence_guide.tsv",
            sep="\t",
            usecols=["uniprot_id", "gene_name"],
        ).drop_duplicates("uniprot_id")
        df_pvals = df_pvals.merge(guide, on="uniprot_id", how="left")

    df_pvals.to_parquet(results_dir / "results.parquet", index=False)

    if cleanup_per_protein:
        per_protein_dir = results_dir / "per_protein"
        for shard in per_protein_dir.glob("*_annotation_pvals.parquet"):
            shard.unlink()
        per_protein_dir.rmdir()

    return df_pvals


def fallback_seq_len(prot_rvas_df: pd.DataFrame, prot_ann_df: pd.DataFrame) -> int:
    """Protein length when reference_dir is unset: a seq_len/length column on ann_df if present, else the max annotated/variant position."""
    for col in ("seq_len", "length"):
        if col in prot_ann_df.columns:
            return int(prot_ann_df[col].max())
    ann_positions = (
        prot_ann_df["end"] if "end" in prot_ann_df.columns else prot_ann_df["aa_pos"]
    )
    return int(max(prot_rvas_df["aa_pos"].max(), ann_positions.max()))


def rows_to_expand(row_meta: dict, expand_include: list[str]) -> list[int]:
    """row_meta indices whose annotation id starts with one of expand_include.

    Works across formats: long ids are "{ann_id_columns[0]}_{ann_id_columns[1]}..." and
    wide ids are "{column}" or "{column}_{value}" (see converters above) - in both cases
    the first id component is a prefix of the id.
    """
    return [
        i
        for i, ids in row_meta.items()
        if any(id_.startswith(prefix) for id_ in ids for prefix in expand_include)
    ]


def process_protein(
    uniprot_id: str,
    prot_rvas_df: pd.DataFrame,
    prot_ann_df: pd.DataFrame,
    seq_len: int | None,
    *,
    ann_format: str,
    ann_include: list[str] | None,
    ann_id_columns: tuple[str, ...] | list[str] | None,
    expand_radius: float,
    expand_include: list[str] | None,
    pae_cutoff: float,
    reference_dir: str | Path | None,
    n_perm: int,
    results_dir: Path,
) -> pd.DataFrame | None:
    """Build the label matrix, run the permutation test, and write the per-protein parquet."""
    seq_len = seq_len or fallback_seq_len(prot_rvas_df, prot_ann_df)

    if ann_format == "long":
        label_matrix, row_meta = convert_long_annotations_to_binary_matrix(
            prot_ann_df,
            seq_len=seq_len,
            ann_include=ann_include,
            ann_id_columns=ann_id_columns,
        )
    else:
        label_matrix, row_meta = convert_wide_annotations_to_binary_matrix(
            prot_ann_df, seq_len=seq_len, ann_id_columns=ann_id_columns
        )
    if label_matrix.shape[0] == 0:
        return None

    if expand_radius > 0:
        adjacency_matrix = get_protein_adjacency_matrix(
            uniprot_id, str(reference_dir), expand_radius, pae_cutoff
        )
        if adjacency_matrix is None:
            warnings.warn(
                f"{uniprot_id}: no structure in reference dir, skipping expansion",
                UserWarning,
            )
        else:
            expand_rows = (
                rows_to_expand(row_meta, expand_include) if expand_include else None
            )
            label_matrix = expand_annotations_neighborhood(
                label_matrix, adjacency_matrix, expand_rows
            )

    pvals_df = test_protein_annotations(
        uniprot_id, prot_rvas_df, label_matrix, row_meta, seq_len, n_perm
    )
    per_protein_dir = results_dir / "per_protein"
    per_protein_dir.mkdir(exist_ok=True)
    pvals_df.to_parquet(
        per_protein_dir / f"{uniprot_id}_annotation_pvals.parquet", index=False
    )
    return pvals_df


def run_annotation_test(
    rvas_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    *,
    ann_format: str = "long",
    ann_include: list[str] | None = None,
    ann_id_columns: tuple[str, ...] | list[str] | None = (
        "concept_type",
        "concept_value",
    ),
    expand_include: list[str] | None = None,
    expand_radius: float = 0,
    pae_cutoff: float = 0,
    reference_dir: str | Path | None = None,
    df_filter: pd.DataFrame | None = None,
    df_include: pd.DataFrame | None = None,
    min_alleles: int = 5,
    n_perm: int = 1000,
    n_workers: int = 1,
    results_dir: str | Path = "outputs/annotation_test",
    cleanup_per_protein: bool = False,
) -> pd.DataFrame:
    """Run the annotation test for every protein present in both rvas_df and ann_df.

    rvas_df: one row per (uniprot_id, aa_pos) with ac_case/ac_control.
    ann_df: annotations in `ann_format`:
        - "long" (default): one row per annotation-position, or per annotation-interval.
          concept_type/concept_value/start/end[/geometry], or (no start/end) aa_pos
          instead - e.g. a cluster table (uniprot_id/aa_pos/cluster_col/cluster_id) with
          ann_id_columns=("cluster_col", "cluster_id"); several rows can share an aa_pos.
          ann_id_columns picks which columns are joined into the annotation id (default
          ("concept_type", "concept_value")); ann_include restricts to rows whose
          ann_id_columns[0] value is in this list. See convert_long_annotations_to_binary_matrix.
        - "wide": one row per residue, ann_id_columns selects which columns are
          themselves annotation ids (no concatenation - a column's values must already be
          ids, or become one via the {column}_{value} disambiguation when >1 column is
          selected). See convert_wide_annotations_to_binary_matrix.
    expand_radius: if > 0, expand annotations to their 3D structural neighborhood
        (CA-CA distance within expand_radius Angstrom, + optional pae_cutoff) before
        testing. 0 (default) disables expansion entirely. Requires reference_dir.
    expand_include: restrict expansion to annotation ids starting with one of these
        (matched against row_meta, so it works for either ann_format), e.g.
        ["Binding site", "Active site"] - broad concepts (Domain, Chain, ...) are left as
        annotated. None (default) expands every annotation. Ignored if expand_radius <= 0.
    reference_dir: structure-informed-rvas reference dir (pdb_files/, pae_files/,
        pdb_pae_file_pos_guide.tsv, protein_sequence_guide.tsv). Used for protein length
        and, if expand_radius > 0, for the adjacency matrix.
    df_filter: variants/positions/proteins to exclude from rvas_df before testing, see
        filter_variants.
    df_include: variants/positions/proteins to restrict rvas_df to before testing (e.g. a
        gene panel) - applied together with df_filter, not instead of it.
    min_alleles: proteins need more than this many total case AND control alleles to be
        tested, see filter_proteins_by_allele_count.
    n_workers: joblib worker processes for the per-protein loop. -1 (default) uses all
        cores; 1 runs in-process (no joblib), useful for debugging/profiling.
    cleanup_per_protein: delete the per-protein parquet shards once results.parquet is
        written, see summarize_annotation_test.

    Writes one `{uniprot_id}_annotation_pvals.parquet` per protein plus a pooled
    `results.parquet` (global_fdr, global_fwer, bh_fdr) to results_dir, and returns the
    latter.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if expand_radius > 0 and reference_dir is None:
        raise ValueError("expand_radius > 0 requires reference_dir")

    rvas_df = filter_variants(rvas_df, df_include, mode="include")
    rvas_df = filter_variants(rvas_df, df_filter, mode="exclude")
    tested_proteins = set(filter_proteins_by_allele_count(rvas_df, min_alleles))
    uniprot_ids = sorted(tested_proteins & set(ann_df["uniprot_id"]))

    protein_lengths = load_protein_lengths(reference_dir) if reference_dir else {}

    # Group once up front - filtering each df with a `== uniprot_id` mask per protein is
    # an O(n_proteins * len(df)) full-table scan and dominates runtime on large tables.
    rvas_by_protein = dict(tuple(rvas_df.groupby("uniprot_id", sort=False)))
    ann_by_protein = dict(tuple(ann_df.groupby("uniprot_id", sort=False)))

    def _call(uniprot_id: str) -> pd.DataFrame | None:
        return process_protein(
            uniprot_id,
            rvas_by_protein[uniprot_id],
            ann_by_protein[uniprot_id],
            protein_lengths.get(uniprot_id),
            ann_format=ann_format,
            ann_include=ann_include,
            ann_id_columns=ann_id_columns,
            expand_radius=expand_radius,
            expand_include=expand_include,
            pae_cutoff=pae_cutoff,
            reference_dir=reference_dir,
            n_perm=n_perm,
            results_dir=results_dir,
        )

    logger.info(f"Running annotation test on {len(uniprot_ids)} proteins")
    if n_workers == 1:
        per_protein_results = [_call(uid) for uid in tqdm(uniprot_ids)]
    else:
        per_protein_results = Parallel(n_jobs=n_workers, backend="threading")(
            delayed(_call)(uid) for uid in tqdm(uniprot_ids)
        )
    per_protein_results = [r for r in per_protein_results if r is not None]

    return summarize_annotation_test(
        per_protein_results,
        n_perm,
        results_dir,
        reference_dir,
        cleanup_per_protein=cleanup_per_protein,
    )


def _read_table(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, sep="\t")
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rvas-path",
        type=Path,
        required=True,
        help="uniprot_id, aa_pos, ac_case, ac_control (.parquet or .tsv)",
    )
    parser.add_argument(
        "--ann-path",
        type=Path,
        required=True,
        help="Annotation table, long or wide format (.parquet or .tsv)",
    )
    parser.add_argument(
        "--ann-format",
        default="long",
        choices=["long", "wide"],
        help="'long' (one row per annotation-position/interval) or 'wide' (one row per residue)",
    )
    parser.add_argument(
        "--ann-id-columns",
        default="concept_type,concept_value",
        help="Long format: comma-separated columns joined into the annotation id, e.g. "
        "'cluster_col,cluster_id' for a cluster table. Wide format: comma-separated "
        "columns to use as annotation ids directly (no concatenation).",
    )
    parser.add_argument(
        "--ann-include",
        default=None,
        help="Comma-separated filter on ann_id_columns[0] (long format only), e.g. 'Domain'",
    )
    parser.add_argument(
        "--expand-radius",
        type=float,
        default=0,
        help="3D neighborhood radius in Angstrom; 0 (default) disables expansion",
    )
    parser.add_argument(
        "--expand-include",
        default=None,
        help="Comma-separated annotation id prefixes to expand, e.g. 'Binding site,Active site' (default: expand all)",
    )
    parser.add_argument(
        "--pae-cutoff",
        type=float,
        default=0,
        help="PAE cutoff for the adjacency matrix; 0 = distance only",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="structure-informed-rvas reference dir",
    )
    parser.add_argument(
        "--df-filter-path",
        type=Path,
        default=None,
        help="Variants/positions/proteins to exclude (uniprot_id[, aa_pos[, aa_ref, aa_alt]])",
    )
    parser.add_argument(
        "--df-include-path",
        type=Path,
        default=None,
        help="Restrict to these variants/positions/proteins, e.g. a gene panel (uniprot_id[, aa_pos[, aa_ref, aa_alt]])",
    )
    parser.add_argument(
        "--min-alleles",
        type=int,
        default=5,
        help="Proteins need more than this many total case AND control alleles",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=1000,
        help="Number of permutations for the empirical null",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Joblib worker processes; -1 = all cores, 1 = no joblib",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/annotation_test"),
        help="Output directory",
    )
    parser.add_argument(
        "--cleanup-per-protein",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete per-protein parquet shards once results.parquet is written",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_annotation_test(
        _read_table(args.rvas_path),
        _read_table(args.ann_path),
        ann_format=args.ann_format,
        ann_include=args.ann_include.split(",") if args.ann_include else None,
        ann_id_columns=tuple(args.ann_id_columns.split(",")),
        expand_include=args.expand_include.split(",") if args.expand_include else None,
        expand_radius=args.expand_radius,
        pae_cutoff=args.pae_cutoff,
        reference_dir=args.reference_dir,
        df_filter=_read_table(args.df_filter_path) if args.df_filter_path else None,
        df_include=_read_table(args.df_include_path) if args.df_include_path else None,
        min_alleles=args.min_alleles,
        n_perm=args.n_perm,
        n_workers=args.n_workers,
        results_dir=args.results_dir,
        cleanup_per_protein=args.cleanup_per_protein,
    )


if __name__ == "__main__":
    main()
