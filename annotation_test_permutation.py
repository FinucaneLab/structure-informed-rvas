"""Annotation test: is a protein annotation enriched for case vs control rare-variant burden.

Per protein: build a binary (n_annotations, seq_len) matrix from the annotation table
(long or wide format, see the converters below), optionally expand local annotations
(binding site, active site, ...) to their 3D structural neighborhood, then test each
annotation against a per-residue permutation null that fixes the total (case+control)
allele count at each residue and redraws case/control labels (multivariate
hypergeometric). Local FDR/FWER come from that per-protein null; global_fdr/global_fwer
pool the null p-values across all tested proteins.

With stratum_col, the null is redrawn within each stratum and the statistic is the
Cochran-Mantel-Haenszel sum of case alleles inside the annotation (exact conditional p).
"""
import argparse
import warnings
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.stats import chi2, fisher_exact
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from cmh import (
    cmh_expected_variance,
    mantel_haenszel_or,
    screen_and_exact_pvalues,
    summarize_strata,
)
from logger_config import get_logger
from utils import (
    DEFAULT_SCREEN_CHI2,
    get_distance_and_pae_matrices,
    threshold_adjacency_matrix,
    get_case_control_ac_matrix,
    get_chi2_pvals_case_control,
    get_stratified_case_ac_matrix,
)

logger = get_logger(__name__)


def protein_seed(uniprot_id: str, seed: int = 0) -> list[int]:
    """Per-protein RNG seed: reproducible and independent across proteins."""
    return [seed, zlib.crc32(uniprot_id.encode())]


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
    *,
    return_radii: bool = False,
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
    
    When a ``radius`` column is present, annotations with identical positions but
    different radii remain separate rows.

    Returns: result array containing the following:
        label_binary_matrix: shape (n_annotations, seq_len), 1 at annotated positions.
        row_meta: dict mapping row index -> [annotation id, ...] (ids of merged rows).
        ``return_radii=True`` additionally returns a dict mapping matrix row indices to their radius (or None)
    """

    if "start" not in ann_df.columns and "aa_pos" in ann_df.columns:
        ann_df = ann_df.assign(start=ann_df["aa_pos"], end=ann_df["aa_pos"])
    ann_df = ann_df.dropna(subset=["start", "end"])
    if ann_include is not None:
        ann_df = ann_df[ann_df[ann_id_columns[0]].isin(ann_include)]
    if "geometry" not in ann_df.columns:
        ann_df = ann_df.assign(geometry="range")

    if ann_df.empty:
        empty = (np.zeros((0, seq_len or 0), dtype=np.uint8), {}, {})
        return empty if return_radii else empty[:2]
    if seq_len is None:
        seq_len = int(ann_df["end"].max())

    label_binary_matrix, row_meta, row_radii = [], {}, {}
    signature_to_row_idx = {}
    # Radius is part of the grouping key: one annotation ID can have separate
    # instances at different radii, and equal footprints at different radii must not merge.
    group_columns = list(ann_id_columns) + ["radius"]
    for group_key, grp in ann_df.groupby(group_columns, sort=False, dropna=False):
        *ann_id_parts, radius_value = group_key
        ann_id = "_".join(str(x) for x in ann_id_parts)
        radius = None if pd.isna(radius_value) else float(radius_value)
        row = np.zeros(seq_len, dtype=np.uint8)
        for start, end, geometry in zip(grp["start"], grp["end"], grp["geometry"]):
            start, end = int(start), int(end)
            if geometry == "pair":
                endpoints = [p - 1 for p in (start, end) if 1 <= p <= seq_len]
                row[endpoints] = 1
            else:
                start0 = max(start - 1, 0)
                end0 = min(end, seq_len)
                if start0 < end0:
                    row[start0:end0] = 1

        signature = tuple(np.flatnonzero(row).tolist())
        key = (signature, radius)
        existing_idx = signature_to_row_idx.get(key)
        if existing_idx is None:
            row_idx = len(label_binary_matrix)
            row_meta[row_idx] = [ann_id]
            row_radii[row_idx] = radius
            signature_to_row_idx[key] = row_idx
            label_binary_matrix.append(row)
        else:
            row_meta[existing_idx].append(ann_id)

    result = (np.array(label_binary_matrix, dtype=np.uint8), row_meta, row_radii)
    return result if return_radii else result[:2]


def convert_wide_annotations_to_binary_matrix(
    ann_df: pd.DataFrame,
    seq_len: int | None = None,
    ann_id_columns: list[str] | None = None,
    *,
    return_radii: bool = False,
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
    ann_id_columns=None or ["*"] uses every column except uniprot_id/aa_pos/seq_len/
    length/end (the non-annotation columns fallback_seq_len also knows about).
    
    ``radius`` is treated as row-level metadata and applies to each annotation present
    at that residue. Rows are grouped by annotation ID and radius, so equal residue sets
    at different radii are preserved as distinct tests.

    Returns: result array containing the following:
        label_binary_matrix: shape (n_annotations, seq_len), 1 at annotated positions.
        row_meta: dict mapping row index -> [id, ...] (annotations that share the exact
            same set of positions are merged into one row).
        ``return_radii=True`` additionally returns a dict mapping matrix row indices to their radius (or None)
    """

    if ann_df.empty:
        empty = (np.zeros((0, seq_len or 0), dtype=np.uint8), {}, {})
        return empty if return_radii else empty[:2]
    if seq_len is None:
        seq_len = int(ann_df["aa_pos"].max())

    if ann_id_columns is None or list(ann_id_columns) == ["*"]:
        non_annotation_columns = [
            "uniprot_id", "aa_pos", "seq_len", "length", "end", "radius"
        ]
        ann_id_columns = [c for c in ann_df.columns if c not in non_annotation_columns]
    single_column = len(ann_id_columns) == 1
    radii = ann_df["radius"].to_numpy() if "radius" in ann_df.columns else np.full(len(ann_df), np.nan)
    positions = ann_df["aa_pos"].to_numpy(dtype=np.int64) - 1

    label_binary_matrix, row_meta, row_radii = [], {}, {}
    signature_to_row_idx = {}
    for col in ann_id_columns:
        not_na = ann_df[col].notna().to_numpy()
        col_values = ann_df[col].to_numpy()[not_na]
        col_positions = positions[not_na]
        col_radii = radii[not_na]
        # Group by both annotation value and radius; this supports radius changes along
        # a wide-format annotation column without merging distinct expansion definitions.
        grouping = pd.DataFrame({"value": col_values, "position": col_positions, "radius": col_radii})
        for (value, radius_value), grp in grouping.groupby(["value", "radius"], sort=False, dropna=False):
            signature = tuple(np.unique(grp["position"].to_numpy(dtype=np.int64)).tolist())
            if not signature:
                continue
            radius = None if pd.isna(radius_value) else float(radius_value)
            ann_id = str(value) if single_column else f"{col}_{value}"
            key = (signature, radius)
            existing_idx = signature_to_row_idx.get(key)
            if existing_idx is None:
                row = np.zeros(seq_len, dtype=np.uint8)
                valid_positions = [p for p in signature if 0 <= p < seq_len]
                row[valid_positions] = 1
                row_idx = len(label_binary_matrix)
                label_binary_matrix.append(row)
                row_meta[row_idx] = [ann_id]
                row_radii[row_idx] = radius
                signature_to_row_idx[key] = row_idx
            else:
                row_meta[existing_idx].append(ann_id)

    result = (np.array(label_binary_matrix, dtype=np.uint8), row_meta, row_radii)
    return result if return_radii else result[:2]


def get_protein_distance_pae_matrices(
    uniprot_id: str, reference_dir: str, pae_cutoff: float = 0
):
    """Raw CA-CA distance matrix (+ optional PAE matrix) for one protein.

    reference_dir contains pdb_pae_file_pos_guide.tsv, pdb_files/, pae_files/. Row/col i
    is aa_pos i+1. Returns (None, None) if uniprot_id isn't in the guide.

    A protein's annotations can carry several distinct expansion radii (see radius_groups
    in process_protein); these two matrices don't depend on radius, so callers should build
    them once per protein and reuse them across radii via utils.threshold_adjacency_matrix,
    rather than re-parsing the pdb/pae files (and rebuilding these O(n_residues^2) matrices)
    once per radius - the dominant cost for very large proteins (e.g. titin).
    """
    reference_dir = Path(reference_dir)
    return get_distance_and_pae_matrices(
        reference_dir / "pdb_pae_file_pos_guide.tsv",
        reference_dir / "pdb_files",
        reference_dir / "pae_files",
        uniprot_id,
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
    adjacency_matrix: (n_residues, n_residues) from utils.threshold_adjacency_matrix.
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


def local_fwer_fdr(observed_pvals: np.ndarray, null_pvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local_fwer = (null_pvals.min(axis=0)[None, :] <= observed_pvals[:, None]).mean(axis=1)
    local_fdr = (null_pvals <= observed_pvals[:, None]).mean(axis=1)
    return local_fwer, local_fdr


def test_protein_annotations(
    uniprot_id: str,
    prot_rvas_df: pd.DataFrame,
    label_matrix: np.ndarray,
    row_meta: dict,
    seq_len: int,
    n_perm: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Permutation test for one protein's annotations.

    Null: for each residue, total (case+control) allele count is fixed and case/control
    labels are redrawn n_perm times (multivariate hypergeometric, see
    scan_test.get_case_control_ac_matrix). p-value per annotation is chi-square
    (Fisher's exact for borderline cases), against the observed count and against each
    permutation, giving local_fdr/local_fwer within this protein.
    """
    case_ac_matrix, ctrl_ac_matrix = get_case_control_ac_matrix(
        prot_rvas_df, n_res=seq_len, n_sim=n_perm, seed=protein_seed(uniprot_id, seed)
    )
    case_ac_matrix = case_ac_matrix.astype(int)
    ctrl_ac_matrix = ctrl_ac_matrix.astype(int)

    # only residues with alleles matter; sparse int8 labels avoid numpy's slow integer matmul
    cols = np.flatnonzero(case_ac_matrix.sum(1) + ctrl_ac_matrix.sum(1))
    labels = csr_matrix(label_matrix[:, cols].astype(np.int8))
    ac_in_case = np.asarray(labels @ case_ac_matrix[cols])  # (n_annotations, 1 + n_perm)
    ac_in_ctrl = np.asarray(labels @ ctrl_ac_matrix[cols])

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
    local_fwer, local_fdr = local_fwer_fdr(observed_pvals, null_pvals)

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
    null_df = pd.DataFrame(null_pvals, columns=[f"null_pval_{i}" for i in range(n_perm)])
    return pd.concat([pvals_df, null_df], axis=1)


def test_protein_annotations_cmh(
    uniprot_id: str,
    prot_rvas_df: pd.DataFrame,
    label_matrix: np.ndarray,
    row_meta: dict,
    n_perm: int,
    stratum_col: str,
    seed: int = 0,
    screen_chi2: float = DEFAULT_SCREEN_CHI2,
) -> pd.DataFrame:
    """CMH-adjusted version of test_protein_annotations.

    Case/control labels are redrawn within each stratum (only on residues carrying alleles);
    the statistic per annotation is the sum over strata of case alleles inside it, with an
    exact conditional p-value where chi2 > screen_chi2 and the chi2 p-value elsewhere (not
    1.0, to avoid a point mass). Strata with no case/control variation inside the protein
    carry no information.
    """
    case_ac_matrix, case_by_stratum, total_by_stratum = get_stratified_case_ac_matrix(
        prot_rvas_df, stratum_col, label_matrix.shape[1], n_perm, protein_seed(uniprot_id, seed)
    )
    cols = np.flatnonzero(total_by_stratum.sum(1))
    labels = csr_matrix(label_matrix[:, cols].astype(np.int8))
    inside_totals = np.asarray(labels @ total_by_stratum[cols], dtype=np.int64)
    inside_case = np.asarray(labels @ case_by_stratum[cols], dtype=np.int64)
    case_totals, stratum_totals = case_by_stratum.sum(0), total_by_stratum.sum(0)
    expected, variance = cmh_expected_variance(inside_totals, case_totals, stratum_totals)
    cmh = {
        "s_case_inside": np.asarray(labels @ case_ac_matrix[cols], dtype=np.int64),
        "expected_sum": expected,
        "variance_sum": variance,
        "inside_totals_by_stratum": inside_totals,
        "inside_case_by_stratum": inside_case,
        "case_totals": case_totals,
        "stratum_totals": stratum_totals,
    }

    pvals, stat = screen_and_exact_pvalues(cmh, screen_chi2)
    pvals = np.where(stat > screen_chi2, pvals, chi2.sf(stat, 1))
    local_fwer, local_fdr = local_fwer_fdr(pvals[:, 0], pvals[:, 1:])

    inside_totals = cmh["inside_totals_by_stratum"]
    case_totals, stratum_totals = cmh["case_totals"], cmh["stratum_totals"]
    informative = (case_totals > 0) & (case_totals < stratum_totals)
    ac_in_case = cmh["s_case_inside"][:, 0]

    pvals_df = pd.DataFrame(
        {
            "uniprot_id": uniprot_id,
            "annotation_id": ["|".join(row_meta[i]) for i in range(len(row_meta))],
            "p_value": pvals[:, 0],
            "local_fwer": local_fwer,
            "local_fdr": local_fdr,
            "odds_ratio": mantel_haenszel_or(
                cmh["inside_case_by_stratum"], inside_totals, case_totals, stratum_totals
            ),
            "ac_in_case": ac_in_case,
            "ac_in_ctrl": inside_totals.sum(axis=1) - ac_in_case,
            "cmh_chisq": stat[:, 0],
            "cmh_variance": cmh["variance_sum"],
            "n_informative_strata": int(informative.sum()),
            "frac_ac_informative": stratum_totals[informative].sum() / max(stratum_totals.sum(), 1),
        }
    )
    null_df = pd.DataFrame(pvals[:, 1:], columns=[f"null_pval_{i}" for i in range(n_perm)])
    return pd.concat([pvals_df, null_df], axis=1)


class NullPvalAccumulator:
    """Pools per-protein permutation nulls for global_fdr/global_fwer one protein at a
    time, so the full (n_proteins, n_annotations, n_perm) null matrix is never in memory
    at once - only a running reduction (sub-threshold null p-values, per-sim min)."""

    def __init__(self, n_perm: int, large_p_threshold: float = 0.05):
        self.n_perm = n_perm
        self.large_p_threshold = large_p_threshold
        self.null_cols = [f"null_pval_{i}" for i in range(n_perm)]
        self.obs_frames: list[pd.DataFrame] = []
        self.pooled_null_chunks: list[np.ndarray] = []
        self.min_pvals_per_sim = np.ones(n_perm)

    def add(self, pvals_df: pd.DataFrame) -> None:
        null_arr = pvals_df[self.null_cols].to_numpy(dtype=float)
        self.pooled_null_chunks.append(null_arr[null_arr < self.large_p_threshold])
        self.min_pvals_per_sim = np.minimum(self.min_pvals_per_sim, null_arr.min(axis=0))
        self.obs_frames.append(pvals_df.drop(columns=self.null_cols))

    def finalize(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        # global_fdr (computed downstream) is a step-down estimate and only valid computed
        # in ascending p_value order (rank i+1 divides the pooled-null count exceeded by
        # observed p-values up to that rank).
        df_pvals = pd.concat(self.obs_frames, ignore_index=True).sort_values(
            "p_value", ascending=True, ignore_index=True
        )
        pooled_null_pvals = np.sort(np.concatenate(self.pooled_null_chunks))
        return df_pvals, pooled_null_pvals, self.min_pvals_per_sim


def compute_global_fdr(
    df_pvals: pd.DataFrame,
    pooled_null_pvals: np.ndarray,
    n_perm: int,
    large_p_threshold: float = 0.05,
) -> np.ndarray:
    """Same math as empirical_fdr._compute_false_discoveries, but takes the pooled+sorted
    sub-threshold null p-values directly instead of a per-protein null_pvals_dict."""
    p_value = df_pvals["p_value"].to_numpy()
    mask = p_value <= large_p_threshold
    false_discoveries = np.full(len(df_pvals), float(len(df_pvals)))
    if mask.any():
        false_discoveries[mask] = (
            np.searchsorted(pooled_null_pvals, p_value[mask], side="right") / n_perm
        )
    return false_discoveries


def compute_global_fwer(df_pvals: pd.DataFrame, min_pvals_per_sim: np.ndarray) -> np.ndarray:
    """Same math as empirical_fdr._compute_fwer, but takes the already-pooled per-simulation
    minimum directly."""
    p_obs = df_pvals["p_value"].to_numpy().reshape(-1, 1)
    return np.mean(min_pvals_per_sim.reshape(1, -1) <= p_obs, axis=1)


def read_per_protein_shards(results_dir: str | Path) -> Iterator[pd.DataFrame]:
    """Lazily read back the per-protein parquet shards written by process_protein, one at a
    time - lets summarize_annotation_test rebuild results.* from a completed/crashed run
    without re-running any permutation tests."""
    per_protein_dir = Path(results_dir) / "per_protein"
    for shard in sorted(per_protein_dir.glob("*_annotation_pvals.parquet")):
        yield pd.read_parquet(shard)


def summarize_annotation_test(
    per_protein_results: Iterable[pd.DataFrame],
    n_perm: int,
    results_dir: Path,
    reference_dir: str | Path | None = None,
    cleanup_per_protein: bool = False,
    format: str = "tsv.gz",
    result_fn: str = "results",
) -> pd.DataFrame:
    """Pool per-protein permutation nulls into an empirical global FDR/FWER + a BH-FDR baseline.

    per_protein_results: an in-memory list (as run_annotation_test builds it) or any other
    iterable of per-protein dataframes, e.g. read_per_protein_shards(results_dir) to rebuild
    results.* from disk without rerunning permutations.

    format: "tsv.gz" (default) or "parquet" for the pooled results file. Per-protein shards
    are always parquet.

    cleanup_per_protein: once results.* is written, delete the per-protein
    `{uniprot_id}_annotation_pvals.parquet` shards (results.* already carries every row
    from them, plus global FDR/FWER) and remove the now-empty per_protein/ dir.
    """
    accumulator = NullPvalAccumulator(n_perm)
    for pvals_df in per_protein_results:
        accumulator.add(pvals_df)
    df_pvals, pooled_null_pvals, min_pvals_per_sim = accumulator.finalize()

    false_discoveries = compute_global_fdr(df_pvals, pooled_null_pvals, n_perm)
    df_pvals["false_discoveries_avg"] = false_discoveries
    df_pvals["global_fdr"] = [x / (i + 1) for i, x in enumerate(false_discoveries)]
    df_pvals["global_fdr"] = df_pvals["global_fdr"][::-1].cummin()[::-1]
    df_pvals["global_fwer"] = compute_global_fwer(df_pvals, min_pvals_per_sim)
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

    results_path = results_dir / (
        f"{result_fn}.parquet" if format == "parquet" else f"{result_fn}.tsv.gz"
    )
    if format == "parquet":
        df_pvals.to_parquet(results_path, index=False)
    else:
        df_pvals.to_csv(results_path, sep="\t", index=False)

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
    expand_include: list[str] | None,
    pae_cutoff: float,
    reference_dir: str | Path | None,
    n_perm: int,
    results_dir: Path,
    seed: int = 0,
    stratum_col: str | None = None,
) -> pd.DataFrame | None:
    """Build the label matrix, run the permutation test, and write the per-protein parquet."""
    seq_len = seq_len or fallback_seq_len(prot_rvas_df, prot_ann_df)

    if ann_format == "long":
        label_matrix, row_meta, row_radii = convert_long_annotations_to_binary_matrix(
            prot_ann_df,
            seq_len=seq_len,
            ann_include=ann_include,
            ann_id_columns=ann_id_columns,
            return_radii=True,
        )
    else:
        label_matrix, row_meta, row_radii = convert_wide_annotations_to_binary_matrix(
            prot_ann_df,
            seq_len=seq_len,
            ann_id_columns=ann_id_columns,
            return_radii=True,
        )
    if label_matrix.shape[0] == 0:
        return None

    # None means expand every annotation row; a provided list restricts expansion
    # to annotation IDs with one of the requested prefixes. Only positive radii expand.
    candidate_rows = (
        range(label_matrix.shape[0])
        if expand_include is None
        else rows_to_expand(row_meta, expand_include)
    )
    radius_groups: dict[float, list[int]] = {}
    for i in candidate_rows:
        radius = row_radii.get(i)
        if radius is not None and pd.notna(radius) and radius > 0:
            radius_groups.setdefault(float(radius), []).append(i)
    if radius_groups:
        # Distance/PAE matrices don't depend on radius - build them once per protein and
        # threshold separately for each of this protein's distinct expansion radii, rather
        # than re-parsing the pdb/pae files (and rebuilding these O(n_residues^2) matrices)
        # once per radius group.
        distance_matrix, pae_matrix = get_protein_distance_pae_matrices(
            uniprot_id, str(reference_dir), pae_cutoff
        )
        if distance_matrix is None:
            warnings.warn(
                f"{uniprot_id}: no structure in reference dir, skipping expansion at radii "
                f"{sorted(radius_groups)}",
                UserWarning,
            )
        else:
            for radius, rows in radius_groups.items():
                adjacency_matrix = threshold_adjacency_matrix(
                    distance_matrix, pae_matrix, radius, pae_cutoff
                )
                label_matrix = expand_annotations_neighborhood(
                    label_matrix, adjacency_matrix, rows
                )

    if stratum_col is None:
        pvals_df = test_protein_annotations(
            uniprot_id, prot_rvas_df, label_matrix, row_meta, seq_len, n_perm, seed
        )
    else:
        pvals_df = test_protein_annotations_cmh(
            uniprot_id, prot_rvas_df, label_matrix, row_meta, n_perm, stratum_col, seed
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
    pae_cutoff: float = 0,
    reference_dir: str | Path | None = None,
    df_filter: pd.DataFrame | None = None,
    df_include: pd.DataFrame | None = None,
    min_alleles: int = 5,
    n_perm: int = 1000,
    seed: int = 0,
    n_workers: int = 1,
    results_dir: str | Path = "outputs/annotation_test",
    cleanup_per_protein: bool = False,
    format: str = "tsv.gz",
    stratum_col: str | None = None,
    result_fn: str = 'results'
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
          selected). ann_id_columns=["*"] uses every column except uniprot_id/aa_pos/
          seq_len/length/end. See convert_wide_annotations_to_binary_matrix.
    expand_include: restrict expansion to annotation ids starting with one of these
        (matched against row_meta, so it works for either ann_format), e.g.
        ["Binding site", "Active site"] - broad concepts (Domain, Chain, ...) are left as
        annotated. None (default) expands every annotation whose radius > 0.
    reference_dir: structure-informed-rvas reference dir (pdb_files/, pae_files/,
        pdb_pae_file_pos_guide.tsv, protein_sequence_guide.tsv). Used for protein length
        and, if expand_radius > 0, for the adjacency matrix.
    df_filter: variants/positions/proteins to exclude from rvas_df before testing, see
        filter_variants.
    df_include: variants/positions/proteins to restrict rvas_df to before testing (e.g. a
        gene panel) - applied together with df_filter, not instead of it.
    min_alleles: proteins need more than this many total case AND control alleles to be
        tested, see filter_proteins_by_allele_count.
    stratum_col: if set, CMH-adjust for this rvas_df column (e.g. ancestry/sex strata): null
        redrawn within strata, see test_protein_annotations_cmh. None (default) = unadjusted.
    n_workers: joblib worker processes for the per-protein loop. -1 (default) uses all
        cores; 1 runs in-process (no joblib), useful for debugging/profiling.
    cleanup_per_protein: delete the per-protein parquet shards once results.* is
        written, see summarize_annotation_test.
    format: "tsv.gz" (default) or "parquet" for the pooled results file. Per-protein
        shards are always parquet.

    Writes one `{uniprot_id}_annotation_pvals.parquet` per protein plus a pooled
    `results.tsv.gz` or `results.parquet` (global_fdr, global_fwer, bh_fdr) to results_dir,
    and returns the latter. Proteins with an existing shard in results_dir/per_protein (same
    n_perm) are reused, so a crashed run resumes; use a fresh results_dir after changing
    other parameters.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if reference_dir is None and "radius" in ann_df.columns and (ann_df["radius"].fillna(0) > 0).any():
        raise ValueError("ann_df has annotations with radius > 0, which requires reference_dir")

    rvas_df = filter_variants(rvas_df, df_include, mode="include")
    rvas_df = filter_variants(rvas_df, df_filter, mode="exclude")
    if stratum_col is not None:
        summarize_strata(rvas_df, stratum_col)
    tested_proteins = set(filter_proteins_by_allele_count(rvas_df, min_alleles))
    uniprot_ids = sorted(tested_proteins & set(ann_df["uniprot_id"].unique()))
    
    # Filter the dataframes to only include tested proteins.
    rvas_df = rvas_df[rvas_df["uniprot_id"].isin(uniprot_ids)]
    ann_df = ann_df[ann_df["uniprot_id"].isin(uniprot_ids)]

    protein_lengths = load_protein_lengths(reference_dir) if reference_dir else {}

    # Group once up front - filtering each df with a `== uniprot_id` mask per protein is
    # an O(n_proteins * len(df)) full-table scan and dominates runtime on large tables.
    rvas_by_protein = dict(tuple(rvas_df.groupby("uniprot_id", sort=False)))
    ann_by_protein = dict(tuple(ann_df.groupby("uniprot_id", sort=False)))

    shard_dir = results_dir / "per_protein"

    def _call(uniprot_id: str) -> pd.DataFrame | None:
        shard = shard_dir / f"{uniprot_id}_annotation_pvals.parquet"
        # ponytail: only n_perm is checked; a changed seed/radius/annotation set in the same results_dir reuses stale shards - use a fresh results_dir
        if shard.exists():
            cached = pd.read_parquet(shard)
            if sum(c.startswith("null_pval_") for c in cached.columns) == n_perm:
                return cached
        return process_protein(
            uniprot_id,
            rvas_by_protein[uniprot_id],
            ann_by_protein[uniprot_id],
            protein_lengths.get(uniprot_id),
            ann_format=ann_format,
            ann_include=ann_include,
            ann_id_columns=ann_id_columns,
            expand_include=expand_include,
            pae_cutoff=pae_cutoff,
            reference_dir=reference_dir,
            n_perm=n_perm,
            results_dir=results_dir,
            seed=seed,
            stratum_col=stratum_col,
        )

    n_resumed = sum(
        (shard_dir / f"{uid}_annotation_pvals.parquet").exists() for uid in uniprot_ids
    )
    logger.info(
        f"Running annotation test on {len(uniprot_ids)} proteins ({n_resumed} existing shards)"
    )
    if n_workers == 1:
        results = (_call(uid) for uid in tqdm(uniprot_ids))
    else:
        results = tqdm(
            Parallel(
                n_jobs=n_workers, backend="threading", return_as="generator_unordered"
            )(delayed(_call)(uid) for uid in uniprot_ids),
            total=len(uniprot_ids),
        )
    return summarize_annotation_test(
        (r for r in results if r is not None),
        n_perm,
        results_dir,
        reference_dir,
        cleanup_per_protein=cleanup_per_protein,
        format=format,
        result_fn=result_fn,
    )


def read_table(path: Path) -> pd.DataFrame:
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
        "columns to use as annotation ids directly (no concatenation), or '*' for every "
        "column except uniprot_id/aa_pos/seq_len/length/end.",
    )
    parser.add_argument(
        "--ann-include",
        default=None,
        help="Comma-separated filter on ann_id_columns[0] (long format only), e.g. 'Domain'",
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
        "--seed",
        type=int,
        default=0,
        help="Base seed for the permutations (combined with a per-protein hash)",
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
        help="Delete per-protein parquet shards once results.* is written",
    )
    parser.add_argument(
        "--format",
        default="tsv.gz",
        choices=["tsv.gz", "parquet"],
        help="Pooled results format (per-protein shards are always parquet)",
    )
    parser.add_argument(
        "--stratum-col",
        default=None,
        help="rvas column defining CMH strata; adjusts the test for stratum-dependent case fraction",
    )
    parser.add_argument(
        '--annotation-result-fn',
        type=str,
        default='results',
        help='''
        Filename for resulting annotation-test pooled results, without the filetype ending. (default results.tsv.gz).
        '''
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_annotation_test(
        read_table(args.rvas_path),
        read_table(args.ann_path),
        ann_format=args.ann_format,
        ann_include=args.ann_include.split(",") if args.ann_include else None,
        ann_id_columns=tuple(args.ann_id_columns.split(",")),
        expand_include=args.expand_include.split(",") if args.expand_include else None,
        pae_cutoff=args.pae_cutoff,
        reference_dir=args.reference_dir,
        df_filter=read_table(args.df_filter_path) if args.df_filter_path else None,
        df_include=read_table(args.df_include_path) if args.df_include_path else None,
        min_alleles=args.min_alleles,
        n_perm=args.n_perm,
        seed=args.seed,
        n_workers=args.n_workers,
        results_dir=args.results_dir,
        cleanup_per_protein=args.cleanup_per_protein,
        format=args.format,
        stratum_col=args.stratum_col,
        result_fn=args.annotation_result_fn
    )


if __name__ == "__main__":
    main()
