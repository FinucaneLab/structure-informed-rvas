"""Annotation test: is a protein annotation enriched for case vs control rare-variant burden.

Per protein: build a binary (n_annotations, seq_len) matrix from the annotation table
(long or wide format, see the converters below), optionally expand local annotations
(binding site, active site, ...) to their 3D structural neighborhood, then test each
annotation against a per-residue permutation null that fixes the total (case+control)
allele count at each residue and redraws case/control labels (multivariate
hypergeometric). local_fdr and local_fwer come from that per-protein null; global_fdr/global_fwer
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
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, issparse
from scipy.stats import chi2, fisher_exact, norm
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from cmh import (
    cmh_expected_variance,
    mantel_haenszel_or,
    screen_and_exact_pvalues,
    summarize_strata,
)
from logger_config import get_logger
from mutation_rate_test import (
    _select_genes,
    binomial_pval_lookup,
    mu_vector_for_protein,
    poisson_pval_lookup,
    simulate_multinomial_null,
    simulate_poisson_null,
    sum_per_residue,
)
from utils import (
    DEFAULT_MIN_ALLELES,
    DEFAULT_SCREEN_CHI2,
    SCREEN_CHI2_GREATER,
    fisher_greater_pvals,
    get_case_control_ac_matrix,
    get_chi2_pvals_case_control,
    get_distance_and_pae_matrices,
    get_stratified_case_ac_matrix,
    threshold_adjacency_matrix,
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
    rvas_df: pd.DataFrame, min_alleles: int = DEFAULT_MIN_ALLELES
) -> list[str]:
    """uniprot_ids with at least min_alleles total case AND control alleles.

    Below that, the permutation null still runs, but there's not enough signal for the
    result to mean much - drop these proteins rather than let them pad the total test count.
    """
    totals = rvas_df.groupby("uniprot_id")[["ac_case", "ac_control"]].sum()
    return totals[
        (totals["ac_case"] >= min_alleles) & (totals["ac_control"] >= min_alleles)
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
    if "radius" not in ann_df.columns:
        ann_df = ann_df.assign(radius=np.nan)  # same fallback as the wide converter
    ann_df = validate_radius(ann_df)  # before the empty return, so bad radii always fail

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
    """Dense version of wide_annotation_rows: (label_binary_matrix, row_meta[, row_radii])."""
    matrix, row_meta, row_radii = wide_annotation_rows(ann_df, seq_len, ann_id_columns)
    result = (matrix.toarray(), row_meta, row_radii)
    return result if return_radii else result[:2]


def wide_annotation_rows(
    ann_df: pd.DataFrame,
    seq_len: int | None = None,
    ann_id_columns: list[str] | None = None,
) -> tuple[csr_matrix, dict, dict]:
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

    Returns (label_binary_matrix, row_meta, row_radii):
        label_binary_matrix: uint8 CSR, shape (n_annotations, seq_len), 1 at annotated positions.
        row_meta: dict mapping row index -> [id, ...] (annotations that share the exact
            same set of positions are merged into one row).
        row_radii: dict mapping matrix row indices to their radius (or None).
    """

    ann_df = validate_radius(ann_df)
    if ann_df.empty:
        return csr_matrix((0, seq_len or 0), dtype=np.uint8), {}, {}
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

    row_positions, row_meta, row_radii = [], {}, {}
    signature_to_row_idx = {}
    for col in ann_id_columns:
        not_na = ann_df[col].notna().to_numpy()
        col_values = ann_df[col].to_numpy()[not_na]
        col_positions = positions[not_na]
        if len(col_values) == 0:
            continue
        # Group by both annotation value and radius; this supports radius changes along
        # a wide-format annotation column without merging distinct expansion definitions.
        codes, uniques = pd.factorize(col_values)
        radius_codes, radius_uniques = pd.factorize(radii[not_na])  # NaN -> -1, its own group
        n_radius = len(radius_uniques) + 1
        key = codes.astype(np.int64) * n_radius + radius_codes + 1
        # sorted unique positions per group, vectorised over all groups of this column
        order = np.lexsort((col_positions, key))
        sorted_key, sorted_pos = key[order], col_positions[order]
        keep = np.ones(len(order), dtype=bool)
        keep[1:] = (sorted_key[1:] != sorted_key[:-1]) | (sorted_pos[1:] != sorted_pos[:-1])
        sorted_key, sorted_pos = sorted_key[keep], sorted_pos[keep]
        starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
        group_positions = np.split(sorted_pos, starts[1:])
        group_keys = sorted_key[starts]
        _, first_seen = np.unique(key, return_index=True)  # same (sorted-key) order as group_keys
        for g in np.argsort(first_seen):  # first-appearance order, like groupby(sort=False)
            group_code, radius_code = divmod(int(group_keys[g]), n_radius)
            radius = None if radius_code == 0 else float(radius_uniques[radius_code - 1])
            value = uniques[group_code]
            ann_id = str(value) if single_column else f"{col}_{value}"
            signature = group_positions[g]
            sig_key = (signature.tobytes(), radius)
            existing_idx = signature_to_row_idx.get(sig_key)
            if existing_idx is None:
                row_idx = len(row_positions)
                row_positions.append(signature[(signature >= 0) & (signature < seq_len)])
                row_meta[row_idx] = [ann_id]
                row_radii[row_idx] = radius
                signature_to_row_idx[sig_key] = row_idx
            else:
                row_meta[existing_idx].append(ann_id)

    indptr = np.cumsum([0] + [len(r) for r in row_positions])
    indices = np.concatenate(row_positions) if row_positions else np.zeros(0, dtype=np.int64)
    label_binary_matrix = csr_matrix(
        (np.ones(len(indices), dtype=np.uint8), indices, indptr), shape=(len(row_positions), seq_len)
    )
    return label_binary_matrix, row_meta, row_radii


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
    """Within-protein min-P FWER and permutation FDR (same estimator as the global one, with this
    protein's null): FDR(t) = E[#null p <= t] / #obs p <= t, capped at 1 and monotonised."""
    n_perm = null_pvals.shape[1]
    local_fwer = (null_pvals.min(axis=0)[None, :] <= observed_pvals[:, None]).mean(axis=1)
    false_discoveries = np.searchsorted(np.sort(null_pvals, axis=None), observed_pvals, side="right") / n_perm
    discoveries = np.searchsorted(np.sort(observed_pvals), observed_pvals, side="right")
    fdr = np.minimum(1.0, false_discoveries / discoveries)
    order = np.argsort(observed_pvals)
    local_fdr = np.empty_like(fdr)
    local_fdr[order] = np.minimum.accumulate(fdr[order][::-1])[::-1]
    return local_fwer, local_fdr


def test_protein_annotations(
    uniprot_id: str,
    prot_rvas_df: pd.DataFrame,
    label_matrix: np.ndarray,
    row_meta: dict,
    seq_len: int,
    n_perm: int,
    seed: int = 0,
    alternative: str = "greater",
) -> pd.DataFrame:
    """Permutation test for one protein's annotations.

    Null: for each residue, total (case+control) allele count is fixed and case/control
    labels are redrawn n_perm times (multivariate hypergeometric, see
    scan_test.get_case_control_ac_matrix). p-value per annotation is chi-square
    (Fisher's exact for borderline cases), against the observed count and against each
    permutation, giving local_fdr/local_fwer within this protein.
    alternative="greater" tests case enrichment inside the annotation only.
    """
    case_ac_matrix, ctrl_ac_matrix = get_case_control_ac_matrix(
        prot_rvas_df, n_res=seq_len, n_sim=n_perm, seed=protein_seed(uniprot_id, seed)
    )

    # only residues with alleles matter; cast after trimming so the int64 copy is only of those rows
    cols = np.flatnonzero(case_ac_matrix.sum(1) + ctrl_ac_matrix.sum(1))
    case_ac = case_ac_matrix[cols].astype(int)
    ctrl_ac = ctrl_ac_matrix[cols].astype(int)
    labels = csr_matrix(label_matrix[:, cols].astype(np.int8))  # sparse int8 avoids numpy's slow integer matmul
    ac_in_case = np.asarray(labels @ case_ac)  # (n_annotations, 1 + n_perm)
    ac_in_ctrl = np.asarray(labels @ ctrl_ac)

    total_ac_case = int(case_ac[:, 0].sum())
    total_ac_ctrl = int(ctrl_ac[:, 0].sum())

    pvals_table, potential_significant_table = get_chi2_pvals_case_control(
        ac_in_case, ac_in_ctrl, total_ac_case, total_ac_ctrl, alternative
    )
    or_table = np.ones_like(pvals_table)

    # Only run the exact test on (a, b) pairs that are both observed somewhere in this
    # protein's data (avoid computing it for every unobserved cell in the lookup table)
    # and flagged borderline by the chi-square pre-filter.
    observed_table = np.zeros_like(potential_significant_table, dtype=bool)
    a_obs, b_obs = ac_in_case.ravel(), ac_in_ctrl.ravel()
    valid_pairs = (a_obs + b_obs) > 0
    observed_table[a_obs[valid_pairs], b_obs[valid_pairs]] = True
    cells = np.argwhere(observed_table & potential_significant_table)
    if alternative == "greater":
        a, b = cells[:, 0], cells[:, 1]
        pvals_table[a, b] = fisher_greater_pvals(a, b, total_ac_case, total_ac_ctrl)
        with np.errstate(divide="ignore", invalid="ignore"):
            or_table[a, b] = a * (total_ac_ctrl - b) / ((total_ac_case - a) * b)
    else:
        for a, b in cells:
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


def test_protein_annotations_mu(
    uniprot_id: str,
    prot_rvas_df: pd.DataFrame,
    label_matrix: np.ndarray,
    row_meta: dict,
    seq_len: int,
    n_perm: int,
    lambda_hat: float,
    mu_lookup: dict,
    seed: int = 0,
    direction: str = "enrichment",
) -> pd.DataFrame:
    """De novo annotation test against a mutation-rate null (de novo analogue of test_protein_annotations).

    Two tests per annotation, as in mutation_rate_test: Test A (`poisson`) X ~ Poisson(lambda_hat * M)
    (absolute; guards against regional constraint) and Test B (`binomial`) X | N_g ~ Binomial(N_g, M / M_g)
    (within gene). M is the annotation's summed per-residue rate, M_g the whole sequence's. Null draws
    come from simulate_poisson_null / simulate_multinomial_null; the permutations give local
    FDR/FWER (annotations overlap, p-values are discrete). Returns both tests stacked, `test` column.

    Test A FDR/FWER are anti-conservative under --min-denovo gene selection (see
    simulate_poisson_null); read it as a direction-of-effect guard. Test B is unaffected.
    """
    prot_rvas_df = prot_rvas_df[prot_rvas_df["aa_pos"] <= seq_len]
    x_obs = sum_per_residue(prot_rvas_df, "ac_case", seq_len, dtype=int)
    m, mu_cov = mu_vector_for_protein(mu_lookup, uniprot_id, seq_len)
    n_g, m_g = int(x_obs.sum()), float(m.sum())
    rng = np.random.default_rng(protein_seed(uniprot_id, seed))
    x_a = np.hstack([x_obs, simulate_poisson_null(m, lambda_hat, n_perm, rng)])
    x_b = np.hstack([x_obs, simulate_multinomial_null(m, n_g, n_perm, rng)])

    labels = csr_matrix(label_matrix.astype(np.int8))
    m_ann = np.asarray(labels @ m).ravel()
    p_frac = m_ann / m_g
    rows = np.arange(labels.shape[0])[:, None]
    annotation_id = ["|".join(row_meta[i]) for i in range(len(row_meta))]

    frames = []
    for test, x, expected, make_lookup in (
        ("poisson", x_a, lambda_hat * m_ann,
         lambda x_max: poisson_pval_lookup(lambda_hat * m_ann, x_max, direction)),
        ("binomial", x_b, n_g * p_frac,
         lambda x_max: binomial_pval_lookup(p_frac, n_g, x_max, direction)),
    ):
        x_ann = np.asarray(labels @ x)  # (n_annotations, 1 + n_perm)
        pvals = make_lookup(int(x_ann.max()))[rows, x_ann]
        local_fwer, local_fdr = local_fwer_fdr(pvals[:, 0], pvals[:, 1:])
        pvals_df = pd.DataFrame(
            {
                "uniprot_id": uniprot_id,
                "annotation_id": annotation_id,
                "test": test,
                "p_value": pvals[:, 0],
                "local_fwer": local_fwer,
                "local_fdr": local_fdr,
                "n_case_in": x_ann[:, 0],
                "expected": expected,
                "obs_exp": x_ann[:, 0] / np.maximum(expected, 1e-300),
                "mu_in": m_ann,
                "n_denovo_gene": n_g,
                "mu_gene": m_g,
                "mu_coverage": mu_cov,
            }
        )
        null_df = pd.DataFrame(pvals[:, 1:], columns=[f"null_pval_{i}" for i in range(n_perm)])
        frames.append(pd.concat([pvals_df, null_df], axis=1))
    return pd.concat(frames, ignore_index=True)


def test_protein_annotations_cmh(
    uniprot_id: str,
    prot_rvas_df: pd.DataFrame,
    label_matrix: np.ndarray,
    row_meta: dict,
    n_perm: int,
    stratum_col: str,
    seed: int = 0,
    screen_chi2: float = DEFAULT_SCREEN_CHI2,
    alternative: str = "greater",
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

    pvals, stat = screen_and_exact_pvalues(cmh, screen_chi2, alternative)
    if alternative == "greater":
        z = (cmh["s_case_inside"] - expected[:, None]) / np.sqrt(np.where(variance > 0, variance, 1.0))[:, None]
        fallback = np.where(variance[:, None] > 0, norm.sf(z), 1.0)
        screened = (stat > (SCREEN_CHI2_GREATER if screen_chi2 == DEFAULT_SCREEN_CHI2 else screen_chi2)) & (z > 0)
    else:
        fallback = chi2.sf(stat, 1)
        screened = stat > screen_chi2
    pvals = np.where(screened, pvals, fallback)
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
    at once - only a running reduction (sub-threshold null p-values as (value, count)
    pairs, since they are heavily tied; per-sim min)."""

    max_chunk_len = 10_000_000  # total stored (value, count) pairs before compacting

    def __init__(self, n_perm: int, large_p_threshold: float = 0.05):
        self.n_perm = n_perm
        self.large_p_threshold = large_p_threshold
        self.null_cols = [f"null_pval_{i}" for i in range(n_perm)]
        self.obs_frames: list[pd.DataFrame] = []
        self.value_chunks: list[np.ndarray] = []
        self.count_chunks: list[np.ndarray] = []
        self.stored = 0
        self.min_pvals_per_sim = np.ones(n_perm)

    def add(self, pvals_df: pd.DataFrame) -> None:
        null_arr = pvals_df[self.null_cols].to_numpy(dtype=float)
        values, counts = np.unique(null_arr[null_arr < self.large_p_threshold], return_counts=True)
        self.value_chunks.append(values)
        self.count_chunks.append(counts)
        self.stored += len(values)
        if self.stored > self.max_chunk_len:
            self.compact()
        self.min_pvals_per_sim = np.minimum(self.min_pvals_per_sim, null_arr.min(axis=0))
        self.obs_frames.append(pvals_df.drop(columns=self.null_cols))

    def compact(self) -> None:
        """Merge all stored chunks into one sorted (values, counts) pair."""
        values = np.concatenate(self.value_chunks)
        counts = np.concatenate(self.count_chunks)
        values, inverse = np.unique(values, return_inverse=True)
        counts = np.bincount(inverse.ravel(), weights=counts).astype(np.int64)  # exact: counts < 2**53
        self.value_chunks, self.count_chunks, self.stored = [values], [counts], len(values)

    def finalize(self) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray], np.ndarray]:
        """(observed frame sorted by p_value, (values, cumcounts), per-sim min p).

        global_fdr (computed downstream) is a step-down estimate and only valid computed
        in ascending p_value order (rank i+1 divides the pooled-null count exceeded by
        observed p-values up to that rank). values are the distinct sub-threshold null
        p-values ascending; cumcounts[i] = number of pooled null p-values <= values[i].
        """
        df_pvals = pd.concat(self.obs_frames, ignore_index=True).sort_values(
            "p_value", ascending=True, ignore_index=True
        )
        self.compact()
        return df_pvals, (self.value_chunks[0], np.cumsum(self.count_chunks[0])), self.min_pvals_per_sim


def compute_global_fdr(
    df_pvals: pd.DataFrame,
    pooled_null: tuple[np.ndarray, np.ndarray],
    n_perm: int,
    large_p_threshold: float = 0.05,
) -> np.ndarray:
    """Same math as empirical_fdr._compute_false_discoveries, but takes the pooled
    sub-threshold null as (sorted distinct values, cumulative counts), see
    NullPvalAccumulator.finalize, instead of a per-protein null_pvals_dict."""
    values, cumcounts = pooled_null
    p_value = df_pvals["p_value"].to_numpy()
    mask = p_value <= large_p_threshold
    false_discoveries = np.full(len(df_pvals), float(len(df_pvals)))
    if mask.any():
        idx = np.searchsorted(values, p_value[mask], side="right")
        n_le = np.concatenate([[0], cumcounts])[idx]  # null p-values <= each observed p
        false_discoveries[mask] = n_le / n_perm
    return false_discoveries


def compute_global_fwer(df_pvals: pd.DataFrame, min_pvals_per_sim: np.ndarray) -> np.ndarray:
    """Same math as empirical_fdr._compute_fwer, but takes the already-pooled per-simulation
    minimum directly."""
    p_obs = df_pvals["p_value"].to_numpy().reshape(-1, 1)
    return np.mean(min_pvals_per_sim.reshape(1, -1) <= p_obs, axis=1)


def write_shard(df: pd.DataFrame, path: str | Path) -> None:
    """Write a per-protein frame, packing the null_pval_{i} columns into one fixed-size-list
    column (3-4x faster to write/read and half the size of 1000 separate columns)."""
    null_cols = [c for c in df.columns if c.startswith("null_pval_")]
    nulls = np.ascontiguousarray(df[null_cols].to_numpy(dtype=float))
    table = pa.Table.from_pandas(df.drop(columns=null_cols), preserve_index=False)
    packed = pa.FixedSizeListArray.from_arrays(pa.array(nulls.ravel()), len(null_cols))
    pq.write_table(table.append_column("null_pvals", packed), path)


def read_shard(path: str | Path) -> pd.DataFrame:
    """Read a shard from write_shard back into a frame with null_pval_{i} columns.
    Shards in the older one-column-per-permutation layout are returned unchanged."""
    table = pq.read_table(path)
    if "null_pvals" not in table.column_names:
        return table.to_pandas()
    packed = table["null_pvals"].combine_chunks()
    n_perm = packed.type.list_size
    nulls = packed.values.to_numpy().reshape(len(packed), n_perm)
    null_df = pd.DataFrame(nulls, columns=[f"null_pval_{i}" for i in range(n_perm)])
    return pd.concat([table.drop_columns("null_pvals").to_pandas(), null_df], axis=1)


def read_per_protein_shards(results_dir: str | Path) -> Iterator[pd.DataFrame]:
    """Lazily read back the per-protein parquet shards written by process_protein, one at a
    time - lets summarize_annotation_test rebuild results.* from a completed/crashed run
    without re-running any permutation tests."""
    per_protein_dir = Path(results_dir) / "per_protein"
    for shard in sorted(per_protein_dir.glob("*_annotation_pvals.parquet")):
        yield read_shard(shard)


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
    result_fn = normalize_result_fn(result_fn)
    accumulators: dict[str | None, NullPvalAccumulator] = {}
    for pvals_df in per_protein_results:
        if "test" in pvals_df.columns:  # mutation-rate mode: one frame holds both tests
            groups = pvals_df.groupby("test")
        else:
            groups = [(None, pvals_df)]
        for test, df in groups:
            accumulators.setdefault(test, NullPvalAccumulator(n_perm)).add(df)

    def pool(accumulator: NullPvalAccumulator) -> pd.DataFrame:
        df_pvals, pooled_null, min_pvals_per_sim = accumulator.finalize()
        false_discoveries = compute_global_fdr(df_pvals, pooled_null, n_perm)
        df_pvals["false_discoveries_avg"] = false_discoveries
        df_pvals["global_fdr"] = [x / (i + 1) for i, x in enumerate(false_discoveries)]
        df_pvals["global_fdr"] = df_pvals["global_fdr"][::-1].cummin()[::-1]
        df_pvals["global_fwer"] = compute_global_fwer(df_pvals, min_pvals_per_sim)
        df_pvals["bh_reject"], df_pvals["bh_fdr"] = fdrcorrection(
            df_pvals["p_value"].to_numpy(), alpha=0.05
        )
        return df_pvals

    if None in accumulators:
        df_pvals = pool(accumulators[None])
    else:
        # A unit passes only if it passes both tests, so combined corrections are the max.
        keys = ["uniprot_id", "annotation_id"]
        shared = ["n_case_in", "n_denovo_gene", "mu_gene", "mu_coverage", "mu_in"]
        per_test = {t: pool(a).drop(columns="test") for t, a in accumulators.items()}
        df_pvals = per_test["poisson"].merge(
            per_test["binomial"].drop(columns=shared),
            on=keys, suffixes=("_poisson", "_binomial"), validate="one_to_one",
        )
        df_pvals = df_pvals.rename(columns={"p_value_poisson": "p_poisson", "p_value_binomial": "p_binomial"})
        for col in ("global_fdr", "global_fwer", "local_fdr", "local_fwer", "bh_fdr"):
            df_pvals[col] = df_pvals[[f"{col}_poisson", f"{col}_binomial"]].max(axis=1)
        df_pvals = df_pvals.sort_values("global_fdr", ignore_index=True)

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


_RESULT_FN_SUFFIXES = (".tsv.gz", ".parquet", ".tsv")


def normalize_result_fn(result_fn: str) -> str:
    """Bare filename stem for the pooled results file.

    Strips a trailing .tsv.gz/.parquet/.tsv (the extension is added from --format) and
    rejects anything with a path component, so results never land outside results_dir.
    """
    if not result_fn or Path(result_fn).name != result_fn or result_fn in (".", ".."):
        raise ValueError(
            f"Annotation result filename must be a plain file name without a directory, got {result_fn!r}"
        )
    for suffix in _RESULT_FN_SUFFIXES:
        if result_fn.endswith(suffix) and len(result_fn) > len(suffix):
            return result_fn[: -len(suffix)]
    return result_fn


def add_result_fn_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--annotation-result-fn",
        type=str,
        default="results",
        help="File name (no directory) for the pooled annotation-test results; the "
        "extension comes from the format (default results.tsv.gz). A trailing "
        ".tsv.gz/.tsv/.parquet is ignored.",
    )


def validate_radius(ann_df: pd.DataFrame) -> pd.DataFrame:
    """ann_df with a numeric, non-negative ``radius`` column (NaN allowed = no expansion).

    Returns ann_df unchanged if it has no radius column. Raises ValueError for
    non-numeric (e.g. 'multiple-small') or negative radii.
    """
    if "radius" not in ann_df.columns:
        return ann_df
    radius = pd.to_numeric(ann_df["radius"], errors="coerce")
    bad = ann_df["radius"][radius.isna() & ann_df["radius"].notna()]
    if not bad.empty:
        raise ValueError(
            "ann_df 'radius' column must be numeric (Angstroms); got non-numeric values "
            f"such as {bad.unique()[:3].tolist()}"
        )
    if (radius < 0).any():
        raise ValueError("ann_df 'radius' column must be non-negative")
    return ann_df.assign(radius=radius)


def _expansion_requested(
    ann_df: pd.DataFrame,
    ann_format: str,
    ann_id_columns: tuple[str, ...] | list[str] | None,
    expand_include: list[str] | None,
) -> bool:
    """True if some annotation would be expanded: radius > 0 and (expand_include is None
    or the annotation id starts with one of its prefixes). Mirrors rows_to_expand, but
    on the raw table so it can run before any protein is processed."""
    if "radius" not in ann_df.columns:
        return False
    positive = ann_df[ann_df["radius"].fillna(0) > 0]
    if positive.empty:
        return False
    if expand_include is None:
        return True
    if not expand_include:
        return False
    prefixes = tuple(expand_include)
    if ann_format == "long":
        cols = list(ann_id_columns)
        ids = positive[cols].astype(str).agg("_".join, axis=1)
        return bool(ids.str.startswith(prefixes).any())
    if ann_id_columns is None or list(ann_id_columns) == ["*"]:
        skip = {"uniprot_id", "aa_pos", "seq_len", "length", "end", "radius"}
        cols = [c for c in positive.columns if c not in skip]
    else:
        cols = list(ann_id_columns)
    for col in cols:
        values = positive[col].dropna().astype(str).unique()
        ids = values if len(cols) == 1 else [f"{col}_{v}" for v in values]
        if any(i.startswith(prefixes) for i in ids):
            return True
    return False


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
    alternative: str = "greater",
    mu_lookup: dict | None = None,
    lambda_hat: float | None = None,
    direction: str = "enrichment",
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
        label_matrix, row_meta, row_radii = wide_annotation_rows(
            prot_ann_df, seq_len=seq_len, ann_id_columns=ann_id_columns
        )  # sparse; densified below only if some row is expanded
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
        if issparse(label_matrix):
            label_matrix = label_matrix.toarray()
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

    if mu_lookup is not None:
        pvals_df = test_protein_annotations_mu(
            uniprot_id, prot_rvas_df, label_matrix, row_meta, seq_len, n_perm,
            lambda_hat, mu_lookup, seed, direction,
        )
    elif stratum_col is None:
        pvals_df = test_protein_annotations(
            uniprot_id, prot_rvas_df, label_matrix, row_meta, seq_len, n_perm, seed, alternative
        )
    else:
        pvals_df = test_protein_annotations_cmh(
            uniprot_id, prot_rvas_df, label_matrix, row_meta, n_perm, stratum_col, seed, alternative=alternative
        )
    per_protein_dir = results_dir / "per_protein"
    per_protein_dir.mkdir(exist_ok=True)
    write_shard(pvals_df, per_protein_dir / f"{uniprot_id}_annotation_pvals.parquet")
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
    min_alleles: int = DEFAULT_MIN_ALLELES,
    n_perm: int = 1000,
    seed: int = 0,
    n_workers: int = 1,
    results_dir: str | Path = "outputs/annotation_test",
    cleanup_per_protein: bool = False,
    format: str = "tsv.gz",
    stratum_col: str | None = None,
    alternative: str = "greater",
    result_fn: str = 'results',
    mu_lookup: dict | None = None,
    lambda_hat: float | None = None,
    min_denovo: int = 5,
    direction: str = "enrichment",
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
        and, for annotations with radius > 0, for the adjacency matrix.
    df_filter: variants/positions/proteins to exclude from rvas_df before testing, see
        filter_variants.
    df_include: variants/positions/proteins to restrict rvas_df to before testing (e.g. a
        gene panel) - applied together with df_filter, not instead of it.
    min_alleles: proteins need at least this many total case AND control alleles to be
        tested, see filter_proteins_by_allele_count.
    stratum_col: if set, CMH-adjust for this rvas_df column (e.g. ancestry/sex strata): null
        redrawn within strata, see test_protein_annotations_cmh. None (default) = unadjusted.
    n_workers: joblib worker processes for the per-protein loop. -1 (default) uses all
        cores; 1 runs in-process (no joblib), useful for debugging/profiling.
    mu_lookup, lambda_hat, min_denovo, direction: de novo mode (rvas_df has ac_case only). Null is
        the mutation-rate model instead of case/control labels, see test_protein_annotations_mu;
        genes need >= min_denovo de novos and a nonzero rate; lambda_hat must come from
        estimate_lambda BEFORE gene selection. Use a fresh results_dir (shards of the two modes
        would collide). Incompatible with stratum_col.
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

    result_fn = normalize_result_fn(result_fn)
    ann_df = validate_radius(ann_df)
    if reference_dir is None and _expansion_requested(
        ann_df, ann_format, ann_id_columns, expand_include
    ):
        raise ValueError(
            "ann_df has annotations that would be expanded (radius > 0 and matching "
            "expand_include), which requires reference_dir"
        )

    rvas_df = filter_variants(rvas_df, df_include, mode="include")
    rvas_df = filter_variants(rvas_df, df_filter, mode="exclude")
    if mu_lookup is not None:
        if stratum_col is not None:
            raise ValueError("stratum_col is not supported with a mutation-rate null: all strata share one rate vector")
        tested_proteins = set(_select_genes(rvas_df, None, min_denovo, mu_lookup))
    else:
        if stratum_col is not None:
            summarize_strata(rvas_df, stratum_col)
        tested_proteins = set(filter_proteins_by_allele_count(rvas_df, min_alleles))
    uniprot_ids = sorted(tested_proteins & set(ann_df["uniprot_id"].unique()))
    
    rvas_df = rvas_df[rvas_df["uniprot_id"].isin(uniprot_ids)]

    protein_lengths = load_protein_lengths(reference_dir) if reference_dir else {}

    # Group once up front - filtering each df with a `== uniprot_id` mask per protein is
    # an O(n_proteins * len(df)) full-table scan and dominates runtime on large tables.
    # ann_df is never copied (a wide table can be >10 GB): only row positions are kept.
    rvas_by_protein = dict(tuple(rvas_df.groupby("uniprot_id", sort=False)))
    ann_rows = ann_df.groupby("uniprot_id", sort=False).indices

    shard_dir = results_dir / "per_protein"

    def _call(uniprot_id: str) -> pd.DataFrame | None:
        shard = shard_dir / f"{uniprot_id}_annotation_pvals.parquet"
        # ponytail: only n_perm is checked; a changed seed/radius/annotation set in the same results_dir reuses stale shards - use a fresh results_dir
        if shard.exists():
            cached = read_shard(shard)
            if sum(c.startswith("null_pval_") for c in cached.columns) == n_perm:
                return cached
        return process_protein(
            uniprot_id,
            rvas_by_protein[uniprot_id],
            ann_df.iloc[ann_rows[uniprot_id]],
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
            alternative=alternative,
            mu_lookup=mu_lookup,
            lambda_hat=lambda_hat,
            direction=direction,
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
        help="Comma-separated annotation id prefixes to expand, e.g. 'Binding site,Active site' (default: expand every annotation with radius > 0)",
    )
    parser.add_argument(
        "--expand-radius",
        type=float,
        default=None,
        help="Neighborhood radius (Angstroms) for expansion, used when the annotation table has no 'radius' column (default: no expansion)",
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
        default=DEFAULT_MIN_ALLELES,
        help="Proteins need at least this many total case AND control alleles",
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
        "--alternative",
        default="greater",
        choices=["two-sided", "greater"],
        help="greater = one-sided case enrichment inside the annotation",
    )
    add_result_fn_argument(parser)
    args = parser.parse_args()
    # At least 1, matching run.py. The comparison in
    # filter_proteins_by_allele_count is inclusive, so 0 is vacuously true and
    # admits proteins with no case alleles at all; each contributes a vacuous
    # p_value = 1.0 row that pads the global FDR and FWER pools.
    if args.min_alleles < 1:
        parser.error(f"--min-alleles must be at least 1, got {args.min_alleles}")
    return args


def main() -> None:
    args = _parse_args()
    ann_df = read_table(args.ann_path)
    if args.expand_radius is not None and "radius" not in ann_df.columns:
        ann_df["radius"] = args.expand_radius
    run_annotation_test(
        read_table(args.rvas_path),
        ann_df,
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
        alternative=args.alternative,
        result_fn=args.annotation_result_fn,
    )


if __name__ == "__main__":
    main()
