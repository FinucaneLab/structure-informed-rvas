"""Assert-based self-check for annotation_test_permutation's matrix builders.

Run directly: python tests/test_annotation_test_permutation.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotation_test_permutation import (
    NullPvalAccumulator,
    compute_global_fdr,
    compute_global_fwer,
    convert_long_annotations_to_binary_matrix,
    convert_wide_annotations_to_binary_matrix,
    expand_annotations_neighborhood,
    read_per_protein_shards,
    summarize_annotation_test,
)
from empirical_fdr import _compute_false_discoveries, _compute_fwer

interval_df = pd.DataFrame(
    {
        "concept_type": ["Domain", "Binding site", "Binding site", "Disulfide bond"],
        "concept_value": ["Kinase", "ATP", "ATP", "Disulfide bond"],
        "start": [2, 5, 8, 3],
        "end": [4, 5, 8, 9],
        "geometry": ["range", "single", "single", "pair"],
    }
)
mat, meta = convert_long_annotations_to_binary_matrix(interval_df, seq_len=10)
assert mat.shape == (3, 10)
assert meta[0] == ["Domain_Kinase"]
assert mat[0].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
assert meta[1] == ["Binding site_ATP"]
assert mat[1].tolist() == [0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
assert meta[2] == ["Disulfide bond_Disulfide bond"]
assert mat[2].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]

filtered_mat, _ = convert_long_annotations_to_binary_matrix(
    interval_df, seq_len=10, ann_include=["Domain"]
)
assert filtered_mat.shape == (1, 10)

# single-residue annotations (aa_pos instead of start/end), ann_id_columns joins several
# columns - the cluster-table case: several rows per aa_pos, id = cluster_col + cluster_id.
long_df = pd.DataFrame(
    {
        "uniprot_id": ["P1"] * 6,
        "aa_pos": [1, 2, 3, 1, 2, 3],
        "cluster_col": ["layer9"] * 3 + ["layer18"] * 3,
        "cluster_id": [1, 1, 2, 5, 5, 5],
    }
)
long_mat, long_meta = convert_long_annotations_to_binary_matrix(
    long_df, seq_len=5, ann_id_columns=("cluster_col", "cluster_id")
)
assert long_mat.shape == (3, 5)
ids_by_signature = {
    tuple(np.flatnonzero(row).tolist()): meta
    for row, meta in zip(long_mat, long_meta.values())
}
assert ids_by_signature[(0, 1)] == ["layer9_1"]
assert ids_by_signature[(2,)] == ["layer9_2"]
assert ids_by_signature[(0, 1, 2)] == ["layer18_5"]

# exact same position set under two different ids -> merged into one tested row.
dup_df = pd.DataFrame(
    {
        "uniprot_id": ["P1"] * 4,
        "aa_pos": [1, 2, 1, 2],
        "cluster_col": ["layer9", "layer9", "layer18", "layer18"],
        "cluster_id": [1, 1, 7, 7],
    }
)
dup_mat, dup_meta = convert_long_annotations_to_binary_matrix(
    dup_df, seq_len=5, ann_id_columns=("cluster_col", "cluster_id")
)
assert dup_mat.shape == (1, 5)
assert dup_meta[0] == ["layer9_1", "layer18_7"]

# single generic column: values are used as-is (no column-name prefix).
column_df = pd.DataFrame(
    {
        "uniprot_id": ["P1"] * 4,
        "aa_pos": [1, 2, 3, 4],
        "annotation": ["Binding site", "Binding site_ATP", "Domain_SET", "Domain_SET"],
    }
)
mat2, meta2 = convert_wide_annotations_to_binary_matrix(column_df, seq_len=5)
assert mat2.shape == (3, 5)
assert meta2[0] == ["Binding site"]
assert mat2[0].tolist() == [1, 0, 0, 0, 0]
assert meta2[2] == ["Domain_SET"]
assert mat2[2].tolist() == [0, 0, 1, 1, 0]

# multiple columns (e.g. concept type / clustering layer per column): id is
# "{column}_{value}" to disambiguate, and NaN ("not annotated here") is dropped.
wide_df = pd.DataFrame(
    {
        "uniprot_id": ["P1"] * 4,
        "aa_pos": [1, 2, 3, 4],
        "Binding site": ["ATP", None, None, None],
        "cluster_layer9": [3, 3, 5, 5],
    }
)
mat3, meta3 = convert_wide_annotations_to_binary_matrix(wide_df, seq_len=5)
assert meta3[0] == ["Binding site_ATP"]
assert mat3[0].tolist() == [1, 0, 0, 0, 0]
ids3 = {ids[0] for ids in meta3.values()}
assert ids3 == {"Binding site_ATP", "cluster_layer9_3", "cluster_layer9_5"}

# ann_id_columns=["*"] matches the None default, and excludes seq_len/length/end alongside
# uniprot_id/aa_pos (columns fallback_seq_len relies on being present but non-annotation).
mat3_star, meta3_star = convert_wide_annotations_to_binary_matrix(
    wide_df, seq_len=5, ann_id_columns=["*"]
)
assert mat3_star.tolist() == mat3.tolist()
assert meta3_star == meta3

wide_df_with_seq_len = wide_df.assign(seq_len=5, length=5, end=5)
mat3_star2, meta3_star2 = convert_wide_annotations_to_binary_matrix(
    wide_df_with_seq_len, seq_len=5, ann_id_columns=["*"]
)
assert mat3_star2.tolist() == mat3.tolist()
assert meta3_star2 == meta3

# expand_annotations_neighborhood: seed at position 3 (0-based idx 2), radius links idx
# 2 to idx 0 and 3 but not 4 (isolated).
toy_adjacency = np.array(
    [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
    ]
)
toy_labels = np.array([[0, 0, 1, 0, 0]], dtype=np.uint8)
expanded = expand_annotations_neighborhood(toy_labels, toy_adjacency)
assert expanded.tolist() == [[1, 0, 1, 1, 0]]

# rows_to_expand: expand only the "Binding site" row, leave the broad "Domain" row alone.
toy_meta = {0: ["Domain_Kinase"], 1: ["Binding site_ATP"]}
toy_mixed_labels = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.uint8)
rows_to_expand_idx = [
    i for i, ids in toy_meta.items() if ids[0].startswith("Binding site_")
]
mixed_expanded = expand_annotations_neighborhood(
    toy_mixed_labels, toy_adjacency, rows_to_expand_idx
)
assert mixed_expanded[0].tolist() == toy_mixed_labels[0].tolist()  # Domain untouched
assert mixed_expanded[1].tolist() == [1, 0, 1, 1, 0]  # Binding site expanded

# NullPvalAccumulator + compute_global_fdr/compute_global_fwer must match the full-memory
# null_pvals_dict computation (empirical_fdr._compute_false_discoveries/_compute_fwer) -
# same math, streamed one protein at a time instead of materialized all at once.
rng = np.random.default_rng(0)
n_perm = 6
null_cols = [f"null_pval_{i}" for i in range(n_perm)]


def make_pvals_df(uniprot_id, p_values):
    n = len(p_values)
    df = pd.DataFrame(
        {
            "uniprot_id": uniprot_id,
            "annotation_id": [f"ann_{i}" for i in range(n)],
            "p_value": p_values,
            "local_fwer": 0.0,
            "local_fdr": 0.0,
            "odds_ratio": 1.0,
            "ac_in_case": 1,
            "ac_in_ctrl": 1,
        }
    )
    null_pvals = rng.uniform(0, 1, size=(n, n_perm))
    df[null_cols] = null_pvals
    return df


protein_frames = [
    make_pvals_df("P1", [0.01, 0.2]),
    make_pvals_df("P2", [0.03, 0.5, 0.001]),
]

accumulator = NullPvalAccumulator(n_perm)
for frame in protein_frames:
    accumulator.add(frame)
df_pvals, pooled_null_pvals, min_pvals_per_sim = accumulator.finalize()

null_pvals_dict = {
    frame["uniprot_id"].iloc[0]: frame[null_cols].to_numpy(dtype=float)
    for frame in protein_frames
}
expected_input = pd.concat(
    [frame.drop(columns=null_cols) for frame in protein_frames], ignore_index=True
).sort_values("p_value", ascending=True, ignore_index=True)
assert df_pvals["p_value"].tolist() == expected_input["p_value"].tolist()

expected_false_discoveries = _compute_false_discoveries(
    df_pvals=expected_input,
    null_pvals_dict=null_pvals_dict,
    uniprot_ids=list(null_pvals_dict),
    n_sims=n_perm,
)
expected_fwer = _compute_fwer(
    df_pvals=expected_input,
    null_pvals_dict=null_pvals_dict,
    uniprot_ids=list(null_pvals_dict),
    n_sims=n_perm,
)
assert np.allclose(
    compute_global_fdr(df_pvals, pooled_null_pvals, n_perm), expected_false_discoveries
)
assert np.allclose(compute_global_fwer(df_pvals, min_pvals_per_sim), expected_fwer)

# summarize_annotation_test must give the same result fed an in-memory list (the live
# per-protein loop) or read_per_protein_shards (rebuilding from disk without rerunning
# permutations) - and format="tsv.gz"/"parquet" must round-trip to the same values.
with tempfile.TemporaryDirectory() as tmpdir:
    results_dir = Path(tmpdir)
    per_protein_dir = results_dir / "per_protein"
    per_protein_dir.mkdir()
    for frame in protein_frames:
        uid = frame["uniprot_id"].iloc[0]
        frame.to_parquet(per_protein_dir / f"{uid}_annotation_pvals.parquet", index=False)

    result_from_list = summarize_annotation_test(
        protein_frames, n_perm, results_dir, cleanup_per_protein=False, format="tsv.gz"
    )
    result_from_shards = summarize_annotation_test(
        read_per_protein_shards(results_dir),
        n_perm,
        results_dir,
        cleanup_per_protein=False,
        format="parquet",
    )
    pd.testing.assert_frame_equal(result_from_list, result_from_shards)

    reloaded_tsv = pd.read_csv(results_dir / "results.tsv.gz", sep="\t")
    reloaded_parquet = pd.read_parquet(results_dir / "results.parquet")
    assert list(reloaded_tsv.columns) == list(reloaded_parquet.columns)
    assert np.allclose(reloaded_tsv["p_value"], reloaded_parquet["p_value"])
    assert np.allclose(reloaded_tsv["global_fwer"], reloaded_parquet["global_fwer"])


# seeding + resume
from scan_test import get_case_control_ac_matrix
from annotation_test_permutation import protein_seed, run_annotation_test

demo = pd.DataFrame({"aa_pos": [1, 3, 5, 8], "ac_case": [3, 1, 2, 2], "ac_control": [1, 2, 1, 3]})
a = get_case_control_ac_matrix(demo, 10, 20, seed=protein_seed("P1"))
b = get_case_control_ac_matrix(demo, 10, 20, seed=protein_seed("P1"))
c = get_case_control_ac_matrix(demo, 10, 20, seed=protein_seed("P2"))
assert (a[0] == b[0]).all() and not (a[0] == c[0]).all()
d = get_case_control_ac_matrix(demo, 10, 20, seed=protein_seed("P1", 1))
assert not (a[0] == d[0]).all()

rv = pd.concat([demo.assign(uniprot_id=u) for u in ("P1", "P2")])
an = pd.DataFrame(
    {"uniprot_id": ["P1", "P2"], "concept_type": "Site", "concept_value": "x", "start": 1, "end": 5}
)
with tempfile.TemporaryDirectory() as d:
    kw = dict(results_dir=d, cleanup_per_protein=False, min_alleles=0)
    run_annotation_test(rv, an, n_perm=5, **kw)
    shard = Path(d) / "per_protein" / "P1_annotation_pvals.parquet"
    s = pd.read_parquet(shard)
    s["p_value"] = 0.123
    s.to_parquet(shard, index=False)
    res = run_annotation_test(rv, an, n_perm=5, **kw)
    assert (res.loc[res.uniprot_id == "P1", "p_value"] == 0.123).all()
    res = run_annotation_test(rv, an, n_perm=6, **kw)
    assert not (res["p_value"] == 0.123).any()

print("ok")
