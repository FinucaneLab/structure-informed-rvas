"""Self-check for annotation_test_permutation: matrix builders, FDR streaming, CMH.

Run directly: python tests/test_annotation_test_permutation.py
Or under pytest: pytest tests/test_annotation_test_permutation.py

The checks are grouped into test functions rather than run at module scope. As a
flat script every assertion ran at import, which under pytest means collection
time: the file contributed no test items, and a single failed assertion aborted
the whole run as a collection error -- taking every other test file with it,
with no way to skip past it, since -k cannot deselect an import.

Shared inputs are built by plain _-prefixed helpers rather than pytest fixtures,
so each test is callable directly and the file still runs as a script.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotation_test_permutation import (
    NullPvalAccumulator,
    compute_global_fdr,
    compute_global_fwer,
    convert_long_annotations_to_binary_matrix,
    convert_wide_annotations_to_binary_matrix,
    expand_annotations_neighborhood,
    protein_seed,
    read_per_protein_shards,
    read_shard,
    shard_n_perm,
    run_annotation_test,
    summarize_annotation_test,
    wide_annotation_rows,
    write_shard,
)
# Imported under another name: pytest collects any module-level callable whose
# name begins with test_, and this is a production function, not a unit test.
# Collected, its first parameter is read as a fixture request and the run errors
# with "fixture 'uniprot_id' not found".
from annotation_test_permutation import (
    test_protein_annotations as run_protein_annotations,
    test_protein_annotations_cmh as run_protein_annotations_cmh,
    test_protein_annotations_mu as run_protein_annotations_mu,
)
from cmh import (
    cmh_expected_variance,
    mantel_haenszel_or,
    screen_and_exact_pvalues,
    summarize_strata,
)
from empirical_fdr import _compute_false_discoveries, _compute_fwer
from utils import fisher_greater_pvals, get_case_control_ac_matrix, get_stratified_case_ac_matrix


# ---------------------------------------------------------------------------
# Shared inputs. Each returns a fresh object, so no test can be affected by what
# another did to it, and the order they run in cannot matter.
# ---------------------------------------------------------------------------

def _interval_df():
    return pd.DataFrame(
        {
            "concept_type": ["Domain", "Binding site", "Binding site", "Disulfide bond"],
            "concept_value": ["Kinase", "ATP", "ATP", "Disulfide bond"],
            "start": [2, 5, 8, 3],
            "end": [4, 5, 8, 9],
            "geometry": ["range", "single", "single", "pair"],
        }
    )


def _wide_df():
    return pd.DataFrame(
        {
            "uniprot_id": ["P1"] * 4,
            "aa_pos": [1, 2, 3, 4],
            "Binding site": ["ATP", None, None, None],
            "cluster_layer9": [3, 3, 5, 5],
        }
    )


def _toy_adjacency():
    return np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )


def _protein_frames(n_perm=6):
    """Two per-protein p-value frames with null draws, seeded for reproducibility."""
    rng = np.random.default_rng(0)
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

    frames = [
        make_pvals_df("P1", [0.01, 0.2]),
        make_pvals_df("P2", [0.03, 0.5, 0.001]),
    ]
    return frames, n_perm, null_cols


def _demo_rvas():
    return pd.DataFrame(
        {"aa_pos": [1, 3, 5, 8], "ac_case": [3, 1, 2, 2], "ac_control": [1, 2, 1, 3]}
    )


def _confounded_rvas():
    """Two strata with opposite case fractions and opposite inside/outside coverage,
    but case fraction identical inside vs outside within each stratum -- pure
    confounding. Used both by the end-to-end adjustment test and by the CMH
    per-protein test, which re-reads it as a known-null input."""
    def toy_rows(stratum, inside, outside):
        return [
            dict(stratum=stratum, aa_pos=pos, ac_case=case, ac_control=ctrl)
            for pos, (case, ctrl) in ((1, inside), (10, outside))
        ]

    conf = pd.DataFrame(
        toy_rows("H", (72, 18), (8, 2)) + toy_rows("L", (2, 8), (18, 72))
    ).assign(uniprot_id="P1")
    conf_ann = pd.DataFrame(
        {"uniprot_id": ["P1"], "concept_type": "Site", "concept_value": "x",
         "start": 1, "end": 5}
    )
    return conf, conf_ann


# ---------------------------------------------------------------------------
# Building the binary annotation matrix
# ---------------------------------------------------------------------------

def test_interval_annotations_become_one_row_per_id():
    mat, meta = convert_long_annotations_to_binary_matrix(_interval_df(), seq_len=10)
    assert mat.shape == (3, 10)
    assert meta[0] == ["Domain_Kinase"]
    assert mat[0].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert meta[1] == ["Binding site_ATP"]
    assert mat[1].tolist() == [0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
    assert meta[2] == ["Disulfide bond_Disulfide bond"]
    assert mat[2].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 1, 0]


def test_ann_include_filters_concept_types():
    filtered_mat, _ = convert_long_annotations_to_binary_matrix(
        _interval_df(), seq_len=10, ann_include=["Domain"]
    )
    assert filtered_mat.shape == (1, 10)


def test_long_form_joins_several_id_columns():
    """Single-residue annotations (aa_pos instead of start/end), ann_id_columns
    joins several columns -- the cluster-table case: several rows per aa_pos,
    id = cluster_col + cluster_id."""
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


def test_identical_position_sets_merge_into_one_row():
    """Exact same position set under two different ids -> one tested row."""
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


def test_single_generic_column_uses_values_as_is():
    """No column-name prefix when there is only one annotation column."""
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


def test_several_columns_are_prefixed_and_nan_dropped():
    """Concept type / clustering layer per column: id is "{column}_{value}" to
    disambiguate, and NaN ("not annotated here") is dropped."""
    mat3, meta3 = convert_wide_annotations_to_binary_matrix(_wide_df(), seq_len=5)
    assert meta3[0] == ["Binding site_ATP"]
    assert mat3[0].tolist() == [1, 0, 0, 0, 0]
    ids3 = {ids[0] for ids in meta3.values()}
    assert ids3 == {"Binding site_ATP", "cluster_layer9_3", "cluster_layer9_5"}


def test_star_ann_id_columns_matches_the_default_and_excludes_length_columns():
    """ann_id_columns=["*"] matches the None default, and excludes
    seq_len/length/end alongside uniprot_id/aa_pos -- columns fallback_seq_len
    relies on being present but which are not annotations."""
    wide_df = _wide_df()
    mat3, meta3 = convert_wide_annotations_to_binary_matrix(wide_df, seq_len=5)

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


# ---------------------------------------------------------------------------
# Neighborhood expansion
# ---------------------------------------------------------------------------

def test_expansion_follows_the_adjacency():
    """Seed at position 3 (0-based idx 2); radius links idx 2 to idx 0 and 3,
    but not 4 (isolated)."""
    toy_labels = np.array([[0, 0, 1, 0, 0]], dtype=np.uint8)
    expanded = expand_annotations_neighborhood(toy_labels, _toy_adjacency())
    assert expanded.tolist() == [[1, 0, 1, 1, 0]]


def test_rows_to_expand_leaves_other_rows_alone():
    """Expand only the "Binding site" row, leave the broad "Domain" row alone."""
    toy_meta = {0: ["Domain_Kinase"], 1: ["Binding site_ATP"]}
    toy_mixed_labels = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.uint8)
    rows_to_expand_idx = [
        i for i, ids in toy_meta.items() if ids[0].startswith("Binding site_")
    ]
    mixed_expanded = expand_annotations_neighborhood(
        toy_mixed_labels, _toy_adjacency(), rows_to_expand_idx
    )
    assert mixed_expanded[0].tolist() == toy_mixed_labels[0].tolist()  # Domain untouched
    assert mixed_expanded[1].tolist() == [1, 0, 1, 1, 0]  # Binding site expanded


# ---------------------------------------------------------------------------
# Streamed corrections match the full-memory computation
# ---------------------------------------------------------------------------

def test_accumulator_matches_the_full_memory_computation():
    """NullPvalAccumulator + compute_global_fdr/compute_global_fwer must match the
    full-memory null_pvals_dict computation (empirical_fdr._compute_false_discoveries
    / _compute_fwer) -- same math, streamed one protein at a time instead of
    materialized all at once."""
    protein_frames, n_perm, null_cols = _protein_frames()

    accumulator = NullPvalAccumulator(n_perm)
    for frame in protein_frames:
        accumulator.add(frame)
    df_pvals, pooled_null, min_pvals_per_sim = accumulator.finalize()

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
        compute_global_fdr(df_pvals, pooled_null, n_perm), expected_false_discoveries
    )
    assert np.allclose(compute_global_fwer(df_pvals, min_pvals_per_sim), expected_fwer)


def test_summarize_agrees_across_sources_and_formats():
    """summarize_annotation_test must give the same result fed an in-memory list
    (the live per-protein loop) or read_per_protein_shards (rebuilding from disk
    without rerunning permutations) -- and format="tsv.gz"/"parquet" must
    round-trip to the same values."""
    protein_frames, n_perm, _ = _protein_frames()

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        per_protein_dir = results_dir / "per_protein"
        per_protein_dir.mkdir()
        for frame in protein_frames:
            uid = frame["uniprot_id"].iloc[0]
            frame.to_parquet(
                per_protein_dir / f"{uid}_annotation_pvals.parquet", index=False
            )

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


# ---------------------------------------------------------------------------
# Seeding and resume
# ---------------------------------------------------------------------------

def test_protein_seed_is_per_protein_and_per_permutation():
    demo = _demo_rvas()
    a = get_case_control_ac_matrix(demo, 10, 20, seed=protein_seed("P1"))
    b = get_case_control_ac_matrix(demo, 10, 20, seed=protein_seed("P1"))
    c = get_case_control_ac_matrix(demo, 10, 20, seed=protein_seed("P2"))
    assert (a[0] == b[0]).all() and not (a[0] == c[0]).all()
    d = get_case_control_ac_matrix(demo, 10, 20, seed=protein_seed("P1", 1))
    assert not (a[0] == d[0]).all()


def test_resume_reuses_shards_only_when_n_perm_is_unchanged():
    demo = _demo_rvas()
    rv = pd.concat([demo.assign(uniprot_id=u) for u in ("P1", "P2")])
    an = pd.DataFrame(
        {"uniprot_id": ["P1", "P2"], "concept_type": "Site", "concept_value": "x",
         "start": 1, "end": 5}
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        kw = dict(results_dir=tmpdir, cleanup_per_protein=False, min_alleles=0)
        run_annotation_test(rv, an, n_perm=5, **kw)

        # Mark P1's shard, so a reused shard is distinguishable from a fresh one.
        shard = Path(tmpdir) / "per_protein" / "P1_annotation_pvals.parquet"
        s = read_shard(shard)
        s["p_value"] = 0.123
        write_shard(s, shard)

        res = run_annotation_test(rv, an, n_perm=5, **kw)
        assert (res.loc[res.uniprot_id == "P1", "p_value"] == 0.123).all()

        res = run_annotation_test(rv, an, n_perm=6, **kw)
        assert not (res["p_value"] == 0.123).any()


# ---------------------------------------------------------------------------
# CMH
# ---------------------------------------------------------------------------

def test_one_stratum_cmh_equals_pooled_fisher():
    """With one stratum, the CMH p-value is the pooled Fisher exact p wherever
    the chi2 screen passes."""
    rng = np.random.default_rng(0)
    L = 30
    one = pd.DataFrame(
        {"aa_pos": rng.integers(1, L + 1, 60), "ac_case": rng.poisson(1.5, 60),
         "ac_control": rng.poisson(1.5, 60)}
    )
    one["stratum"] = "s"
    labels = np.zeros((4, L), dtype=bool)
    for i, (lo, hi) in enumerate([(0, 8), (5, 15), (10, 11), (20, 30)]):
        labels[i, lo:hi] = True
    case_ac, case_k, tot_k = get_stratified_case_ac_matrix(
        one, "stratum", L, 50, protein_seed("P1")
    )
    lab1 = csr_matrix(labels.astype(np.int8))
    inside_tot, inside_case = np.asarray(lab1 @ tot_k), np.asarray(lab1 @ case_k)
    e1, v1 = cmh_expected_variance(inside_tot, case_k.sum(0), tot_k.sum(0))
    cmh = {
        "s_case_inside": np.asarray(lab1 @ case_ac), "expected_sum": e1, "variance_sum": v1,
        "inside_totals_by_stratum": inside_tot, "case_totals": case_k.sum(0),
        "stratum_totals": tot_k.sum(0),
    }
    pv, stat = screen_and_exact_pvalues(cmh, 0.0, "two-sided")  # screen off: every informative row gets an exact p
    tot_case, tot_ctrl = int(case_k.sum()), int(tot_k.sum() - case_k.sum())
    a = cmh["s_case_inside"]
    b = inside_tot.sum(1)[:, None] - a
    n_compared = 0
    for r, c in [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 7)]:
        if stat[r, c] > 0:
            _, ref = fisher_exact([[a[r, c], tot_case - a[r, c]], [b[r, c], tot_ctrl - b[r, c]]])
            assert abs(pv[r, c] - ref) < 1e-9, (r, c, pv[r, c], ref)
            n_compared += 1
    # Every comparison above is guarded. If screen_and_exact_pvalues regressed so
    # that no cell were a candidate, pvals would stay at its np.ones_like default
    # and this -- the only check that CMH equals pooled Fisher -- would pass green
    # having compared nothing. Five of the six cells fire today.
    assert n_compared > 0, 'no cell passed the screen, so nothing was compared'


def test_stratification_removes_pure_confounding():
    conf, conf_ann = _confounded_rvas()
    with tempfile.TemporaryDirectory() as tmpdir:
        kw = dict(n_perm=200, min_alleles=0, cleanup_per_protein=False)
        unadj = run_annotation_test(conf, conf_ann, results_dir=Path(tmpdir) / "u", **kw)
        adj = run_annotation_test(
            conf, conf_ann, results_dir=Path(tmpdir) / "a", stratum_col="stratum", **kw
        )
    assert unadj["p_value"].iloc[0] < 1e-6
    assert adj["p_value"].iloc[0] > 0.1 and abs(adj["odds_ratio"].iloc[0] - 1) < 1e-9
    assert adj["n_informative_strata"].iloc[0] == 2 and adj["frac_ac_informative"].iloc[0] == 1.0


def test_mantel_haenszel_or():
    """Stratum 1 (a,b,c,d)=(3,1,1,3), stratum 2 (1,1,2,4)."""
    mh = mantel_haenszel_or(
        np.array([[3, 1]]), np.array([[4, 2]]), np.array([4, 3]), np.array([8, 8])
    )
    assert np.isclose(mh[0], (9 / 8 + 4 / 8) / (1 / 8 + 2 / 8))


def test_summarize_strata():
    strata = summarize_strata(
        pd.DataFrame({"stratum": ["a", "a", "b"], "ac_case": [1, 2, 0],
                      "ac_control": [1, 0, 3]}), "stratum"
    )
    assert strata.loc["a", "n_variants"] == 2 and strata.loc["b", "case_frac"] == 0


def test_cmh_annotation_test_counts_and_null_moments():
    """Observed counts are consistent; the null has the right moments."""
    rng = np.random.default_rng(1)
    L = 40
    multi = pd.DataFrame(
        {
            "aa_pos": rng.integers(1, L + 1, 120),
            "ac_case": rng.poisson(1.0, 120),
            "ac_control": rng.poisson(1.5, 120),
            "stratum": rng.choice(["a", "b", "c", "d"], 120, p=[0.5, 0.3, 0.19, 0.01]),
        }
    )
    multi.loc[multi["stratum"] == "d", ["ac_case", "ac_control"]] = [1, 0]  # single-allele stratum
    multi = pd.concat([multi, pd.DataFrame(
        {"aa_pos": [3], "ac_case": [0], "ac_control": [0], "stratum": ["e"]})])  # empty stratum
    labels = np.zeros((5, L), dtype=bool)
    for i, (lo, hi) in enumerate([(0, 8), (5, 15), (10, 11), (20, 40), (30, 33)]):
        labels[i, lo:hi] = True
    meta = {i: (f"a{i}",) for i in range(len(labels))}
    args = ("P1", multi, labels, meta, 4000, "stratum")
    res = run_protein_annotations_cmh(*args)
    obs_case = labels @ multi.groupby("aa_pos")["ac_case"].sum().reindex(range(1, L + 1), fill_value=0).to_numpy()
    obs_total = labels @ (multi.groupby("aa_pos")[["ac_case", "ac_control"]].sum().sum(axis=1)).reindex(range(1, L + 1), fill_value=0).to_numpy()
    assert (res["ac_in_case"] == obs_case).all() and (res["ac_in_case"] + res["ac_in_ctrl"] == obs_total).all()

    case_ac, case_k, tot_k = get_stratified_case_ac_matrix(
        multi, "stratum", L, 4000, protein_seed("P1")
    )
    lab = csr_matrix(labels.astype(np.int8))
    s_null = np.asarray(lab @ case_ac)[:, 1:]
    exp, var = cmh_expected_variance(np.asarray(lab @ tot_k), case_k.sum(0), tot_k.sum(0))
    informative = np.flatnonzero(var > 0)
    # Same guard as above: an all-zero variance would skip the loop entirely and
    # take both moment checks with it, silently.
    assert informative.size > 0, 'no annotation row had positive variance'
    for r in informative:
        assert abs(s_null[r].mean() - exp[r]) < 5 * np.sqrt(var[r] / s_null.shape[1]), r
        assert 0.85 < s_null[r].var() / var[r] < 1.15, r


def test_cmh_annotation_test_is_null_on_pure_confounding():
    conf, _ = _confounded_rvas()
    conf_res = run_protein_annotations_cmh(
        "P1", conf, np.isin(np.arange(1, 11), range(1, 6))[None, :], {0: ("x",)}, 200, "stratum"
    )
    assert conf_res["p_value"].iloc[0] > 0.1 and abs(conf_res["odds_ratio"].iloc[0] - 1) < 1e-9


def test_one_sided_annotation_test():
    # control-enriched set -> p ~ 1; case-enriched set -> p_greater <= p_two-sided
    oa = pd.DataFrame({"aa_pos": [1, 2, 3, 4, 5, 6, 7, 8], "ac_case": [8, 6, 1, 1, 0, 1, 1, 0],
                       "ac_control": [0, 1, 4, 4, 5, 1, 1, 4]})
    lab = np.zeros((2, 8), dtype=bool)
    lab[0, :2] = True  # case-enriched
    lab[1, 2:5] = True  # control-enriched
    meta = {0: ("case",), 1: ("ctrl",)}
    two = run_protein_annotations("P1", oa, lab, meta, 8, 50, alternative="two-sided")
    gr = run_protein_annotations("P1", oa, lab, meta, 8, 50)
    assert gr["p_value"][0] <= two["p_value"][0] and gr["p_value"][1] > 0.9


def test_fisher_greater_matches_scipy():
    for a, b in [(5, 1), (2, 3), (0, 4)]:
        ref = fisher_exact([[a, 10 - a], [b, 12 - b]], alternative="greater")[1]
        assert np.isclose(fisher_greater_pvals(a, b, 10, 12), ref)


def _mu_protein(n_res=200, n_ann=100, width=15, seed=0):
    """Random rate vector, random contiguous annotations; returns (uid-independent) pieces."""
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.01, 0.1, n_res)
    mu_lookup = {"P": (np.arange(1, n_res + 1), m, np.ones(n_res), np.ones(n_res))}
    labels = np.zeros((n_ann, n_res), dtype=int)
    for i, start in enumerate(rng.integers(0, n_res - width, n_ann)):
        labels[i, start:start + width] = 1
    row_meta = {i: ("A", str(i)) for i in range(n_ann)}
    return rng, m, mu_lookup, labels, row_meta


def _rvas_from_counts(x):
    pos = np.flatnonzero(x)
    return pd.DataFrame({"uniprot_id": "P", "aa_pos": pos + 1, "ac_case": x[pos]})


def _run_mu(x, mu_lookup, labels, row_meta, lambda_hat, n_perm=200, direction="enrichment"):
    return run_protein_annotations_mu(
        "P", _rvas_from_counts(x), labels, row_meta, len(x), n_perm, lambda_hat, mu_lookup,
        seed=1, direction=direction,
    )


def test_mu_null_is_calibrated():
    # one-sided discrete p-values are conservative, so allow only an upper tolerance
    frac = {"poisson": [], "binomial": []}
    for rep in range(10):
        rng, m, mu_lookup, labels, row_meta = _mu_protein(seed=rep)
        res = _run_mu(rng.poisson(5 * m), mu_lookup, labels, row_meta, 5.0, n_perm=50)
        for test, g in res.groupby("test"):
            frac[test].append((g["p_value"] < 0.05).mean())
    assert 0.01 < np.mean(frac["binomial"]) < 0.07, frac["binomial"]
    assert 0.01 < np.mean(frac["poisson"]) < 0.07, frac["poisson"]


def test_mu_power_ranks_injected_annotation_first():
    rng, m, mu_lookup, labels, row_meta = _mu_protein(seed=3)
    x = rng.poisson(5 * m)
    x[labels[0] == 1] += 3
    res = _run_mu(x, mu_lookup, labels, row_meta, 5.0)
    both = res.pivot(index="annotation_id", columns="test", values="local_fdr").max(axis=1)
    assert both.idxmin() == "A|0"
    assert res[(res.test == "binomial") & (res.annotation_id == "A|0")]["p_value"].iloc[0] < 1e-3


def test_mu_regional_constraint_passes_binomial_not_poisson():
    n_res, lam = 200, 4.0
    rng, m, mu_lookup, labels, row_meta = _mu_protein(n_res=n_res, n_ann=5, seed=4)
    m[:] = 0.05
    mu_lookup["P"] = (np.arange(1, n_res + 1), m, np.ones(n_res), np.ones(n_res))
    x = rng.poisson(lam * m)
    x[:n_res // 2] = 0  # first half constrained
    labels[0] = 0
    labels[0, n_res // 2:] = 1
    res = _run_mu(x, mu_lookup, labels, row_meta, lam).set_index(["annotation_id", "test"])
    assert res.loc[("A|0", "binomial"), "p_value"] < 1e-3
    assert res.loc[("A|0", "poisson"), "p_value"] > 0.05


def test_mu_summary_pivots_and_takes_max():
    rng, m, mu_lookup, labels, row_meta = _mu_protein(seed=5)
    res = _run_mu(rng.poisson(5 * m), mu_lookup, labels, row_meta, 5.0, n_perm=20)
    with tempfile.TemporaryDirectory() as d:
        out = summarize_annotation_test([res], 20, Path(d), format="parquet")
    assert len(out) == len(row_meta)
    assert {"p_poisson", "p_binomial", "global_fdr_poisson", "global_fdr_binomial"} <= set(out.columns)
    assert (out["global_fdr"] == out[["global_fdr_poisson", "global_fdr_binomial"]].max(axis=1)).all()


if __name__ == '__main__':
    n_run = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            print(f'  {name}')
            fn()
            n_run += 1
    # Without this, a rename or a move into a class leaves the scan matching
    # nothing and the run prints ok and exits 0 having tested nothing.
    assert n_run > 0, 'the test_* scan matched nothing'
    print(f'ok ({n_run} tests)')


# ---------------------------------------------------------------------------
# ann_df is sliced by row position, never copied
# ---------------------------------------------------------------------------

def test_wide_input_order_and_untested_proteins_do_not_matter():
    rv = pd.concat([_demo_rvas().assign(uniprot_id=u) for u in ("P1", "P2")])
    wide = pd.concat(
        [
            pd.DataFrame({"uniprot_id": u, "aa_pos": range(1, 9), "cl": [1, 1, 2, 2, 3, 3, 4, 4]})
            for u in ("P1", "P2", "P3")  # P3 has no rvas -> untested
        ]
    )
    interleaved = wide.iloc[np.argsort(np.tile(np.arange(8), 3), kind="stable")]  # rows of P1,P2,P3 alternate
    kw = dict(ann_format="wide", ann_id_columns=["*"], expand_include=[], n_perm=5, min_alleles=0, n_workers=1)
    out = []
    for ann in (wide, interleaved):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_annotation_test(rv, ann, results_dir=tmpdir, **kw)
        out.append(res.sort_values(["uniprot_id", "annotation_id"]).reset_index(drop=True))
    assert set(out[0].uniprot_id) == {"P1", "P2"}
    pd.testing.assert_frame_equal(out[0], out[1])


def test_compressed_null_matches_sorted_array_when_heavily_tied():
    n_perm = 20
    rng = np.random.default_rng(1)
    grid = np.array([0.0, 0.001, 0.01, 0.02, 0.05, 0.3])  # few distinct values -> many ties
    accumulator = NullPvalAccumulator(n_perm)
    accumulator.max_chunk_len = 5  # force several compactions
    raw = []
    for u in ("P1", "P2", "P3"):
        frame = pd.DataFrame({"uniprot_id": u, "annotation_id": ["a", "b", "c", "d"],
                              "p_value": rng.choice(grid, 4)})
        frame[[f"null_pval_{i}" for i in range(n_perm)]] = rng.choice(grid, (4, n_perm))
        accumulator.add(frame)
        raw.append(frame[[f"null_pval_{i}" for i in range(n_perm)]].to_numpy().ravel())
    df_pvals, pooled_null, _ = accumulator.finalize()
    sorted_null = np.sort(np.concatenate(raw))
    sorted_null = sorted_null[sorted_null < 0.05]
    p = df_pvals["p_value"].to_numpy()
    expected = np.where(p <= 0.05, np.searchsorted(sorted_null, p, side="right") / n_perm, float(len(p)))
    assert (compute_global_fdr(df_pvals, pooled_null, n_perm) == expected).all()


def test_sparse_and_dense_label_matrices_give_equal_frames():
    rng = np.random.default_rng(5)
    L = 40
    rv = pd.DataFrame({"uniprot_id": "P", "aa_pos": rng.integers(1, L + 1, 60), "ac_case": rng.poisson(1.0, 60) + 1,
                       "ac_control": rng.poisson(1.5, 60) + 1, "stratum": rng.choice(["a", "b"], 60)})
    ann = pd.DataFrame({"uniprot_id": "P", "aa_pos": range(1, L + 1), "cl": np.repeat(np.arange(8), 5)})
    sparse, meta, _ = wide_annotation_rows(ann, L, ["*"])
    dense = sparse.toarray()
    assert np.array_equal(dense, convert_wide_annotations_to_binary_matrix(ann, L, ["*"])[0])
    cc = [run_protein_annotations("P", rv, m, meta, L, 20) for m in (dense, sparse)]
    cmh = [run_protein_annotations_cmh("P", rv, m, meta, 20, "stratum") for m in (dense, sparse)]
    _, mu_m, mu_lookup, _, _ = _mu_protein(n_res=L, n_ann=1)
    x = rng.poisson(5 * mu_m)
    mu = [run_protein_annotations_mu("P", _rvas_from_counts(x), m, meta, L, 20, 5.0, mu_lookup, seed=1) for m in (dense, sparse)]
    for a, b in (cc, cmh, mu):
        pd.testing.assert_frame_equal(a, b)


def test_shard_round_trip_and_old_format_still_readable():
    frames, _, _ = _protein_frames()
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(frames):
            new, old = Path(tmpdir) / f"new{i}.parquet", Path(tmpdir) / f"old{i}.parquet"
            write_shard(frame, new)
            frame.to_parquet(old, index=False)  # pre-write_shard layout: one column per permutation
            pd.testing.assert_frame_equal(read_shard(new), frame)  # exact float round-trip
            pd.testing.assert_frame_equal(read_shard(old), frame)


def test_shard_roundtrip_edge_cases(tmp_path):
    frame = pd.DataFrame({"test": ["a", "b"], "null_pval_1": [0.2, 0.3], "null_pval_0": [0.1, 0.4]})
    write_shard(frame, tmp_path / "s.parquet")  # reordered null columns are written in index order
    out = read_shard(tmp_path / "s.parquet")
    assert shard_n_perm(tmp_path / "s.parquet") == 2
    assert out["null_pval_0"].tolist() == [0.1, 0.4]
    write_shard(frame[["test"]], tmp_path / "z.parquet")  # n_perm=0
    assert shard_n_perm(tmp_path / "z.parquet") == 0
    assert not list(tmp_path.glob("*.tmp"))
