from __future__ import annotations

import hashlib
import os

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cmh import exact_cmh_minlike_lookup
from empirical_fdr import compute_fdr
from logger_config import get_logger
from utils import get_adjacency_matrix, write_dataset


logger = get_logger(__name__)

DEFAULT_SCREEN_CHI2 = 2.706


def _protein_seed(base_seed, uniprot_id):
    payload = f"{int(base_seed)}:{uniprot_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _per_residue_counts(df_s, n_res):
    case = np.zeros(n_res, dtype=np.int64)
    control = np.zeros(n_res, dtype=np.int64)

    if len(df_s) == 0:
        return case, control

    agg = (
        df_s.groupby("aa_pos", as_index=False)[["ac_case", "ac_control"]]
        .sum()
    )
    pos = agg["aa_pos"].to_numpy(dtype=np.int64) - 1
    if np.any(pos < 0) or np.any(pos >= n_res):
        bad = agg.loc[(pos < 0) | (pos >= n_res), "aa_pos"].tolist()
        raise ValueError(f"Amino-acid positions outside structure bounds: {bad[:10]}")

    case[pos] = agg["ac_case"].to_numpy(dtype=np.int64)
    control[pos] = agg["ac_control"].to_numpy(dtype=np.int64)
    return case, control


def _draw_stratified_null_cases(total_per_residue, n_case, n_sims, rng):
    total_per_residue = np.asarray(total_per_residue, dtype=np.int64)
    n_total = int(total_per_residue.sum())
    n_case = int(n_case)

    if n_case < 0 or n_case > n_total:
        raise ValueError("Invalid case total for multivariate hypergeometric draw")
    if n_sims <= 0:
        return np.empty((len(total_per_residue), 0), dtype=np.int32)
    if n_case == 0:
        return np.zeros((len(total_per_residue), n_sims), dtype=np.int32)
    if n_case == n_total:
        return np.repeat(
            total_per_residue[:, None].astype(np.int32),
            n_sims,
            axis=1,
        )

    draws = rng.multivariate_hypergeometric(
        total_per_residue,
        n_case,
        size=n_sims,
    )
    return draws.T.astype(np.int32, copy=False)


def _build_cmh_inputs(df, adjacency_matrix, n_sims, stratum_col, seed):
    if stratum_col not in df.columns:
        raise KeyError(f"Missing stratum column '{stratum_col}'")
    if df[stratum_col].isna().any():
        raise ValueError(f"Missing values found in stratum column '{stratum_col}'")

    n_res = adjacency_matrix.shape[0]
    adj = csr_matrix(adjacency_matrix.astype(np.int8, copy=False))

    work = (
        df.groupby([stratum_col, "aa_pos"], as_index=False)[["ac_case", "ac_control"]]
        .sum()
    )
    work[stratum_col] = work[stratum_col].astype(str)
    strata = sorted(work[stratum_col].unique().tolist())
    if not strata:
        raise ValueError("No strata found")
    grouped = {s: g for s, g in work.groupby(stratum_col, sort=False)}

    k = len(strata)
    s_case_inside = np.zeros((n_res, n_sims + 1), dtype=np.int32)
    expected_sum = np.zeros(n_res, dtype=np.float64)
    variance_sum = np.zeros(n_res, dtype=np.float64)

    inside_totals_by_stratum = np.zeros((n_res, k), dtype=np.int32)
    case_totals = np.zeros(k, dtype=np.int64)
    stratum_totals = np.zeros(k, dtype=np.int64)

    total_inside_all = np.zeros(n_res, dtype=np.int64)
    original_case_all = np.zeros(n_res, dtype=np.int64)
    original_control_all = np.zeros(n_res, dtype=np.int64)

    rng = np.random.default_rng(seed)

    for j, stratum in enumerate(strata):
        case, control = _per_residue_counts(grouped[stratum], n_res)
        total = case + control
        n_total = int(total.sum())
        n_case = int(case.sum())

        if n_total == 0:
            continue

        case_totals[j] = n_case
        stratum_totals[j] = n_total
        original_case_all += case
        original_control_all += control

        inside_total = np.asarray(adj @ total).reshape(-1).astype(np.int64)
        inside_case_obs = np.asarray(adj @ case).reshape(-1).astype(np.int64)

        inside_totals_by_stratum[:, j] = inside_total.astype(np.int32)
        total_inside_all += inside_total
        s_case_inside[:, 0] += inside_case_obs.astype(np.int32)

        expected_sum += inside_total * (n_case / n_total)
        if n_total > 1:
            variance_sum += (
                inside_total
                * (n_total - inside_total)
                * n_case
                * (n_total - n_case)
                / (float(n_total) ** 2 * (n_total - 1.0))
            )

        null_case_per_residue = _draw_stratified_null_cases(
            total,
            n_case,
            n_sims,
            rng,
        )
        if n_sims > 0:
            inside_case_null = adj @ null_case_per_residue
            s_case_inside[:, 1:] += np.asarray(
                inside_case_null,
                dtype=np.int32,
            )

    observed_nbhd_case = s_case_inside[:, 0].astype(np.int64)
    observed_nbhd_control = total_inside_all - observed_nbhd_case

    return {
        "strata": strata,
        "s_case_inside": s_case_inside,
        "expected_sum": expected_sum,
        "variance_sum": variance_sum,
        "inside_totals_by_stratum": inside_totals_by_stratum,
        "case_totals": case_totals,
        "stratum_totals": stratum_totals,
        "observed_nbhd_case": observed_nbhd_case,
        "observed_nbhd_control": observed_nbhd_control,
        "original_case": original_case_all,
        "original_control": original_control_all,
    }


def _screen_and_exact_pvalues(cmh, screen_chi2=DEFAULT_SCREEN_CHI2):
    s = cmh["s_case_inside"]
    e = cmh["expected_sum"]
    v = cmh["variance_sum"]

    stat = np.zeros_like(s, dtype=np.float64)
    informative = v > 0
    stat[informative, :] = (
        (s[informative, :] - e[informative, None]) ** 2
        / v[informative, None]
    )

    candidate = (stat > float(screen_chi2)) & informative[:, None]
    pvals = np.ones_like(stat, dtype=np.float64)

    inside_by_k = cmh["inside_totals_by_stratum"]
    case_totals = cmh["case_totals"]
    totals = cmh["stratum_totals"]

    cache = {}
    rows = np.where(candidate.any(axis=1))[0]

    for r in rows:
        cols = np.where(candidate[r])[0]
        inside = inside_by_k[r, :]
        key = inside.tobytes()

        if key not in cache:
            support, p_lookup = exact_cmh_minlike_lookup(
                inside,
                case_totals,
                totals,
            )
            cache[key] = (int(support[0]), int(support[-1]), p_lookup)

        lo, hi, p_lookup = cache[key]
        sval = s[r, cols].astype(np.int64)
        if np.any(sval < lo) or np.any(sval > hi):
            raise RuntimeError(
                f"CMH sufficient statistic outside exact support at residue {r + 1}"
            )
        pvals[r, cols] = p_lookup[sval - lo]

    return pvals, stat


def compute_all_cmh_pvals(
    df,
    pdb_file_pos_guide,
    pdb_dir,
    pae_dir,
    uniprot_id,
    n_sims,
    stratum_col,
    radius=15.0,
    pae_cutoff=15.0,
    screen_chi2=DEFAULT_SCREEN_CHI2,
    seed=1,
):
    if isinstance(radius, str):
        raise NotImplementedError(
            "CMH currently supports one numeric neighborhood radius. "
            "Use --neighborhood-radius 15 for the default analysis."
        )

    adjacency_matrix = get_adjacency_matrix(
        pdb_file_pos_guide,
        pdb_dir,
        pae_dir,
        uniprot_id,
        radius,
        pae_cutoff,
    )
    if adjacency_matrix is None:
        raise FileNotFoundError(f"No usable adjacency matrix for {uniprot_id}")

    cmh = _build_cmh_inputs(
        df=df,
        adjacency_matrix=adjacency_matrix,
        n_sims=n_sims,
        stratum_col=stratum_col,
        seed=_protein_seed(seed, uniprot_id),
    )
    pval_matrix, cmh_stat_matrix = _screen_and_exact_pvalues(
        cmh,
        screen_chi2=screen_chi2,
    )

    pval_columns = ["p_value"] + [f"null_pval_{i}" for i in range(n_sims)]
    df_pvals = pd.DataFrame(pval_matrix, columns=pval_columns)
    df_pvals["nbhd_case"] = cmh["observed_nbhd_case"]
    df_pvals["nbhd_control"] = cmh["observed_nbhd_control"]
    df_pvals["radius"] = float(radius)
    df_pvals["original_case"] = cmh["original_case"]
    df_pvals["original_control"] = cmh["original_control"]
    df_pvals["cmh_chisq"] = cmh_stat_matrix[:, 0]
    df_pvals["cmh_variance"] = cmh["variance_sum"]

    columns = [
        "nbhd_case",
        "nbhd_control",
        "radius",
        "original_case",
        "original_control",
        "cmh_chisq",
        "cmh_variance",
    ] + pval_columns
    return df_pvals[columns], adjacency_matrix


def write_cmh_pvals(results_dir, uniprot_id, df_pvals, pval_file):
    path = os.path.join(results_dir, pval_file)
    with h5py.File(path, "a") as fid:
        null_cols = [c for c in df_pvals.columns if c.startswith("null_pval_")]
        write_dataset(fid, uniprot_id, df_pvals[["p_value"]])
        write_dataset(fid, f"{uniprot_id}_null_pval", df_pvals[null_cols])
        write_dataset(
            fid,
            f"{uniprot_id}_nbhd",
            df_pvals[["nbhd_case", "nbhd_control"]],
        )
        write_dataset(fid, f"{uniprot_id}_radius", df_pvals[["radius"]])
        write_dataset(
            fid,
            f"{uniprot_id}_original",
            df_pvals[["original_case", "original_control"]],
        )
        write_dataset(fid, f"{uniprot_id}_cmh_chisq", df_pvals[["cmh_chisq"]])
        write_dataset(fid, f"{uniprot_id}_cmh_variance", df_pvals[["cmh_variance"]])


def _filter_proteins(df_rvas, df_fdr_filter=None, min_alleles=5):
    grouped = df_rvas.groupby("uniprot_id")[["ac_case", "ac_control"]].sum()
    keep = grouped[
        (grouped["ac_case"] > min_alleles)
        & (grouped["ac_control"] > min_alleles)
    ].index.to_numpy()

    if df_fdr_filter is not None:
        keep = np.intersect1d(
            keep,
            df_fdr_filter["uniprot_id"].unique(),
        )
    return keep.tolist()


def _remove_neighborhood(df, adjacency_matrix, positions, uniprot_id):
    out = df.copy()
    for center in map(int, positions.split(",")):
        if center < 1 or center > adjacency_matrix.shape[0]:
            raise ValueError(
                f"Neighborhood center {center} is outside {uniprot_id} structure bounds"
            )
        logger.info(f"Removing neighborhood of position {center} for {uniprot_id}")
        nbhd = set(np.where(adjacency_matrix[center - 1] == 1)[0] + 1)
        out = out[~out["aa_pos"].isin(nbhd)].copy()
    return out.reset_index(drop=True)


def scan_test_cmh(
    df_rvas,
    reference_dir,
    radius,
    pae_cutoff,
    results_dir,
    n_sims,
    no_fdr,
    fdr_only,
    fdr_cutoff,
    df_fdr_filter,
    ignore_ac,
    fdr_file,
    pval_file,
    remove_nbhd,
    stratum_col,
    screen_chi2=DEFAULT_SCREEN_CHI2,
    seed=1,
):
    if fdr_only:
        df_results = compute_fdr(
            results_dir,
            fdr_cutoff,
            df_fdr_filter,
            reference_dir,
            pval_file,
        )
        df_results.to_csv(
            os.path.join(results_dir, fdr_file),
            sep="\t",
            index=False,
        )
        return

    if ignore_ac:
        raise ValueError(
            "--ignore-ac is not supported with --test-method cmh because a variant "
            "can occur in multiple strata and therefore has no unique stratum after "
            "collapsing to one presence/absence observation."
        )
    if isinstance(radius, str):
        raise ValueError(
            "--test-method cmh currently requires one numeric --neighborhood-radius"
        )
    if df_rvas is None:
        raise ValueError("CMH scan requires mapped RVAS data")

    required = {"uniprot_id", "aa_pos", "ac_case", "ac_control", stratum_col}
    missing = required.difference(df_rvas.columns)
    if missing:
        raise KeyError(f"Missing required CMH columns: {sorted(missing)}")

    os.makedirs(results_dir, exist_ok=True)
    pdb_file_pos_guide = os.path.join(reference_dir, "pdb_pae_file_pos_guide.tsv")
    pdb_dir = os.path.join(reference_dir, "pdb_files")
    pae_dir = os.path.join(reference_dir, "pae_files")

    uniprot_ids = _filter_proteins(df_rvas, df_fdr_filter)
    logger.info(
        f"Selected {len(uniprot_ids)} proteins for stratified CMH analysis"
    )

    for i, uniprot_id in enumerate(uniprot_ids, start=1):
        logger.info(
            f"CMH processing {uniprot_id} (protein {i} out of {len(uniprot_ids)})"
        )
        try:
            df = df_rvas[df_rvas["uniprot_id"] == uniprot_id].copy()

            if remove_nbhd is not None:
                adjacency_matrix = get_adjacency_matrix(
                    pdb_file_pos_guide,
                    pdb_dir,
                    pae_dir,
                    uniprot_id,
                    radius,
                    pae_cutoff,
                )
                if adjacency_matrix is None:
                    raise FileNotFoundError(
                        f"No usable adjacency matrix for {uniprot_id}"
                    )
                df = _remove_neighborhood(
                    df,
                    adjacency_matrix,
                    remove_nbhd,
                    uniprot_id,
                )

            if (df["ac_case"].sum() < 5) or (df["ac_control"].sum() < 5):
                logger.warning(
                    f"{uniprot_id}: There must be at least 5 case and 5 control alleles. Skipping."
                )
                continue

            df_pvals, _ = compute_all_cmh_pvals(
                df=df,
                pdb_file_pos_guide=pdb_file_pos_guide,
                pdb_dir=pdb_dir,
                pae_dir=pae_dir,
                uniprot_id=uniprot_id,
                n_sims=n_sims,
                stratum_col=stratum_col,
                radius=radius,
                pae_cutoff=pae_cutoff,
                screen_chi2=screen_chi2,
                seed=seed,
            )
            write_cmh_pvals(
                results_dir,
                uniprot_id,
                df_pvals,
                pval_file,
            )

        except FileNotFoundError as exc:
            logger.error(f"{uniprot_id}: Required file not found - {exc}")
        except (KeyError, ValueError, FloatingPointError) as exc:
            logger.error(f"{uniprot_id}: Invalid CMH input - {exc}")
        except MemoryError as exc:
            logger.error(f"{uniprot_id}: Insufficient memory - {exc}")
        except Exception as exc:
            logger.exception(f"{uniprot_id}: Unexpected CMH error - {exc}")

    if not no_fdr:
        fdr_filter = df_fdr_filter
        if fdr_filter is None:
            fdr_filter = pd.DataFrame({"uniprot_id": uniprot_ids})
        df_results = compute_fdr(
            results_dir,
            fdr_cutoff,
            fdr_filter,
            reference_dir,
            pval_file,
        )
        df_results.to_csv(
            os.path.join(results_dir, fdr_file),
            sep="\t",
            index=False,
        )
