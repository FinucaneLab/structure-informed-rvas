"""
Value tracking: how many variants, residues, neighborhoods and proteins are left
after each filtering step of a run.

Every run appends to <results_dir>/value_tracking.log (and echoes to the console).
Each count line looks like

    [counts] Remove common variants | vars=1234 res=1100 nbhds<=56789 prots=321

and carries the same numbers as a dict on the LogRecord (`record.counts`), so tests
and scripts can read them without parsing the text.

Fields:
    vars     rows of the variant table at that step
    res      distinct (uniprot_id, aa_pos) pairs carrying a variant
    nbhds    neighborhoods actually tested (exact; only after the scan/FDR)
    nbhds<=  total sequence length of the proteins still in play -- an upper bound on
             the neighborhoods that can be tested, since proteins can still be skipped
             or truncated to their structure later
    prots    proteins still in play
Anything else (n_denovo, sum_mu, round, ...) is mode-specific and named as such.
"Scanned" lines (and the conditional re-scan lines) also carry skipped=N with a
skipped_<reason>=N per reason, and centres_neutralized=N when conditioning set
centres to p = 1 (they stay in the FDR pool, but were not re-tested).

Until setup_value_tracking is called, log_counts writes nothing, so library callers
and tests that import the scan functions directly are unaffected.
"""

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

LOG_FILENAME = 'value_tracking.log'

logger = logging.getLogger('value_tracking')
logger.setLevel(logging.INFO)
# Own handlers only: not the per-module files logger_config sets up.
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

_seq_len = {}
_paused = 0


def setup_value_tracking(results_dir, reference_dir=None, run_info=None):
    """Point the tracker at <results_dir>/value_tracking.log and write a run header.

    Re-running setup (e.g. a second run in the same process) replaces the handlers
    rather than stacking them.
    """
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(results_dir, LOG_FILENAME), mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(ch)

    _seq_len.clear()
    if reference_dir:
        guide = os.path.join(reference_dir, 'protein_sequence_guide.tsv')
        if os.path.exists(guide):
            df = pd.read_csv(guide, sep='\t', usecols=['uniprot_id', 'sequence'])
            df = df.drop_duplicates('uniprot_id')
            _seq_len.update(zip(df.uniprot_id, df.sequence.str.len()))
        else:
            logger.warning(f'{guide} not found; nbhds<= upper bounds will be omitted')

    logger.info('=' * 60)
    header = f'[run] {datetime.now().isoformat(timespec="seconds")} {" ".join(sys.argv)}'
    if run_info:
        header += ' | ' + ' '.join(f'{k}={v}' for k, v in run_info.items())
    logger.info(header)


@contextmanager
def paused():
    """Suppress count lines inside the block -- e.g. the conditional re-scans, whose
    gene selection would otherwise re-log the whole input funnel each round."""
    global _paused
    _paused += 1
    try:
        yield
    finally:
        _paused -= 1


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.4g}'
    return str(v)


def log_counts(step, df=None, uniprot_ids=None, n_nbhds=None, nbhd_bound=True, **extra):
    """Log the counts left after `step`.

    df: the variant table after the step. None (e.g. --fdr-only with no RVAS data)
        just leaves the variant fields out.
    uniprot_ids: proteins still in play, when that is not simply df's proteins (a gene
        list chosen from df, or a step with no variant table at all).
    n_nbhds: exact number of neighborhoods/annotations tested. Without it, the
        sequence-length upper bound is logged instead, unless nbhd_bound=False.
    extra: mode-specific fields, logged as key=value after the standard ones.
    """
    if _paused:
        return
    counts = {}
    if df is not None:
        counts['vars'] = int(len(df))
        if {'uniprot_id', 'aa_pos'}.issubset(df.columns):
            counts['res'] = int(len(df[['uniprot_id', 'aa_pos']].drop_duplicates()))
        if uniprot_ids is None and 'uniprot_id' in df.columns:
            uniprot_ids = df['uniprot_id'].unique()
    if uniprot_ids is not None:
        uniprot_ids = set(uniprot_ids)
    if n_nbhds is not None:
        counts['nbhds'] = int(n_nbhds)
    elif nbhd_bound and uniprot_ids is not None and _seq_len:
        counts['nbhds_max'] = int(sum(_seq_len.get(u, 0) for u in uniprot_ids))
    if uniprot_ids is not None:
        counts['prots'] = len(uniprot_ids)
    counts.update(extra)
    if not counts:
        return
    body = ' '.join(
        f'{"nbhds<" if k == "nbhds_max" else k}={_fmt(v)}' for k, v in counts.items()
    )
    logger.info(f'[counts] {step} | {body}', extra={'counts': counts, 'step': step})


# ---------------------------------------------------------------------------
# per-gene bookkeeping the scans collect into a caller-owned `stats` dict
# ---------------------------------------------------------------------------

def record_skip(stats, reason, uniprot_id):
    """Note why a gene was not written. stats may be None (nothing collected)."""
    if stats is not None:
        stats.setdefault('skipped', {}).setdefault(reason, []).append(uniprot_id)


def record_neutralized(stats, uniprot_id, n_centres):
    """Note how many centres conditioning set to p = 1 in a written gene."""
    if stats is not None:
        stats.setdefault('neutralized', {})[uniprot_id] = int(n_centres)


def stats_fields(stats, genes=None):
    """key=value fields for a count line: skips by reason, and centres neutralized
    (restricted to `genes`, e.g. the ones actually written)."""
    if not stats:
        return {}
    out = {}
    skipped = stats.get('skipped', {})
    if skipped:
        out['skipped'] = sum(len(v) for v in skipped.values())
        for reason, uids in sorted(skipped.items()):
            out[f'skipped_{reason}'] = len(uids)
    neutralized = stats.get('neutralized', {})
    if neutralized:
        keep = neutralized if genes is None else {u: n for u, n in neutralized.items()
                                                  if u in set(genes)}
        out['centres_neutralized'] = sum(keep.values())
    return out


def skip_reason(exc):
    """Reason label for an exception that cost a gene its scan."""
    reason = getattr(exc, 'reason', None)
    if reason:
        return reason
    if isinstance(exc, FileNotFoundError):
        return 'missing_file'
    if isinstance(exc, KeyError):
        return 'missing_key'
    if isinstance(exc, MemoryError):
        return 'out_of_memory'
    if isinstance(exc, ValueError):
        return 'invalid_data'
    if isinstance(exc, ArithmeticError):
        return 'numerical_error'
    return 'error'


class GeneSkipped(ValueError):
    """A ValueError that says why the gene was skipped. Subclassing ValueError keeps
    every existing per-protein handler catching it unchanged."""

    def __init__(self, message, reason):
        super().__init__(message)
        self.reason = reason
