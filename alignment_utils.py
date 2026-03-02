"""
Alignment utilities for functional site annotation transfer.
Replaces deprecated Bio.pairwise2 with Bio.Align.PairwiseAligner.
"""

from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
import numpy as np


def get_pairwise_alignment(seq1, seq2):
    """
    Perform global pairwise alignment using Bio.Align.PairwiseAligner.

    Args:
        seq1: First sequence (query)
        seq2: Second sequence (target)

    Returns:
        Best alignment object from PairwiseAligner
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]

    return best_alignment


def alignment_to_strings(alignment):
    """
    Convert alignment object to aligned string format with gaps.

    Args:
        alignment: Bio.Align alignment object

    Returns:
        tuple: (aligned_seq1, aligned_seq2) - sequences with gaps represented as '-'
    """
    coords = alignment.coordinates
    seq1 = alignment.sequences[0]  # query
    seq2 = alignment.sequences[1]  # target

    aligned1 = []
    aligned2 = []

    # coords is a 2xN array where row 0 is positions in seq1, row 1 is positions in seq2
    for i in range(len(coords[0]) - 1):
        start1, end1 = coords[0][i], coords[0][i+1]
        start2, end2 = coords[1][i], coords[1][i+1]

        len1 = end1 - start1
        len2 = end2 - start2

        if len1 == 0:
            # Gap in sequence 1
            aligned1.extend(['-'] * len2)
            aligned2.extend(list(seq2[start2:end2]))
        elif len2 == 0:
            # Gap in sequence 2
            aligned1.extend(list(seq1[start1:end1]))
            aligned2.extend(['-'] * len1)
        else:
            # Both aligned
            aligned1.extend(list(seq1[start1:end1]))
            aligned2.extend(list(seq2[start2:end2]))

    return ''.join(aligned1), ''.join(aligned2)


def calculate_local_identity(alignment, position, window=5):
    """
    Calculate local sequence identity in a window around a position.

    Args:
        alignment: Bio.Align alignment object
        position: Query position (1-indexed)
        window: Window size on each side

    Returns:
        float: Local identity (0-1)
    """
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)

    # Find alignment position corresponding to query position
    align_pos = get_alignment_position(aligned_seq1, position)
    if align_pos is None:
        return 0.0

    # Get window boundaries in alignment coordinates
    start = max(0, align_pos - window)
    end = min(len(aligned_seq1), align_pos + window + 1)

    # Count matches in window
    matches = 0
    comparisons = 0

    for i in range(start, end):
        if aligned_seq1[i] != '-' and aligned_seq2[i] != '-':
            comparisons += 1
            if aligned_seq1[i] == aligned_seq2[i]:
                matches += 1

    if comparisons == 0:
        return 0.0

    return matches / comparisons


def calculate_gap_density(alignment, position, window=5):
    """
    Calculate gap density in a window around a position.

    Args:
        alignment: Bio.Align alignment object
        position: Query position (1-indexed)
        window: Window size on each side

    Returns:
        float: Gap density (0-1, where 0 = no gaps)
    """
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)

    # Find alignment position
    align_pos = get_alignment_position(aligned_seq1, position)
    if align_pos is None:
        return 1.0  # Treat as fully gapped if position not found

    # Get window boundaries
    start = max(0, align_pos - window)
    end = min(len(aligned_seq1), align_pos + window + 1)

    # Count gaps in window
    gaps = 0
    total_positions = end - start

    for i in range(start, end):
        if aligned_seq1[i] == '-' or aligned_seq2[i] == '-':
            gaps += 1

    if total_positions == 0:
        return 1.0

    return gaps / total_positions


def calculate_position_score(alignment, position):
    """
    Calculate position-specific alignment score contribution.

    Args:
        alignment: Bio.Align alignment object
        position: Query position (1-indexed)

    Returns:
        float: Normalized position score (0-1)
    """
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)

    # Find alignment position
    align_pos = get_alignment_position(aligned_seq1, position)
    if align_pos is None:
        return 0.0

    # Score this position
    if aligned_seq1[align_pos] == '-' or aligned_seq2[align_pos] == '-':
        # Gap penalty
        score = 0.0
    elif aligned_seq1[align_pos] == aligned_seq2[align_pos]:
        # Match
        score = 1.0
    else:
        # Mismatch
        score = 0.0

    return score


def get_alignment_position(aligned_seq, query_pos):
    """
    Convert query position to alignment position.

    Args:
        aligned_seq: Aligned sequence string (with gaps)
        query_pos: Position in original sequence (1-indexed)

    Returns:
        int: Position in aligned sequence (0-indexed), or None if not found
    """
    current_query_pos = 0

    for align_pos in range(len(aligned_seq)):
        if aligned_seq[align_pos] != '-':
            current_query_pos += 1

        if current_query_pos == query_pos:
            return align_pos

    return None


def calculate_confidence_score(alignment, position):
    """
    Calculate overall confidence score for annotation transfer at a position.

    Weighted average of:
    - Local sequence identity (40%)
    - Inverse gap density (30%)
    - Position-specific score (30%)

    Args:
        alignment: Bio.Align alignment object
        position: Query position (1-indexed)

    Returns:
        float: Confidence score (0-1)
    """
    local_identity = calculate_local_identity(alignment, position)
    gap_density = calculate_gap_density(alignment, position)
    position_score = calculate_position_score(alignment, position)

    confidence = (
        0.4 * local_identity +
        0.3 * (1 - gap_density) +
        0.3 * position_score
    )

    return confidence


def get_aligned_position(alignment, query_pos):
    """
    Get the target position aligned to a query position.

    Args:
        alignment: Bio.Align alignment object
        query_pos: Position in query sequence (1-indexed)

    Returns:
        int: Position in target sequence (1-indexed), or None if position is gapped
    """
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)

    # Find alignment position for query
    align_pos = get_alignment_position(aligned_seq1, query_pos)
    if align_pos is None:
        return None

    # Check if target has a gap at this position
    if aligned_seq2[align_pos] == '-':
        return None

    # Count non-gap positions in target up to alignment position
    target_pos = 0
    for i in range(align_pos + 1):
        if aligned_seq2[i] != '-':
            target_pos += 1

    return target_pos


def transfer_all_annotations(query_seq, target_seq, target_annotations):
    """
    Transfer all annotations from target to query for ALL positions.

    Args:
        query_seq: Query protein sequence
        target_seq: Target protein sequence
        target_annotations: Dict mapping target positions to list of annotations

    Returns:
        list: List of dicts with transferred annotations
    """
    alignment = get_pairwise_alignment(query_seq, target_seq)
    aligned_seq1, aligned_seq2 = alignment_to_strings(alignment)

    transferred = []

    # Iterate through all query positions
    for query_pos in range(1, len(query_seq) + 1):
        # Get corresponding target position
        target_pos = get_aligned_position(alignment, query_pos)

        if target_pos is None:
            continue

        # Get annotations at this target position
        target_pos_str = str(target_pos)
        if target_pos_str not in target_annotations:
            continue

        # Calculate confidence for this position
        confidence = calculate_confidence_score(alignment, query_pos)

        # Transfer each annotation at this position
        for annotation in target_annotations[target_pos_str]:
            transferred.append({
                'query_position': query_pos,
                'target_position': target_pos,
                'feature_type': annotation['feature_type'],
                'feature_description': annotation['description'],
                'alignment_confidence': confidence
            })

    return transferred
