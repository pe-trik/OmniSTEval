# Copyright 2026 Peter Polák.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Core alignment algorithm for SoftSegmenter.

Implements a dynamic programming alignment algorithm (similar to Needleman-Wunsch / DTW)
to align hypothesis words to reference words based on Jaccard character-set similarity.

From "Better Late Than Never: Evaluation of Latency Metrics for Simultaneous
Speech-to-Text Translation" (https://arxiv.org/abs/2509.17349).
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from multiprocessing import Pool
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

INF = float("inf")
PUNCT = set([".", "!", "?", ",", ";", ":", "-", "(", ")"])
CHINESE_PUNCT = set(["。", "！", "？", "，", "；", "：", "—", "（", "）"])
JAPAN_PUNCT = set(["。", "！", "？", "，", "；", "：", "ー", "（", "）"])
ALL_PUNCT = PUNCT.union(CHINESE_PUNCT).union(JAPAN_PUNCT)


class AlignmentOperation(IntEnum):
    """Enum for alignment operations."""
    MATCH = 0
    DELETE = 1
    INSERT = 2
    NONE = 3


@dataclass
class Word:
    """
    Represents a word with associated emission timestamps.

    Attributes:
        text: The word text.
        emission_cu: Computation-unaware emission timestamp (ideal delay).
        seq_id: Sequence identifier for alignment.
        emission_ca: Computation-aware emission timestamp (elapsed).
        main: Whether this is a main word (not a subtoken).
        original: The original word before tokenization.
        recording_length: Total recording length.
    """

    text: str
    emission_cu: Optional[float] = None
    seq_id: Optional[int] = None
    emission_ca: Optional[float] = None
    main: bool = True
    original: Optional[str] = None
    recording_length: Optional[float] = None


def similarity(ref_word: Word, hyp_word: Word, char_level: bool) -> float:
    """
    Compute the similarity metric between two words based on Jaccard similarity of
    character sets (or exact match for character-level). If one word is punctuation
    and the other is not, return a negative score to discourage misalignment.

    Args:
        ref_word: Reference word.
        hyp_word: Hypothesis word.
        char_level: Whether to use character-level comparison.

    Returns:
        Similarity score between the words.
    """
    ref_text = ref_word.text
    hyp_text = hyp_word.text

    # Discourage aligning punctuation with non-punctuation
    ref_is_punct = ref_text in ALL_PUNCT
    hyp_is_punct = hyp_text in ALL_PUNCT
    if ref_is_punct ^ hyp_is_punct:
        return -INF

    # Character-level: exact match
    if char_level:
        return float(ref_text == hyp_text)

    # Word-level: Jaccard similarity over character sets
    ref_set = set(ref_text)
    hyp_set = set(hyp_text)
    inter = len(ref_set & hyp_set)
    union = len(ref_set) + len(hyp_set) - inter
    return (inter / union) if union else 0.0


def align_sequences(seq1: List[Word], seq2: List[Word], char_level: bool) -> tuple:
    """
    Align two sequences maximizing the similarity metric.

    Implements a dynamic programming algorithm similar to Needleman-Wunsch,
    but with a custom similarity score and no gap penalties. Insertions and
    deletions are allowed without penalty; matches are scored based on Jaccard
    similarity of character sets (word-level) or exact match (character-level).

    Args:
        seq1: First sequence (typically reference).
        seq2: Second sequence (typically hypothesis).
        char_level: Whether to use character-level comparison.

    Returns:
        Tuple of two aligned sequences with None for gaps.
    """
    n = len(seq1) + 1
    m = len(seq2) + 1
    dp = [[0.0] * m for _ in range(n)]
    dp_back = [[AlignmentOperation.NONE] * m for _ in range(n)]

    # Fill the first row and column of the matrix
    for i in range(n):
        dp[i][0] = 0
        dp_back[i][0] = AlignmentOperation.DELETE
    for j in range(m):
        dp[0][j] = 0
        dp_back[0][j] = AlignmentOperation.INSERT
    dp[0][0] = 0
    dp_back[0][0] = AlignmentOperation.MATCH

    # Fill the alignment matrix
    for i in range(1, n):
        for j in range(1, m):
            match = dp[i - 1][j - 1] + similarity(seq1[i - 1], seq2[j - 1], char_level)
            delete = dp[i - 1][j]
            insert = dp[i][j - 1]
            # Priority: MATCH > DELETE > INSERT when scores are tied
            if match >= delete and match >= insert:
                dp[i][j] = match
                dp_back[i][j] = AlignmentOperation.MATCH
            elif delete >= insert:
                dp[i][j] = delete
                dp_back[i][j] = AlignmentOperation.DELETE
            else:
                dp[i][j] = insert
                dp_back[i][j] = AlignmentOperation.INSERT

    # Backtrack to find the alignment
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        if dp_back[i][j] == AlignmentOperation.MATCH:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif dp_back[i][j] == AlignmentOperation.DELETE:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(None)
            i -= 1
        elif dp_back[i][j] == AlignmentOperation.INSERT:
            aligned_seq1.append(None)
            aligned_seq2.append(seq2[j - 1])
            j -= 1
        else:
            break
    aligned_seq1.reverse()
    aligned_seq2.reverse()
    return aligned_seq1, aligned_seq2


def process_alignment(
    ref_words: List[Optional[Word]],
    hyp_words: List[Optional[Word]],
    char_level: bool,
) -> List[Word]:
    """
    Process the alignment to assign sequence IDs to hypothesis words.

    Hypothesis words aligned to gaps (None in reference) are re-assigned to
    the nearest reference segment based on similarity scores.

    Args:
        ref_words: Aligned reference words (with None for gaps).
        hyp_words: Aligned hypothesis words (with None for gaps).
        char_level: Whether using character-level alignment.

    Returns:
        Processed hypothesis words with assigned sequence IDs.
    """
    assert len(ref_words) == len(hyp_words), \
        "Number of reference and hypothesis words do not match."

    def get_next_non_none_ref(idx):
        while idx < len(ref_words) and ref_words[idx] is None:
            idx += 1
        if idx == len(ref_words):
            return idx, None
        return idx, ref_words[idx]

    new_hyp_words = []
    last_ref = None
    nexti = 0
    for i, (ref, hyp) in enumerate(zip(ref_words, hyp_words)):
        if ref is None and i >= nexti:
            if hyp is not None:
                nexti, next_ref = get_next_non_none_ref(i)
                assert (
                    next_ref is not None or last_ref is not None
                ), "No reference word found."
                next_ref_score = (
                    similarity(hyp, next_ref, char_level) if next_ref is not None else -INF
                )
                prev_ref_score = (
                    similarity(hyp, last_ref, char_level) if last_ref is not None else -INF
                )
                if next_ref_score > prev_ref_score:
                    ref = next_ref
                else:
                    ref = last_ref
                    nexti = i
                logger.debug(
                    f"{hyp.text} aligned to {ref.text if ref else 'none'} with score "
                    f"{next_ref_score:.2f} vs {prev_ref_score:.2f} to "
                    f"{last_ref.text if last_ref else 'none'}"
                )
        # last_ref can be set to next_ref to avoid non-monotonicity
        if ref is not None and i >= nexti:
            last_ref = ref
        if hyp is not None:
            if ref is None:
                continue
            hyp.seq_id = ref.seq_id
            new_hyp_words.append(hyp)

    return new_hyp_words


def calculate_mwer(ref_words: list, hyp_words: list) -> float:
    """Calculate the Match Word Error Rate (MWER) between aligned reference and hypothesis."""
    assert len(ref_words) == len(hyp_words), \
        "Number of reference and hypothesis words do not match."
    matched = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref is not None and hyp is not None)
    return matched / len(ref_words) * 100.0


def _process_single_recording(args: tuple) -> list:
    """
    Process alignment for a single audio recording. Top-level function for multiprocessing.

    Args:
        args: Tuple of (recording_index, ref_words, hyp_words, char_level).

    Returns:
        Processed hypothesis words with assigned sequence IDs.
    """
    i, ref, hyp, char_level = args
    aligned_ref, aligned_hyp = align_sequences(ref, hyp, char_level)
    mwer = calculate_mwer(aligned_ref, aligned_hyp)
    logger.debug(f"Matched words for recording {i}: {mwer:.1f}%")
    return process_alignment(aligned_ref, aligned_hyp, char_level)


def align_words(
    ref_words: List[List[Word]],
    hyp_words: List[List[Word]],
    char_level: bool,
) -> Dict[int, List[Word]]:
    """
    Align all hypothesis words to reference words across all recordings.

    Runs alignment in parallel using multiprocessing and groups the resulting
    hypothesis words by their assigned reference sequence IDs.

    Args:
        ref_words: Reference words grouped by recording.
        hyp_words: Hypothesis words grouped by recording.
        char_level: Whether to use character-level alignment.

    Returns:
        Mapping from reference sequence ID to aligned hypothesis words.
    """
    assert len(ref_words) == len(hyp_words), \
        (f"Number of reference and hypothesis recordings do not match: "
         f"{len(ref_words)} vs {len(hyp_words)}")

    new_segmentation: Dict[int, List[Word]] = {}
    for inst_ref in ref_words:
        for ref in inst_ref:
            if ref.seq_id is not None:
                new_segmentation[ref.seq_id] = []

    args_list = [
        (i, ref, hyp, char_level)
        for i, (ref, hyp) in enumerate(zip(ref_words, hyp_words))
    ]
    with Pool() as pool:
        results = pool.map(_process_single_recording, args_list)

    assert len(results) == len(ref_words), \
        (f"Number of results and reference recordings do not match: "
         f"{len(results)} vs {len(ref_words)}")

    for result in results:
        for word in result:
            if word.main:
                new_segmentation[word.seq_id].append(word)

    return new_segmentation
