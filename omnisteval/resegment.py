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
Main resegmentation pipeline for SoftSegmenter.

Applies the SoftSegmenter alignment algorithm from "Better Late Than Never:
Evaluation of Latency Metrics for Simultaneous Speech-to-Text Translation"
(https://arxiv.org/abs/2509.17349) to align hypothesis outputs with reference
segments, then evaluates with quality and latency metrics.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from .alignment import Word, align_words
from .data import Instance
from .tokenization import tokenize_words
from .scoring import evaluate_instances

logger = logging.getLogger(__name__)


def resegment(
    ref_words: List[List[Word]],
    hyp_words: List[List[Word]],
    segmentation: list,
    ref_sentences: List[str],
    *,
    char_level: bool = False,
    lang: Optional[str] = None,
    has_emission_timestamps: bool = False,
) -> Tuple[List[Instance], List[dict]]:
    """
    Align hypothesis words to reference segments and build resegmented instances.

    This is the core in-memory API: it takes already-loaded data (word lists,
    segmentation metadata, reference sentences), runs tokenization and alignment,
    and returns structured ``Instance`` objects ready for evaluation.

    Args:
        ref_words: Reference words grouped by recording/document.
        hyp_words: Hypothesis words grouped by recording/document.
        segmentation: Segmentation metadata (list of dicts with 'offset', 'duration').
        ref_sentences: Reference sentence strings (one per segment).
        char_level: Whether to use character-level alignment and scoring.
        lang: Language code for Moses tokenizer (None to skip tokenization).
        has_emission_timestamps: Whether hypothesis words carry emission timestamps.

    Returns:
        Tuple of (instances, instances_dicts) where instances is a list of
        ``Instance`` objects and instances_dicts is the serialisable list of dicts.
    """
    ref_words = tokenize_words(ref_words, lang)
    hyp_words = tokenize_words(hyp_words, lang)

    # Align
    new_segmentation = align_words(ref_words, hyp_words, char_level)

    # Build resegmented instances
    instances: List[Instance] = []
    instances_dict_list: List[dict] = []
    for idx, (seg, ref) in enumerate(zip(segmentation, ref_sentences)):
        new_seg = new_segmentation.get(idx, [])

        # Determine recording length
        recording_lengths = [w.recording_length for w in new_seg if w.recording_length is not None]
        if not recording_lengths:
            recording_length = seg["offset"] + seg["duration"]
        else:
            recording_length = max(recording_lengths)
            assert all(
                w.recording_length == recording_length for w in new_seg
            ), f"Recording lengths do not match for segment {idx}: {recording_lengths}"

        # Build prediction text
        prediction_parts = [w.original for w in new_seg if w.original is not None]
        prediction = "".join(prediction_parts) if char_level else " ".join(prediction_parts)

        seg_dict: dict = {
            "index": idx,
            "docid": seg.get("doc_id", 0),
            "segid": seg.get("seg_id", idx),
            "prediction": prediction,
            "reference": ref,
        }
        if seg["duration"] is not None and seg["duration"] > 0:
            seg_dict["source_length"] = seg["duration"]
        if has_emission_timestamps:
            seg_dict["emission_cu"] = [w.emission_cu - seg["offset"] for w in new_seg]
            seg_dict["emission_ca"] = [w.emission_ca - seg["offset"] for w in new_seg]
            seg_dict["time_to_recording_end"] = recording_length - seg["offset"]

        instances_dict_list.append(seg_dict)
        instances.append(
            Instance.from_dict(seg_dict, latency_unit="char" if char_level else "word")
        )

    return instances, instances_dict_list
