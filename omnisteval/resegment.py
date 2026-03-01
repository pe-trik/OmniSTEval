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


def build_resegmented_instances(
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


def resegment(
    ref_words: List[List[Word]],
    hyp_words: List[List[Word]],
    segmentation: list,
    ref_sentences: List[str],
    output_folder: Optional[str] = None,
    *,
    char_level: bool = False,
    lang: Optional[str] = None,
    bleu_tokenizer: str = "13a",
    fix_emission_ca_flag: bool = False,
    compute_quality: bool = True,
    compute_latency: bool = True,
    compute_comet: bool = False,
    comet_model: str = "Unbabel/wmt22-comet-da",
    source_sentences: Optional[List[str]] = None,
    all_have_emission_ca: bool = True,
) -> Tuple[List[Instance], List[dict]]:
    """
    Main resegmentation pipeline: load, align, resegment, and evaluate.

    Exactly one of reference_segmentation or text_segmentation must be provided.

    Args:
        ref_sentences_file: Path to reference sentences file.
        hypothesis_file: Path to hypothesis file (JSONL or text).
        output_folder: Directory for output files. When ``None``, no files are
            written (scores are still returned).
        reference_segmentation: Path to YAML/JSON speech segmentation file.
        text_segmentation: Path to text segmentation file (one 0-based doc_id per line).
        hypothesis_format: Format of hypothesis file ('jsonl', 'text', or 'simulstream').
        char_level: Use character-level units.
        lang: Language code for Moses tokenizer.
        bleu_tokenizer: Tokenizer for SacreBLEU.
        offset_delays: Offset delays relative to first segment.
        fix_emission_ca_flag: Fix computation-aware emission timestamps for CA-YAAL.
        emission_cu_key: JSON key for computation-unaware emission timestamps in JSONL.
        emission_ca_key: JSON key for computation-aware emission timestamps in JSONL.
        compute_quality: Compute quality metrics (BLEU, chrF, COMET).
        compute_latency: Compute latency metrics (YAAL).
        compute_comet: Compute COMET score.
        comet_model: COMET model name.
        source_sentences_file: Path to source sentences file (for COMET).

    Returns:
        Dictionary of computed metric scores.
    """
    # This function expects already-loaded reference/hypothesis data:
    # - ref_words: List[List[Word]] grouped by recording/document
    # - hyp_words: List[List[Word]] grouped by recording/document
    # - segmentation: segmentation metadata (list of dicts)
    # - ref_sentences: list of reference strings (one per segment)

    # Align and build instances
    instances, instances_dict_list = build_resegmented_instances(
        ref_words,
        hyp_words,
        segmentation,
        ref_sentences,
        char_level=char_level,
        lang=lang,
        has_emission_timestamps=all_have_emission_ca,
    )

    return instances, instances_dict_list


def _metric_display_name(key: str, is_longform: bool) -> str:
    """Return a human-friendly display name for a metric key.

    Mirrors the display logic used by the CLI report.
    """
    metric_display = {
        "bleu":              "BLEU",
        "chrf":              "chrF",
        "comet":             "COMET",
        "yaal":              "YAAL (CU)",
        "ca_yaal":           "YAAL (CA)",
        "swf":               "Simultaneous Words Fraction (%)",
        "efsw":              "Expected Simul. Words Fraction (%)",
        "dsptv":             "Degeneracy Test Value",
        "degenerate_policy": "Likely Degenerate Simultaneous Policy",
    }

    if key in metric_display:
        disp = metric_display[key]
        if is_longform and key in ("yaal", "ca_yaal"):
            # Longform YAAL naming
            if key == "yaal":
                return "LongYAAL (CU)"
            else:
                return "LongYAAL (CA)"
        return disp

    if key.startswith("ca_long_"):
        return "Long" + key[len("ca_long_"):].upper() + " (CA)"
    if key.startswith("long_"):
        return "Long" + key[len("long_"):].upper() + " (CU)"
    if key in ("al", "laal", "ap", "dal", "yaal"):
        return key.upper() + " (CU)"
    if key.startswith("ca_"):
        return key[len("ca_"):].upper() + " (CA)"
    return key.upper()


def evaluate_log(
    instances: List[Instance],
    output_folder: Optional[str] = None,
    *,
    is_longform: bool = False,
    char_level: bool = False,
    bleu_tokenizer: str = "13a",
    all_have_emission_ca: bool = False,
    fix_emission_ca_flag: bool = False,
    compute_quality: bool = True,
    compute_latency: bool = True,
    compute_comet: bool = False,
    comet_model: str = "Unbabel/wmt22-comet-da",
    source_sentences: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate a segment-level hypothesis log (shortform or pre-resegmented).

    Supports two input formats, auto-detected from the JSONL fields:

    1. **SimulEval JSONL** (shortform): each line has ``prediction``, ``delays``,
       ``elapsed``, ``source_length``.  Requires ``--ref_sentences_file``.
    2. **Resegmented JSONL** (our format): each line has ``prediction``,
       ``reference``, ``emission_cu``, ``emission_ca``, ``source_length``,
       ``time_to_recording_end``.

    Args:
        hypothesis_file: Path to the JSONL hypothesis file.
        output_folder: Directory for output files. When ``None``, no files are
            written (scores are still returned).
        ref_sentences_file: Path to reference sentences (required for SimulEval format).
        is_longform: Use long-form YAAL (True for pre-resegmented long-form,
            False for native short-form systems).
        char_level: Whether to use character-level latency units.
        bleu_tokenizer: Tokenizer for SacreBLEU.
        emission_cu_key: JSON key for computation-unaware emission timestamps.
        emission_ca_key: JSON key for computation-aware emission timestamps.
        fix_emission_ca_flag: Fix computation-aware emission timestamps.
        compute_quality: Compute quality metrics.
        compute_latency: Compute latency metrics.
        compute_comet: Compute COMET score.
        comet_model: COMET model name.
        source_sentences_file: Path to source sentences file (for COMET).

    Returns:
        Dictionary of computed metric scores.
    """
    # `instances` is expected to be a list of `Instance` objects already
    # prepared by the caller (CLI loader). The caller is also responsible for
    # determining `all_have_emission_ca` and passing `source_sentences` when
    # COMET is requested.

    scores = evaluate_instances(
        instances,
        compute_quality=compute_quality,
        compute_latency=compute_latency,
        is_longform=is_longform,
        bleu_tokenizer=bleu_tokenizer,
        all_have_emission_ca=all_have_emission_ca,
        fix_emission_ca_flag=fix_emission_ca_flag,
        compute_comet=compute_comet,
        comet_model=comet_model,
        source_sentences=source_sentences,
    )

    # Save scores
    if scores and output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        with open(
            os.path.join(output_folder, "scores.tsv"), "w", encoding="utf-8"
        ) as f:
            f.write("metric\tvalue\n")
            for k, v in scores.items():
                name = _metric_display_name(k, is_longform)
                if k == "degenerate_policy":
                    val_str = "YES" if v == 1.0 else ("NO" if v == 0.0 else "N/A")
                else:
                    val_str = f"{v:.4f}"
                f.write(f"{name}\t{val_str}\n")

    return scores
