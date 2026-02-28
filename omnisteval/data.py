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
Data loading, structures, and I/O for SoftSegmenter.
"""

import json
import logging
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .alignment import Word
from .tokenization import unicode_normalize

logger = logging.getLogger(__name__)

INF = float("inf")


@dataclass
class Instance:
    """
    Stores information about a resegmented output instance.

    Attributes:
        prediction: The hypothesis text for this segment.
        reference: The reference text for this segment.
        latency_unit: Unit for length computation ('word' or 'char').
        index: Segment index.
        source_length: Duration of the source segment (ms).
        emission_cu: Computation-unaware emission timestamps.
        emission_ca: Computation-aware emission timestamps.
        time_to_recording_end: Time remaining until end of the recording, relative to segment offset.
        longform: Whether this is a long-form instance.
        metrics: Dictionary of computed metric scores.
    """

    prediction: str = ""
    reference: str = ""
    latency_unit: str = "word"
    index: Optional[int] = None
    source_length: Optional[float] = None
    emission_cu: Optional[List[float]] = None
    emission_ca: Optional[List[float]] = None
    time_to_recording_end: Optional[float] = None
    longform: Optional[bool] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def reference_length(self) -> int:
        return self.string_to_len(self.reference, self.latency_unit)

    @staticmethod
    def string_to_len(string: str, latency_unit: str) -> int:
        if latency_unit == "word":
            return len(string.split(" "))
        elif latency_unit == "char":
            return len(string.strip())
        else:
            raise ValueError(f"Unknown latency unit: {latency_unit}")

    @classmethod
    def from_dict(cls, info: Dict[str, Any], latency_unit: str = "word") -> "Instance":
        """Create an Instance from a dictionary, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in info.items() if k in known}
        kwargs["latency_unit"] = latency_unit
        return cls(**kwargs)


# unicode normalization is provided by tokenization.unicode_normalize


def load_reference(
    reference_segmentation: str,
    ref_sentences_file: str,
    char_level: bool,
    offset_delays: bool,
) -> Tuple[List[List[Word]], list, List[str]]:
    """
    Load reference segmentation and sentences, and convert to Word objects.

    Args:
        reference_segmentation: Path to YAML or JSON segmentation file.
        ref_sentences_file: Path to reference sentences file (one per line).
        char_level: Whether to use character-level units.
        offset_delays: Whether to offset delays relative to the first segment per recording.

    Returns:
        Tuple of (words, segmentation, reference_sentences) where words is grouped
        by recording, segmentation is the parsed data, and reference_sentences is
        the list of reference strings.
    """
    if reference_segmentation.endswith(".json"):
        with open(reference_segmentation, "r", encoding="utf-8") as f:
            segmentation = json.load(f)
    elif reference_segmentation.endswith((".yaml", ".yml")):
        with open(reference_segmentation, "r", encoding="utf-8") as f:
            segmentation = yaml.load(f, Loader=yaml.CLoader)
    else:
        raise ValueError("Unsupported segmentation file format. Use YAML or JSON.")

    for seg in segmentation:
        seg["duration"] = seg["duration"] * 1000  # Convert to milliseconds
        seg["offset"] = seg["offset"] * 1000

    with open(ref_sentences_file, "r", encoding="utf-8") as f:
        reference_sentences = [line.strip() for line in f]
    if char_level:
        reference_sentences = [s.replace(" ", "") for s in reference_sentences]

    assert len(segmentation) == len(reference_sentences), \
        "Number of segments and reference sentences do not match."

    words: List[List[Word]] = []
    doc_id = -1
    seg_id_counter: Dict[int, int] = {}
    for i, (segment, ref_sentence) in enumerate(zip(segmentation, reference_sentences)):
        if i == 0 or segmentation[i - 1]["wav"] != segment["wav"]:
            first_offset = segment["offset"] if offset_delays else 0
            words.append([])
            doc_id += 1
        if offset_delays:
            segment["offset"] -= first_offset

        seg_id_counter.setdefault(doc_id, 0)
        segment["doc_id"] = doc_id
        segment["seg_id"] = seg_id_counter[doc_id]
        seg_id_counter[doc_id] += 1

        ref_sentence = ref_sentence.strip().lower()
        units = list(ref_sentence) if char_level else ref_sentence.split()

        # The emission timestamp ensures hypothesis words emitted before
        # this segment are not aligned to it
        emission_cu = segment["offset"]
        words[-1].extend([Word(unit, emission_cu=emission_cu, seq_id=i) for unit in units])

    return words, segmentation, reference_sentences


def get_segmentation_order(segmentation: list) -> List[str]:
    """
    Extract the unique ordering of audio files from the segmentation.

    Args:
        segmentation: Parsed segmentation data.

    Returns:
        Ordered list of unique audio file names.
    """
    order: List[str] = []
    for segment in segmentation:
        if not order or order[-1] != segment["wav"]:
            order.append(segment["wav"])
    return order


def fix_emission_ca(words: List[Word]) -> List[Word]:
    """
    Fix computation-aware emission timestamps to be incremental.

    SimulEval computes elapsed as ELAPSED_i = DELAY_i + TOTAL_RUNTIME_i,
    but we need NEW_EMISSION_CA_i = EMISSION_CA_i - EMISSION_CA_{i-1} + EMISSION_CU_{i-1}
    to make timestamps relative to the previous word.
    """
    new_emission_ca = []
    for i, word in enumerate(words):
        assert word.emission_ca is not None and word.emission_cu is not None, \
            "Words must have both emission_ca and emission_cu to fix timestamps."
        if i > 0:
            new_emission_ca.append(word.emission_ca - words[i - 1].emission_ca + words[i - 1].emission_cu)
        else:
            new_emission_ca.append(word.emission_ca)
    for i, word in enumerate(words):
        if i == 0:
            word.emission_ca = new_emission_ca[i]
        else:
            prev_emission_ca = words[i - 1].emission_ca
            assert prev_emission_ca is not None, "Previous word must have emission_ca set."
            word.emission_ca = max(new_emission_ca[i], prev_emission_ca)
    return words


def load_hypothesis_jsonl(
    hypothesis_file: str,
    char_level: bool,
    segmentation_order: List[str],
    fix_emission_ca_flag: bool,
    emission_cu_key: str = "delays",
    emission_ca_key: str = "elapsed",
) -> Tuple[List[List[Word]], bool]:
    """
    Load hypothesis from a JSONL file (SimulEval output format).

    Each line is a JSON object with 'source', 'prediction', and emission timestamp fields.

    Args:
        hypothesis_file: Path to the JSONL file.
        char_level: Whether to use character-level units.
        segmentation_order: Ordered list of audio file names.
        fix_emission_ca_flag: Whether to fix computation-aware emission timestamps.
        emission_cu_key: JSON key for computation-unaware emission timestamps (default: 'delays').
        emission_ca_key: JSON key for computation-aware emission timestamps (default: 'elapsed').

    Returns:
        Tuple of (words, all_have_emission_ca) where words is grouped by recording.
    """
    hypotheses: Dict[str, dict] = {}
    source_lengths: Dict[str, float] = {}
    with open(hypothesis_file, "r", encoding="utf-8") as f:
        for line in f:
            h = json.loads(line.strip())
            source = h["source"]
            h_name = os.path.basename(source[0] if isinstance(source, list) else source)
            assert h_name in segmentation_order, f"Missing hypothesis for {h_name}"
            assert h_name not in hypotheses, f"Duplicate hypothesis for {h_name}"
            source_lengths[h_name] = h.get("source_length", INF)
            hypotheses[h_name] = h

    assert len(hypotheses) == len(segmentation_order), \
        "Number of hypotheses and recordings do not match."

    ordered_hyps = [hypotheses[name] for name in segmentation_order]
    ordered_lengths = [source_lengths[name] for name in segmentation_order]

    all_have_emission_ca = all(emission_ca_key in h for h in ordered_hyps)

    words: List[List[Word]] = []
    for i, (h, rec_length) in enumerate(zip(ordered_hyps, ordered_lengths)):
        prediction = unicode_normalize(h["prediction"])
        units = list(prediction) if char_level else prediction.split()
        cu_values = h[emission_cu_key]
        assert len(units) == len(cu_values), \
            (f"Number of units and {emission_cu_key} do not match for hypothesis {i}: "
             f"{len(units)} vs {len(cu_values)}")

        if emission_ca_key not in h:
            ca_values = list(cu_values)
        else:
            ca_values = h[emission_ca_key]
        assert len(units) == len(ca_values), \
            (f"Number of units and {emission_ca_key} do not match for hypothesis {i}: "
             f"{len(units)} vs {len(ca_values)}")

        instance_words = [
            Word(unit, emission_cu=emission_cu, emission_ca=emission_ca, recording_length=rec_length)
            for unit, emission_cu, emission_ca in zip(units, cu_values, ca_values)
        ]
        if fix_emission_ca_flag:
            instance_words = fix_emission_ca(instance_words)
        words.append(instance_words)

    return words, all_have_emission_ca


def load_hypothesis_text(
    hypothesis_file: str,
    char_level: bool,
    num_documents: int,
) -> List[List[Word]]:
    """
    Load hypothesis from a plain text file (one line per document).

    Lines must be in the same order as the documents.
    No emission timestamp information is available in text-only mode.

    Args:
        hypothesis_file: Path to the text file.
        char_level: Whether to use character-level units.
        num_documents: Expected number of documents (hypothesis lines).

    Returns:
        List of lists of Word objects (one per document), with delays set to 0.
    """
    with open(hypothesis_file, "r", encoding="utf-8") as f:
        hypotheses = [line.strip() for line in f]

    assert len(hypotheses) == num_documents, \
        (f"Number of hypothesis lines ({len(hypotheses)}) does not match "
         f"number of documents ({num_documents})")

    words: List[List[Word]] = []
    for h_text in hypotheses:
        prediction = unicode_normalize(h_text)
        units = list(prediction) if char_level else prediction.split()
        instance_words = [
            Word(unit)
            for unit in units
        ]
        words.append(instance_words)
    return words


def load_text_segmentation(
    text_segmentation_file: str,
    ref_sentences_file: str,
    char_level: bool,
) -> Tuple[List[List[Word]], list, List[str], int]:
    """
    Load text-based segmentation and reference sentences.

    The text segmentation file has one entry per line in the format:
        docid=DOC_ID,segid=SEG_ID
    where DOC_ID is a 0-based document index and SEG_ID is a 0-based
    segment index within that document. Each line corresponds to one
    reference sentence. The number of unique document IDs must equal
    the number of hypothesis lines.

    Args:
        text_segmentation_file: Path to the text segmentation file.
        ref_sentences_file: Path to reference sentences file (one per line).
        char_level: Whether to use character-level units.

    Returns:
        Tuple of (words, segmentation, reference_sentences, num_documents) where
        words is grouped by document, segmentation is a list of dicts with
        'doc_id' per segment, reference_sentences is the list of reference
        strings, and num_documents is the total number of unique documents.
    """
    doc_ids = []
    seg_ids = []
    with open(text_segmentation_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            try:
                parts = dict(p.split("=") for p in line.split(","))
                doc_ids.append(int(parts["docid"]))
                seg_ids.append(int(parts["segid"]))
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"Invalid text segmentation format at line {line_num}: '{line}'. "
                    f"Expected format: docid=DOC_ID,segid=SEG_ID"
                ) from e

    with open(ref_sentences_file, "r", encoding="utf-8") as f:
        reference_sentences = [line.strip() for line in f]
    if char_level:
        reference_sentences = [s.replace(" ", "") for s in reference_sentences]

    assert len(doc_ids) == len(reference_sentences), \
        (f"Number of segmentation entries ({len(doc_ids)}) does not match "
         f"number of reference sentences ({len(reference_sentences)})")

    num_documents = max(doc_ids) + 1
    assert all(0 <= d < num_documents for d in doc_ids), \
        "Document IDs must be 0-based contiguous integers."

    # Build segmentation metadata (analogous to speech segmentation)
    segmentation = []
    for i, (doc_id, seg_id) in enumerate(zip(doc_ids, seg_ids)):
        segmentation.append({"doc_id": doc_id, "seg_id": seg_id, "offset": 0.0, "duration": 0.0})

    # Group reference words by document
    words: List[List[Word]] = [[] for _ in range(num_documents)]
    for i, (seg, ref_sentence) in enumerate(zip(segmentation, reference_sentences)):
        ref_sentence_lower = ref_sentence.strip().lower()
        units = list(ref_sentence_lower) if char_level else ref_sentence_lower.split()
        words[seg["doc_id"]].extend([Word(unit, emission_cu=0.0, seq_id=i) for unit in units])

    return words, segmentation, reference_sentences, num_documents
