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

def _normalize_reference_names(segmentation: list) -> list:
    """
    Normalize recording names in segmentation to match hypothesis sources.

    This function extracts the base name without extension from the 'wav' field
    in each segment, which is expected to correspond to the source names in the
    hypothesis data. For example, "recording1.wav" becomes "recording1".

    Args:
        segmentation: Parsed segmentation data with 'wav' fields.

    Returns:
        List of normalized recording names corresponding to each segment.
    """
    names = set(seg["wav"] for seg in segmentation)

    # try removing extensions
    candidate_names = [os.path.splitext(seg["wav"])[0] for seg in segmentation]
    if len(set(candidate_names)) == len(names):
        for seg, candidate_name in zip(segmentation, candidate_names):
            seg["wav"] = candidate_name

    # try base name
    base_names = [os.path.basename(seg["wav"]) for seg in segmentation]
    if len(set(base_names)) == len(names):
        for seg, base_name in zip(segmentation, base_names):
            seg["wav"] = base_name
        return segmentation
    
    return segmentation

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

    segmentation = _normalize_reference_names(segmentation)

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
        if i > 0:
            if word.emission_ca is None or words[i - 1].emission_ca is None or words[i - 1].emission_cu is None:
                raise ValueError(
                    f"Cannot fix emission_ca for word index {i} because of missing values: "
                    f"current emission_ca={word.emission_ca}, previous emission_ca={words[i - 1].emission_ca}, "
                    f"previous emission_cu={words[i - 1].emission_cu}"
                )
            new_emission_ca.append(word.emission_ca - words[i - 1].emission_ca + words[i - 1].emission_cu)  # type: ignore
        else:
            new_emission_ca.append(word.emission_ca)
    for i, word in enumerate(words):
        if i == 0:
            word.emission_ca = new_emission_ca[i]
        else:
            prev_emission_ca = words[i - 1].emission_ca
            if new_emission_ca[i] is None or prev_emission_ca is None:
                raise ValueError(
                    f"Cannot set emission_ca for word index {i} because of missing values: "
                    f"new_emission_ca={new_emission_ca[i]}, previous emission_ca={prev_emission_ca}"
                )
            word.emission_ca = max(new_emission_ca[i], prev_emission_ca)
    return words

def _resolve_recording_names(segmentation_order: List[str], hypothesis_names: List[str]) -> Dict[str, str]:
    """
    Resolve recording names between segmentation and hypothesis data.

    This function attempts to match recording names from the segmentation order
    with those in the hypothesis data, allowing for common variations such as:
    - Segmentation names may be base names without extensions or with different extensions (e.g., "recording1" vs "recording1.wav").
    - Hypothesis names may include extensions or be full paths, while segmentation names are base names.

    """
    segmentation_names = set(segmentation_order)
    mapping = {}
    for hypothesis_name in hypothesis_names:
        base_name = os.path.basename(hypothesis_name)
        no_ext = os.path.splitext(base_name)[0]
        base_name_no_ext = os.path.basename(os.path.splitext(hypothesis_name)[0])
        if hypothesis_name in segmentation_names:
            mapping[hypothesis_name] = hypothesis_name
        elif base_name in segmentation_names:
            mapping[hypothesis_name] = base_name
        elif no_ext in segmentation_names:
            mapping[hypothesis_name] = no_ext
        elif base_name_no_ext in segmentation_names:
            mapping[hypothesis_name] = base_name_no_ext
        else:
            raise ValueError(
                f"Could not resolve recording name for hypothesis '{hypothesis_name}'. "
                f"Segmentation names: {segmentation_names}"
            )
    return mapping

def load_hypothesis_jsonl(
    hypothesis_file: str,
    char_level: bool,
    segmentation_order: List[str],
    fix_emission_ca_flag: bool,
    emission_cu_key: str = "delays",
    emission_ca_key: str = "elapsed",
) -> List[List[Word]]:
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
        List of lists of Word objects, where each inner list corresponds to a recording.
    """
    hypotheses: Dict[str, dict] = {}
    source_lengths: Dict[str, float] = {}
    with open(hypothesis_file, "r", encoding="utf-8") as f:
        for line in f:
            h = json.loads(line.strip())
            source = h["source"]
            if isinstance(source, list):
                h_name = source[0]
            else:
                h_name = source
            if h_name in hypotheses:
                raise ValueError(f"Duplicate hypothesis for source '{h_name}' in hypothesis file.")
            
            source_lengths[h_name] = h.get("source_length", INF)
            hypotheses[h_name] = h

    if len(hypotheses) != len(segmentation_order):
        raise ValueError(
            f"Number of hypotheses ({len(hypotheses)}) does not match number of segments ({len(segmentation_order)}). "
            f"Segmentation order: {segmentation_order}, Hypothesis sources: {list(hypotheses.keys())}"
        )
    
    # Resolve recording names between segmentation and hypothesis data
    hypotheses_names = list(hypotheses.keys())
    name_mapping = _resolve_recording_names(segmentation_order, hypotheses_names)
    source_lengths = {name_mapping[h_name]: length for h_name, length in source_lengths.items()}
    hypotheses = {name_mapping[h_name]: h for h_name, h in hypotheses.items()}

    ordered_hyps = [hypotheses[name] for name in segmentation_order]
    ordered_lengths = [source_lengths[name] for name in segmentation_order]

    num_with_ca = sum(1 for h in ordered_hyps if emission_ca_key in h)
    num_with_cu = sum(1 for h in ordered_hyps if emission_cu_key in h)
    if num_with_ca > 0 and num_with_ca < len(ordered_hyps):
        raise ValueError(f"Some hypotheses have {emission_ca_key} but not all. Found {num_with_ca} with {emission_ca_key} out of {len(ordered_hyps)}.")
    if num_with_cu > 0 and num_with_cu < len(ordered_hyps):
        raise ValueError(f"Some hypotheses have {emission_cu_key} but not all. Found {num_with_cu} with {emission_cu_key} out of {len(ordered_hyps)}.")
    
    words: List[List[Word]] = []
    for i, (h, rec_length) in enumerate(zip(ordered_hyps, ordered_lengths)):
        prediction = unicode_normalize(h["prediction"])
        units = list(prediction) if char_level else prediction.split()

        if emission_cu_key in h:
            cu_values = h[emission_cu_key]
            assert len(units) == len(cu_values), \
                (f"Number of units and {emission_cu_key} do not match for hypothesis {i}: "
                f"{len(units)} vs {len(cu_values)}")
        else:
            cu_values = [None] * len(units)

        if emission_ca_key in h:
            ca_values = h[emission_ca_key]
            assert len(units) == len(ca_values), \
                (f"Number of units and {emission_ca_key} do not match for hypothesis {i}: "
                f"{len(units)} vs {len(ca_values)}")
        else:
            ca_values = [None] * len(units)

        instance_words = [
            Word(unit, emission_cu=emission_cu, emission_ca=emission_ca, recording_length=rec_length)
            for unit, emission_cu, emission_ca in zip(units, cu_values, ca_values)
        ]
        if fix_emission_ca_flag:
            instance_words = fix_emission_ca(instance_words)
        words.append(instance_words)

    return words


def load_hypothesis_simulstream(
    hypothesis_file: str,
    eval_config_file: str,
    char_level: bool,
    segmentation_order: List[str],
) -> List[List[Word]]:
    try:
        logger.info("Loading SimulStream log with config: %s", eval_config_file)
        from simulstream.metrics.readers import LogReader
        from simulstream.config import yaml_config
    except ImportError:
        raise ImportError("Simulstream requires the 'simulstream' package. Install with: pip install simulstream")

    words: List[List[Word]] = []

    eval_config = yaml_config(eval_config_file)

    reader = LogReader(
        filepath=hypothesis_file,
        config=eval_config,
        latency_unit="char" if char_level else "word",
    )

    output_with_latency = reader.final_outputs_and_latencies()
    for recording_name ,hypothesis in output_with_latency.items():
        prediction = unicode_normalize(hypothesis.final_text)
        units = list(prediction) if char_level else prediction.split()

        cu_values = hypothesis.ideal_delays
        ca_values = hypothesis.computational_aware_delays

        if len(units) != len(cu_values):
            raise ValueError(
                f"Number of units ({len(units)}) does not match number of delays ({len(cu_values)}) for output {recording_name}"
            )
        if len(units) != len(ca_values):
            raise ValueError(
                f"Number of units ({len(units)}) does not match number of computational-aware delays ({len(ca_values)}) for output {recording_name}"
            )

        instance_words = [
            Word(unit, emission_cu=cu * 1000, emission_ca=ca * 1000)
            for unit, cu, ca in zip(units, cu_values, ca_values)
        ]
        words.append(instance_words)

    # Map recording names from the log output to their indices in `words`
    name_to_index = {
        recording_name: idx
        for idx, (recording_name, _) in enumerate(output_with_latency.items())
    }

    # Validate that all expected recordings are present in the log output
    missing_recordings = [wav for wav in segmentation_order if wav not in name_to_index]
    if missing_recordings:
        logger.error(
            "The following recordings from segmentation_order are missing in the SimulStream log output: %s",
            ", ".join(missing_recordings),
        )
        raise KeyError(
            f"Missing recordings in SimulStream log output: {', '.join(missing_recordings)}"
        )

    # Reorder `words` to match `segmentation_order`
    reordered_words = [words[name_to_index[wav]] for wav in segmentation_order]

    return reordered_words


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
        words[seg["doc_id"]].extend([Word(unit, seq_id=i) for unit in units])

    return words, segmentation, reference_sentences, num_documents
