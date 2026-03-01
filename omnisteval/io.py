"""I/O and reporting helpers for OmniSTEval.

Contains shared functions for loading hypothesis/reference data, dumping
instances and scores, and formatting reports. This centralises duplicated
logic used by the CLI and resegmentation pipeline.
"""
from typing import List, Tuple, Optional, Dict
import json
import os

from .data import (
    Instance,
    load_reference,
    load_hypothesis_jsonl,
    load_hypothesis_text,
    load_text_segmentation,
    get_segmentation_order,
    load_hypothesis_simulstream,
)
from . import __version__


def load_shortform_instances(
    hypothesis_file: str,
    ref_sentences_file: str,
    emission_cu_key: str = "delays",
    emission_ca_key: str = "elapsed",
    char_level: bool = False,
) -> Tuple[List[Instance], bool]:
    """Load SimulEval-style shortform JSONL + references and return Instances.

    Returns tuple (instances, all_have_emission_ca).
    """
    with open(hypothesis_file, "r", encoding="utf-8") as f:
        hyp_lines = [json.loads(line.strip()) for line in f if line.strip()]
    assert hyp_lines, f"Hypothesis file is empty: {hypothesis_file}"

    with open(ref_sentences_file, "r", encoding="utf-8") as f:
        ref_sentences = [line.strip() for line in f]
    assert len(ref_sentences) == len(hyp_lines), (
        f"Number of references ({len(ref_sentences)}) does not match number of hypothesis lines ({len(hyp_lines)})."
    )

    all_have_emission_ca = all(emission_ca_key in h for h in hyp_lines)
    instances: List[Instance] = []
    for i, (h, ref) in enumerate(zip(hyp_lines, ref_sentences)):
        prediction = h.get("prediction", "")
        cu_values = h.get(emission_cu_key, [])
        ca_values = h.get(emission_ca_key, list(cu_values))
        source_length = h.get("source_length", None)

        info: dict = {
            "index": i,
            "prediction": prediction,
            "reference": ref,
        }
        if source_length is not None:
            info["source_length"] = source_length
        if cu_values:
            info["emission_cu"] = cu_values
        if ca_values:
            info["emission_ca"] = ca_values

        instances.append(Instance.from_dict(info, latency_unit=("char" if char_level else "word")))

    return instances, all_have_emission_ca


def load_pre_resegmented_instances(resegmented_file: str, char_level: bool = False) -> Tuple[List[Instance], bool]:
    with open(resegmented_file, "r", encoding="utf-8") as f:
        hyp_lines = [json.loads(line.strip()) for line in f if line.strip()]
    assert hyp_lines, f"Hypothesis file is empty: {resegmented_file}"
    latency_unit = "char" if char_level else "word"
    instances = [Instance.from_dict(h, latency_unit=latency_unit) for h in hyp_lines]
    all_have_emission_ca = all(
        h.get("emission_ca") is not None and len(h.get("emission_ca", [])) > 0
        for h in hyp_lines
    )
    return instances, all_have_emission_ca


def load_resegmentation_inputs(
    speech_segmentation: Optional[str],
    text_segmentation: Optional[str],
    ref_sentences_file: str,
    hypothesis_file: str,
    hypothesis_format: str = "jsonl",
    char_level: bool = False,
    offset_delays: bool = False,
    fix_emission_ca_flag: bool = False,
    emission_cu_key: str = "delays",
    emission_ca_key: str = "elapsed",
) -> Tuple[List[List], List[List], list, List[str], bool]:
    """Load inputs required for resegmentation.

    Returns (ref_words, hyp_words, segmentation, ref_sentences, all_have_emission_ca).
    """
    if speech_segmentation is not None:
        ref_words, segmentation, ref_sentences = load_reference(
            speech_segmentation, ref_sentences_file, char_level, offset_delays
        )
        segmentation_order = get_segmentation_order(segmentation)

        if hypothesis_format == "jsonl":
            hyp_words, all_have_emission_ca = load_hypothesis_jsonl(
                hypothesis_file,
                char_level,
                segmentation_order,
                fix_emission_ca_flag,
                emission_cu_key,
                emission_ca_key,
            )
        elif hypothesis_format == "text":
            hyp_words = load_hypothesis_text(hypothesis_file, char_level, len(segmentation_order))
            all_have_emission_ca = False
        elif hypothesis_format == "simulstream":
            hyp_words, all_have_emission_ca = load_hypothesis_simulstream(
                hypothesis_file, char_level, segmentation_order, fix_emission_ca_flag
            )
        else:
            raise ValueError(f"Unknown hypothesis format: {hypothesis_format}")
    elif text_segmentation is not None:
        # text segmentation
        ref_words, segmentation, ref_sentences, num_documents = load_text_segmentation(
            text_segmentation, ref_sentences_file, char_level
        )
        hyp_words = load_hypothesis_text(hypothesis_file, char_level, num_documents)
        all_have_emission_ca = False
    else:
        raise ValueError("Either speech_segmentation or text_segmentation must be provided.")

    return ref_words, hyp_words, segmentation, ref_sentences, all_have_emission_ca


def dump_instances_jsonl(instances_dict_list: List[dict], output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "instances.resegmented.jsonl"), "w", encoding="utf-8") as f:
        for d in instances_dict_list:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def dump_scores_tsv(scores: Dict[str, float], output_folder: str, is_longform: bool = False) -> None:
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "scores.tsv"), "w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        for k, v in scores.items():
            f.write(f"{k}\t{v:.4f}\n")


def format_report(mode_label: str, settings: dict, scores: dict) -> str:
    """Return a human-readable evaluation report string.

    This mirrors the previous CLI formatting.
    """
    is_longform = "longform" in mode_label.lower()
    lines: list = ["=" * 64, f"OmniSTEval v{__version__}  |  {mode_label}", "=" * 64, ""]

    if settings:
        lines.append("Settings")
        lines.append("-" * 64)
        kw = max(len(k) for k in settings)
        for k, v in settings.items():
            lines.append(f"  {k:<{kw}}  {v}")
        lines.append("")

    if scores:
        lines.append("Scores")
        lines.append("-" * 64)
        rendered = {}
        for k, v in scores.items():
            # Use a simple uppercase name for now; the CLI will map nicer names
            rendered[k.upper()] = f"YES" if (k == "degenerate_policy" and v == 1.0) else ("NO" if (k == "degenerate_policy" and v == 0.0) else f"{v:.4f}")
        kw = max(len(k) for k in rendered)
        for k, v in rendered.items():
            lines.append(f"  {k:<{kw}}  {v}")
        lines.append("")

        if scores.get("degenerate_policy") == 1.0:
            lines.append("  *** Likely Degenerate Simultaneous Policy ***")
            lines.append("")

    lines.append("=" * 64)
    return "\n".join(lines)
