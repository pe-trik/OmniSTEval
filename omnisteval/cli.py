#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
OmniSTEval CLI.

Provides two subcommands:

  shortform
    Evaluate shortform (segment-level) SimulEval outputs directly.
    No resegmentation is needed. YAAL is computed with is_longform=False.

  longform
    Either re-segment long-form translation outputs to match reference
    segmentation and then evaluate, or evaluate an already-resegmented log.
    YAAL is computed with is_longform=True.

Based on the SoftSegmenter algorithm from "Better Late Than Never:
Evaluation of Latency Metrics for Simultaneous Speech-to-Text Translation"
(https://arxiv.org/abs/2509.17349).
"""

import logging
import os
import json
from argparse import ArgumentParser

from .resegment import evaluate_log, resegment, _metric_display_name
from .io import (
    load_shortform_instances,
    load_pre_resegmented_instances,
    load_resegmentation_inputs,
    dump_instances_jsonl,
    dump_scores_tsv,
    format_report,
)
from .data import Instance
from . import __version__

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_WIDTH = 64


# Use shared `format_report` from omnisteval.io


def _save_report(report: str, output_folder: str) -> None:
    """Write *report* to evaluation_report.txt inside *output_folder*."""
    os.makedirs(output_folder, exist_ok=True)
    path = os.path.join(output_folder, "evaluation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")


def _evaluate_and_report(
    instances,
    title: str,
    settings: dict,
    args,
    is_longform: bool,
    all_have_emission_ca: bool,
    *,
    source_sentences=None,
):
    """Evaluate given `instances`, print formatted report and dump outputs.

    Returns the computed `scores` dict.
    """
    scores = evaluate_log(
        instances=instances,
        output_folder=None,
        is_longform=is_longform,
        char_level=args.char_level,
        bleu_tokenizer=args.bleu_tokenizer,
        all_have_emission_ca=all_have_emission_ca,
        fix_emission_ca_flag=args.fix_simuleval_emission_ca,
        compute_quality=not args.no_quality,
        compute_latency=not args.no_latency,
        compute_comet=args.comet,
        comet_model=args.comet_model,
        source_sentences=source_sentences,
    )

    report = format_report(title, settings, scores)
    print(report)
    if args.output_folder:
        dump_scores_tsv(scores, args.output_folder, is_longform=is_longform)
        _save_report(report, args.output_folder)

    return scores


def _build_settings(args, base: dict) -> dict:
    """Build common `settings` dict for report formatting.

    `base` should contain keys specific to the command/mode; this helper
    adds `Metrics`, `Version` and the optional `COMET model` entry.
    """
    metrics_enabled = ", ".join(filter(None, [
        "quality" if not args.no_quality else None,
        "latency" if not args.no_latency else None,
        "COMET" if args.comet else None,
    ])) or "none"
    settings = dict(base)
    settings["Metrics"] = metrics_enabled
    settings["Version"] = __version__
    if args.comet:
        settings["COMET model"] = args.comet_model
    return settings


def _build_base(mode: str, args) -> dict:
    """Return a base settings `dict` for the given *mode*.

    Mode is one of: "shortform", "pre_resegmented", "resegmentation".
    """
    seg_type = "speech" if getattr(args, "speech_segmentation", None) else "text"
    if mode == "shortform":
        return {
            "Hypothesis":        args.hypothesis_file,
            "Reference":         args.ref_sentences_file,
            "BLEU tokenizer":    args.bleu_tokenizer,
            "Char-level":        "yes" if args.char_level else "no",
            "Fix CA emissions":  "yes" if args.fix_simuleval_emission_ca else "no",
        }
    if mode == "pre_resegmented":
        return {
            "Hypothesis":        args.resegmented_hypothesis,
            "BLEU tokenizer":    args.bleu_tokenizer,
            "Char-level":        "yes" if args.char_level else "no",
            "Fix CA emissions":  "yes" if args.fix_simuleval_emission_ca else "no",
            "Segmentation":      args.speech_segmentation or args.text_segmentation,
            "Seg. type":         seg_type,
            "Language":          args.lang or "none",
        }
    if mode == "resegmentation":
        return {
            "Hypothesis":        args.hypothesis_file,
            "Hypothesis format": args.hypothesis_format,
            "Reference":         args.ref_sentences_file,
            "Segmentation":      args.speech_segmentation or args.text_segmentation,
            "Seg. type":         seg_type,
            "Language":          args.lang or "none",
            "BLEU tokenizer":    args.bleu_tokenizer,
            "Char-level":        "yes" if args.char_level else "no",
            "Offset delays":     "yes" if args.offset_delays else "no",
            "Fix CA emissions":  "yes" if args.fix_simuleval_emission_ca else "no",
        }
    raise ValueError(f"Unknown base mode: {mode}")


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------

def _add_output_args(parser):
    """Add output arguments (shared by both subcommands)."""
    group = parser.add_argument_group("Output")
    group.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help=(
            "Directory where output files will be written. "
            "When omitted, results are printed to stdout only."
        ),
    )


def _add_metric_args(parser):
    """Add metric-selection arguments (shared by both subcommands)."""
    group = parser.add_argument_group("Metrics")
    group.add_argument(
        "--no-quality",
        action="store_true",
        default=False,
        help="Disable quality metrics (BLEU, chrF, COMET).",
    )
    group.add_argument(
        "--no-latency",
        action="store_true",
        default=False,
        help="Disable latency metrics (YAAL).",
    )
    group.add_argument(
        "--comet",
        action="store_true",
        default=False,
        help=(
            "Enable COMET scoring. Requires --source_sentences_file and the "
            "'unbabel-comet' package (pip install unbabel-comet)."
        ),
    )
    group.add_argument(
        "--comet_model",
        type=str,
        default="Unbabel/wmt22-comet-da",
        help="COMET model name. Default: 'Unbabel/wmt22-comet-da'.",
    )
    group.add_argument(
        "--bleu_tokenizer",
        type=str,
        default="13a",
        help=(
            "Tokenizer for SacreBLEU (e.g., '13a', 'intl', 'ja-mecab', 'zh'). "
            "Default: '13a'."
        ),
    )
    group.add_argument(
        "--source_sentences_file",
        type=str,
        default=None,
        help=(
            "Path to source sentences file (one per segment, for COMET scoring). "
            "Required when --comet is enabled."
        ),
    )


def _add_jsonl_field_args(parser):
    """Add JSONL field-name arguments (shared by both subcommands)."""
    group = parser.add_argument_group("JSONL field names")
    group.add_argument(
        "--emission_cu_key",
        type=str,
        default="delays",
        help=(
            "JSON key for computation-unaware emission timestamps in JSONL "
            "hypothesis file. Default: 'delays'."
        ),
    )
    group.add_argument(
        "--emission_ca_key",
        type=str,
        default="elapsed",
        help=(
            "JSON key for computation-aware emission timestamps in JSONL "
            "hypothesis file. Default: 'elapsed'."
        ),
    )


def _add_processing_args(parser):
    """Add misc processing arguments (shared by both subcommands)."""
    group = parser.add_argument_group("Processing options")
    group.add_argument(
        "--char_level",
        action="store_true",
        help="Use character-level alignment and scoring instead of word-level.",
    )
    group.add_argument(
        "--fix_simuleval_emission_ca",
        action="store_true",
        help=(
            "Fix computation-aware emission timestamps for CA-YAAL. "
            "Corrects SimulEval's cumulative elapsed times to incremental values."
        ),
    )


# ---------------------------------------------------------------------------
# Subcommand parsers
# ---------------------------------------------------------------------------

def _build_shortform_parser(subparsers):
    parser = subparsers.add_parser(
        "shortform",
        help="Evaluate shortform (segment-level) outputs.",
        description=(
            "Evaluate a segment-level SimulEval JSONL log against references. "
            "No resegmentation is performed; each line is taken as one segment. "
            "YAAL is computed with is_longform=False."
        ),
    )

    inp = parser.add_argument_group("Input files")
    inp.add_argument(
        "--hypothesis_file",
        type=str,
        required=True,
        help=(
            "Path to the hypothesis file (JSONL, one JSON per line with "
            "'prediction', delays, elapsed, source_length)."
        ),
    )
    inp.add_argument(
        "--ref_sentences_file",
        type=str,
        required=True,
        help="Path to the reference sentences file (one sentence per line).",
    )

    _add_output_args(parser)
    _add_metric_args(parser)
    _add_jsonl_field_args(parser)
    _add_processing_args(parser)
    return parser


def _build_longform_parser(subparsers):
    parser = subparsers.add_parser(
        "longform",
        help="Re-segment and/or evaluate long-form outputs.",
        description=(
            "Either (A) re-segment long-form translation outputs to match "
            "reference segmentation and evaluate, or (B) evaluate an "
            "already-resegmented JSONL log. YAAL is computed with "
            "is_longform=True in both cases."
        ),
    )

    # --- Mode A: resegmentation ---
    seg = parser.add_argument_group(
        "Resegmentation inputs",
        "Provide these to re-segment a raw hypothesis file. "
        "Mutually exclusive with --resegmented_hypothesis.",
    )
    seg.add_argument(
        "--speech_segmentation",
        type=str,
        default=None,
        help=(
            "Path to the YAML or JSON file with speech segmentation. "
            "Each entry must have 'wav', 'offset', and 'duration' fields. "
            "Mutually exclusive with --text_segmentation."
        ),
    )
    seg.add_argument(
        "--text_segmentation",
        type=str,
        default=None,
        help=(
            "Path to a text segmentation file. Each line has the format "
            "'docid=DOC_ID,segid=SEG_ID' (0-based), one line per reference "
            "sentence. Mutually exclusive with --speech_segmentation."
        ),
    )
    seg.add_argument(
        "--ref_sentences_file",
        type=str,
        default=None,
        help="Path to the reference sentences file (one sentence per line).",
    )
    seg.add_argument(
        "--hypothesis_file",
        type=str,
        default=None,
        help=(
            "Path to the hypothesis file (JSONL or plain text, "
            "see --hypothesis_format)."
        ),
    )
    seg.add_argument(
        "--hypothesis_format",
        type=str,
        choices=["jsonl", "text", "simulstream"],
        default="jsonl",
        help=(
            "Format of the hypothesis file. "
            "'jsonl': SimulEval output (one JSON per line). "
            "'text': one hypothesis per line. "
            "'simulstream': Simulstream log format (streaming log). Default: jsonl."
        ),
    )

    # --- Mode B: pre-resegmented evaluation ---
    reseg = parser.add_argument_group(
        "Pre-resegmented evaluation",
        "Provide --resegmented_hypothesis to evaluate an already-resegmented "
        "JSONL log. Mutually exclusive with segmentation inputs.",
    )
    reseg.add_argument(
        "--resegmented_hypothesis",
        type=str,
        default=None,
        help=(
            "Path to a resegmented JSONL file (e.g. the output of a previous "
            "'omnisteval longform' run). Each line has 'prediction', "
            "'reference', 'emission_cu', 'emission_ca', 'source_length', etc."
        ),
    )

    # --- Resegmentation-specific options ---
    ropt = parser.add_argument_group("Resegmentation options")
    ropt.add_argument(
        "--lang",
        type=str,
        default=None,
        help=(
            "Language code for Moses tokenizer (e.g., 'en', 'de'). "
            "Use None for Chinese/Japanese or to skip tokenization."
        ),
    )
    ropt.add_argument(
        "--offset_delays",
        action="store_true",
        help=(
            "Offset delays relative to the first segment of each recording. "
            "Useful when hypothesis delays are relative to the concatenated "
            "recording start."
        ),
    )

    _add_output_args(parser)
    _add_metric_args(parser)
    _add_jsonl_field_args(parser)
    _add_processing_args(parser)
    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _run_shortform(parser, args):
    """Handle the ``shortform`` subcommand."""
    if args.comet and args.source_sentences_file is None:
        parser.error("--source_sentences_file is required when --comet is enabled.")
    if args.comet and args.no_quality:
        logger.warning("--comet is ignored when --no-quality is set.")

    instances, all_have_emission_ca = load_shortform_instances(
        args.hypothesis_file,
        args.ref_sentences_file,
        emission_cu_key=args.emission_cu_key,
        emission_ca_key=args.emission_ca_key,
        char_level=args.char_level,
    )

    source_sentences = None
    if args.comet and args.source_sentences_file is not None:
        with open(args.source_sentences_file, "r", encoding="utf-8") as f:
            source_sentences = [line.strip() for line in f]

    base = _build_base("shortform", args)
    settings = _build_settings(args, base)

    scores = _evaluate_and_report(
        instances,
        "Shortform evaluation",
        settings,
        args,
        is_longform=False,
        all_have_emission_ca=all_have_emission_ca,
        source_sentences=source_sentences,
    )


def _run_longform(parser, args):
    """Handle the ``longform`` subcommand."""
    has_segmentation = (
        args.speech_segmentation is not None
        or args.text_segmentation is not None
    )
    has_resegmented = args.resegmented_hypothesis is not None

    # --- mutual-exclusion validation ---
    if not has_segmentation and not has_resegmented:
        parser.error(
            "Provide segmentation inputs (--speech_segmentation or "
            "--text_segmentation with --hypothesis_file and "
            "--ref_sentences_file), or --resegmented_hypothesis."
        )
    if has_segmentation and has_resegmented:
        parser.error(
            "--resegmented_hypothesis is mutually exclusive with "
            "--speech_segmentation / --text_segmentation."
        )

    if args.comet and args.source_sentences_file is None:
        parser.error("--source_sentences_file is required when --comet is enabled.")
    if args.comet and args.no_quality:
        logger.warning("--comet is ignored when --no-quality is set.")

    # Load source sentences (optional)
    source_sentences = None
    if args.comet and args.source_sentences_file is not None:
        with open(args.source_sentences_file, "r", encoding="utf-8") as f:
            source_sentences = [line.strip() for line in f]

    # Two main flows: pre-resegmented evaluation or resegmentation + evaluation.
    if has_resegmented:
        instances, all_have_emission_ca = load_pre_resegmented_instances(
            args.resegmented_hypothesis, char_level=args.char_level
        )
        mode = "pre_resegmented"
    else:
        # Validate resegmentation-specific requirements
        if args.ref_sentences_file is None:
            parser.error("--ref_sentences_file is required for resegmentation mode.")
        if args.hypothesis_file is None:
            parser.error("--hypothesis_file is required for resegmentation mode.")
        if args.speech_segmentation is not None and args.text_segmentation is not None:
            parser.error(
                "--speech_segmentation and --text_segmentation are mutually exclusive."
            )

        if args.text_segmentation is not None:
            if args.hypothesis_format != "text":
                logger.warning(
                    "--text_segmentation forces --hypothesis_format=text. Overriding."
                )
                args.hypothesis_format = "text"
            if not args.no_latency:
                logger.info(
                    "Text segmentation does not support latency metrics. Disabling latency."
                )
                args.no_latency = True

        if args.hypothesis_format == "text" and not args.no_latency:
            logger.warning(
                "Text-only hypothesis format does not support latency metrics. Disabling latency."
            )
            args.no_latency = True

        # Load inputs for resegmentation and run it
        ref_words, hyp_words, segmentation, ref_sentences, all_have_emission_ca = load_resegmentation_inputs(
            args.speech_segmentation,
            args.text_segmentation,
            args.ref_sentences_file,
            args.hypothesis_file,
            hypothesis_format=args.hypothesis_format,
            char_level=args.char_level,
            offset_delays=args.offset_delays,
            fix_emission_ca_flag=args.fix_simuleval_emission_ca,
            emission_cu_key=args.emission_cu_key,
            emission_ca_key=args.emission_ca_key,
        )

        if args.comet and source_sentences is not None:
            assert len(source_sentences) == len(ref_sentences), (
                f"Number of source sentences ({len(source_sentences)}) does not match number of reference segments ({len(ref_sentences)})"
            )

        instances, instances_dict_list = resegment(
            ref_words=ref_words,
            hyp_words=hyp_words,
            segmentation=segmentation,
            ref_sentences=ref_sentences,
            output_folder=None,
            char_level=args.char_level,
            lang=args.lang,
            bleu_tokenizer=args.bleu_tokenizer,
            fix_emission_ca_flag=args.fix_simuleval_emission_ca,
            compute_quality=not args.no_quality,
            compute_latency=not args.no_latency,
            compute_comet=args.comet,
            comet_model=args.comet_model,
            source_sentences=source_sentences,
            all_have_emission_ca=all_have_emission_ca,
        )

        if args.output_folder:
            dump_instances_jsonl(instances_dict_list, args.output_folder)

        mode = "resegmentation"

    # Common evaluation and reporting
    base = _build_base(mode, args)
    settings = _build_settings(args, base)
    title = "Longform evaluation (pre-resegmented)" if mode == "pre_resegmented" else "Longform evaluation (with resegmentation)"
    _evaluate_and_report(
        instances,
        title,
        settings,
        args,
        is_longform=True,
        all_have_emission_ca=all_have_emission_ca,
        source_sentences=source_sentences,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = ArgumentParser(
        prog="omnisteval",
        description=(
            "OmniSTEval: Evaluate simultaneous speech/text translation "
            "systems (shortform and long-form)."
        ),
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    _build_shortform_parser(subparsers)
    _build_longform_parser(subparsers)

    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    # Mosestokenizer is very verbose at INFO; silence it by default.
    logging.getLogger("mosestokenizer").setLevel(logging.WARNING)
    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        parser.exit(1)

    if args.subcommand == "shortform":
        _run_shortform(parser, args)
    elif args.subcommand == "longform":
        _run_longform(parser, args)


if __name__ == "__main__":
    main()
