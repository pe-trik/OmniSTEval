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
OmniSTEval — Evaluate simultaneous speech/text translation systems.

Supports both shortform (segment-level) and longform (document-level)
evaluation. For longform systems, re-segments translation outputs
to match reference segmentation, enabling segment-level quality and
latency evaluation.

Implements the SoftSegmenter alignment algorithm from "Better Late Than Never:
Evaluation of Latency Metrics for Simultaneous Speech-to-Text Translation"
(https://arxiv.org/abs/2509.17349).
"""

__version__ = "0.1.1"

from .alignment import Word, align_words, align_sequences, similarity
from .data import Instance
from .resegment import resegment, build_resegmented_instances, evaluate_log
from .scoring import (
    evaluate_instances,
    SacreBLEUScorer,
    ChrFScorer,
    COMETScorer,
    YAALScorer,
)
from .tokenization import tokenize_words

# Internal data-loading helpers (not part of the public API).
from .data import (
    load_reference as _load_reference,
    load_hypothesis_jsonl as _load_hypothesis_jsonl,
    load_hypothesis_text as _load_hypothesis_text,
    load_text_segmentation as _load_text_segmentation,
)

__all__ = [
    # Core types
    "Word",
    "Instance",
    # Main pipeline
    "resegment",
    "build_resegmented_instances",
    "evaluate_log",
    # Alignment
    "align_words",
    "align_sequences",
    "similarity",
    # Evaluation
    "evaluate_instances",
    "SacreBLEUScorer",
    "ChrFScorer",
    "COMETScorer",
    "YAALScorer",
    # Tokenization
    "tokenize_words",
]
