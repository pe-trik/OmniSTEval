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
Quality and latency scoring for SoftSegmenter.

Provides scorers for BLEU, chrF, COMET (quality) and YAAL (latency).
 
 Note on provenance:
     The implementations of AL, LAAL, AP and DAL latency scorers are adapted
     from the SimulEval project (latency_scorer.py). See SimulEval for the
     original implementations and licensing considerations; this file
     documents that these algorithms were adapted for use in OmniSTEval.
"""

import logging
import math
from statistics import mean
from typing import Dict, List, Optional, Sequence

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

from .data import Instance

logger = logging.getLogger(__name__)


class SacreBLEUScorer:
    """Corpus-level BLEU scorer using SacreBLEU."""

    def __init__(self, tokenizer: str = "13a"):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Instance]) -> float:
        try:
            return (
                BLEU(tokenize=self.tokenizer)
                .corpus_score(
                    [ins.prediction for ins in instances],
                    [[ins.reference for ins in instances]],
                )
                .score
            )
        except Exception as e:
            logger.error(f"BLEU scoring failed: {e}")
            return 0.0


class ChrFScorer:
    """Corpus-level chrF scorer using SacreBLEU."""

    def __call__(self, instances: Sequence[Instance]) -> float:
        try:
            return (
                CHRF()
                .corpus_score(
                    [ins.prediction for ins in instances],
                    [[ins.reference for ins in instances]],
                )
                .score
            )
        except Exception as e:
            logger.error(f"chrF scoring failed: {e}")
            return 0.0


class COMETScorer:
    """
    COMET scorer using the unbabel-comet library.

    Requires the 'unbabel-comet' package and source sentences.
    The model is loaded lazily on first use.
    """

    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        try:
            from comet import download_model, load_from_checkpoint # type: ignore
            model_path = download_model(self.model_name)
            self.model = load_from_checkpoint(model_path)
        except ImportError:
            raise ImportError(
                "COMET requires the 'unbabel-comet' package. "
                "Install with: pip install unbabel-comet"
            )

    def __call__(
        self,
        instances: Sequence[Instance],
        source_sentences: List[str],
    ) -> float:
        self._load_model()
        assert self.model is not None, "COMET model failed to load."
        data = [
            {
                "src": src,
                "mt": ins.prediction,
                "ref": ins.reference,
            }
            for src, ins in zip(source_sentences, instances)
        ]
        output = self.model.predict(data, batch_size=8, gpus=0)
        return output.system_score


class ALScorer:
    """Average Lagging (AL) scorer."""

    def __init__(self, computation_aware: bool = False):
        self.computation_aware = computation_aware

    def get_delays_lengths(self, ins: Instance):
        timestamp_type = "emission_ca" if self.computation_aware else "emission_cu"
        delays = getattr(ins, timestamp_type, None)
        assert delays
        tgt_len = len(delays) if ins.reference is None else ins.reference_length
        src_len = getattr(ins, "source_length", None)
        return delays, src_len, tgt_len

    def compute(self, ins: Instance) -> Optional[float]:
        delays, source_length, target_length = self.get_delays_lengths(ins)
        if source_length is None or source_length <= 0:
            return None
        if delays[0] > source_length:
            return delays[0]

        AL = 0.0
        gamma = target_length / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            AL += d - t_minus_1 / gamma
            tau = t_minus_1 + 1

            if d >= source_length:
                break
        return AL / tau if tau > 0 else None

    def __call__(self, instances: Sequence[Instance]) -> float:
        scores = []
        for ins in instances:
            delays = getattr(ins, "emission_ca" if self.computation_aware else "emission_cu", None)
            if delays is None or len(delays) == 0:
                logger.warning(f"Instance {ins.index} has no emission timestamps. Skipped.")
                continue
            score = self.compute(ins)
            if score is not None:
                scores.append(score)
        return mean(scores) if scores else float("nan")


class LAALScorer(ALScorer):
    """Length-Adaptive Average Lagging (LAAL) scorer."""

    def compute(self, ins: Instance) -> Optional[float]:
        delays, source_length, target_length = self.get_delays_lengths(ins)
        if source_length is None or source_length <= 0:
            return None
        if delays[0] > source_length:
            return delays[0]

        LAAL = 0.0
        gamma = max(len(delays), target_length) / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            LAAL += d - t_minus_1 / gamma
            tau = t_minus_1 + 1

            if d >= source_length:
                break
        return LAAL / tau if tau > 0 else None


class APScorer:
    """Average Proportion (AP) scorer."""

    def __init__(self, computation_aware: bool = False):
        self.computation_aware = computation_aware

    def __call__(self, instances: Sequence[Instance]) -> float:
        scores = []
        for ins in instances:
            delays = getattr(ins, "emission_ca" if self.computation_aware else "emission_cu", None)
            if delays is None or len(delays) == 0:
                logger.warning(f"Instance {ins.index} has no emission timestamps. Skipped.")
                continue
            source_length = getattr(ins, "source_length", None)
            tgt_len = len(delays) if ins.reference is None else ins.reference_length
            if source_length is None or source_length == 0 or tgt_len == 0:
                continue
            scores.append(sum(delays) / (source_length * tgt_len))
        return mean(scores) if scores else float("nan")


class DALScorer:
    """Differentiable Average Lagging (DAL) scorer."""

    def __init__(self, computation_aware: bool = False):
        self.computation_aware = computation_aware

    def __call__(self, instances: Sequence[Instance]) -> float:
        scores = []
        for ins in instances:
            delays = getattr(ins, "emission_ca" if self.computation_aware else "emission_cu", None)
            if delays is None or len(delays) == 0:
                logger.warning(f"Instance {ins.index} has no emission timestamps. Skipped.")
                continue
            source_length = getattr(ins, "source_length", None)
            if source_length is None or source_length == 0:
                continue

            DAL = 0.0
            target_length = len(delays)
            gamma = target_length / source_length
            g_prime_last = 0.0
            for i_minus_1, g in enumerate(delays):
                if i_minus_1 + 1 == 1:
                    g_prime = g
                else:
                    g_prime = max(g, g_prime_last + 1 / gamma)

                DAL += g_prime - i_minus_1 / gamma
                g_prime_last = g_prime

            DAL /= target_length
            scores.append(DAL)
        return mean(scores) if scores else float("nan")


class YAALScorer:
    r"""
    Yet Another Average Lagging (YAAL) scorer.

    From "Better Late Than Never: Evaluation of Latency Metrics for
    Simultaneous Speech-to-Text Translation" (https://arxiv.org/abs/2509.17349).
    """

    def __init__(self, computation_aware: bool = False, is_longform: bool = False):
        self.computation_aware = computation_aware
        self.is_longform = is_longform

    def get_delays_lengths(self, ins: Instance):
        """
        Extract emission timestamps, source length, and target length from an instance.

        Returns:
            Tuple of (emission_timestamps, source_length, target_length).
        """
        timestamp_type = "emission_ca" if self.computation_aware else "emission_cu"
        delays = getattr(ins, timestamp_type, None)
        assert delays
        tgt_len = len(delays) if ins.reference is None else ins.reference_length
        src_len = getattr(ins, "source_length", None)
        return delays, src_len, tgt_len

    def compute(self, ins: Instance) -> Optional[float]:
        """
        Compute YAAL latency for one instance.

        Returns:
            YAAL score, or None if the instance should be skipped.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)
        assert source_length is not None, "Source length must be provided for YAAL scoring."
        is_longform = (ins.longform is not None and ins.longform) or self.is_longform
        recording_end = ins.time_to_recording_end if ins.time_to_recording_end is not None else float("inf")

        if (delays[0] >= source_length and not is_longform) or (delays[0] >= recording_end):
            return None

        assert source_length > 0, "Source length must be greater than 0."

        yaal = 0.0
        gamma = max(len(delays), target_length) / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            if (d >= source_length and not is_longform) or (d >= recording_end):
                break
            yaal += d - t_minus_1 / gamma
            tau = t_minus_1 + 1

        return yaal / tau if tau > 0 else None

    def __call__(self, instances: Sequence[Instance]) -> float:
        scores = []
        timestamp_type = "emission_ca" if self.computation_aware else "emission_cu"
        for ins in instances:
            delays = getattr(ins, timestamp_type, None)
            if delays is None or len(delays) == 0:
                logger.warning(f"Instance {ins.index} has no emission timestamps. Skipped.")
                continue
            score = self.compute(ins)
            if score is not None:
                scores.append(score)

        return mean(scores) if scores else float("nan")




class ShortformDegeneracyScorer:
    r"""
    Shortform-only degeneracy diagnostics.

    Computes three percent-scale (0–100) metrics and a boolean flag:

    * **SWF** – Simultaneous Words Fraction: the fraction of output words
      emitted before the end-of-segment signal, expressed as a percentage.
      Corpus-level micro-average: ``100 × Σ #{d < src} / Σ |delays|``.

    * **EFSW** – Expected Fraction of Simultaneous Words: the fraction
      of the segment that falls before the average delay (YAAL), expressed
      as a percentage.
      ``100 × Σ max(0, src − YAAL_i) / Σ src``.

    * **DSPTV** – Degenerate Simultaneous Policy Test Value:
      the signed difference ``SWF − EFSW``.

    * **Degenerate Policy** (boolean 0/1): 1 when ``|DSPTV| > 20``.
    """

    def __call__(self, instances: Sequence[Instance]) -> Dict[str, float]:
        yaal_scorer = YAALScorer(is_longform=False)
        total_simultaneous = 0
        total_words = 0
        efsw_num = 0.0
        efsw_den = 0.0

        for ins in instances:
            delays = getattr(ins, "emission_cu", None)
            source_length = getattr(ins, "source_length", None)
            assert delays is not None, f"Instance {ins.index} has no emission_cu timestamps."
            assert source_length is not None, f"Instance {ins.index} has no source_length."

            # SWF components
            total_simultaneous += sum(1 for d in delays if d < source_length)
            total_words += len(delays)

            # EFSW components
            yaal_i = yaal_scorer.compute(ins)
            if yaal_i is not None and not math.isnan(yaal_i):
                efsw_num += max(0.0, source_length - yaal_i)
                efsw_den += source_length

        if total_words == 0:
            nan = float("nan")
            return {"swf": nan, "efsw": nan, "dsptv": nan, "degenerate_policy": nan}

        swf = 100.0 * total_simultaneous / total_words
        efsw = 100.0 * efsw_num / efsw_den if efsw_den > 0 else float("nan")
        dsptv = (efsw - swf) if not math.isnan(efsw) else float("nan")
        degenerate = 1.0 if (not math.isnan(dsptv) and abs(dsptv) > 20) else 0.0

        return {
            "swf": swf,
            "efsw": efsw,
            "dsptv": dsptv,
            "degenerate_policy": degenerate,
        }


def evaluate_instances(
    resegmented_instances: List[Instance],
    *,
    compute_quality: bool = True,
    compute_latency: bool = True,
    is_longform: bool = True,
    bleu_tokenizer: str = "13a",
    all_have_emission_ca: bool = False,
    fix_emission_ca_flag: bool = False,
    compute_comet: bool = False,
    comet_model: str = "Unbabel/wmt22-comet-da",
    source_sentences: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate resegmented instances with the selected metrics.

    Args:
        resegmented_instances: List of resegmented instances.
        compute_quality: Whether to compute quality metrics (BLEU, chrF, COMET).
        compute_latency: Whether to compute latency metrics (YAAL).
        is_longform: Whether to use long-form YAAL (True for resegmented long-form
            outputs, False for short-form segment-level systems).
        bleu_tokenizer: Tokenizer for SacreBLEU.
        all_have_emission_ca: Whether all instances have computation-aware timestamps.
        fix_emission_ca_flag: Whether computation-aware timestamps were already fixed.
        compute_comet: Whether to compute COMET (requires source_sentences).
        comet_model: COMET model name.
        source_sentences: Source sentences for COMET scoring.

    Returns:
        Dictionary of metric names to scores.
    """
    scores: Dict[str, float] = {}

    if compute_quality:
        scores["bleu"] = SacreBLEUScorer(bleu_tokenizer)(resegmented_instances)
        scores["chrf"] = ChrFScorer()(resegmented_instances)
        if compute_comet and source_sentences is not None:
            scores["comet"] = COMETScorer(comet_model)(resegmented_instances, source_sentences)

    if compute_latency:
        ca_unaware = YAALScorer(is_longform=is_longform)(resegmented_instances)
        scores["yaal"] = ca_unaware

        # Add AL/LAAL/AP/DAL (non-CA)
        prefix = "long_" if is_longform else ""
        try:
            scores[f"{prefix}al"] = ALScorer(computation_aware=False)(resegmented_instances)
            scores[f"{prefix}laal"] = LAALScorer(computation_aware=False)(resegmented_instances)
            scores[f"{prefix}ap"] = APScorer(computation_aware=False)(resegmented_instances)
            scores[f"{prefix}dal"] = DALScorer(computation_aware=False)(resegmented_instances)
        except Exception as e:
            logger.warning(f"Failed to compute AL/LAAL/AP/DAL: {e}")

        if all_have_emission_ca:
            ca_aware = YAALScorer(computation_aware=True, is_longform=is_longform)(resegmented_instances)
            scores["ca_yaal"] = ca_aware

            # CA variants for other latency scorers
            try:
                scores[f"ca_{prefix}al"] = ALScorer(computation_aware=True)(resegmented_instances)
                scores[f"ca_{prefix}laal"] = LAALScorer(computation_aware=True)(resegmented_instances)
                scores[f"ca_{prefix}ap"] = APScorer(computation_aware=True)(resegmented_instances)
                scores[f"ca_{prefix}dal"] = DALScorer(computation_aware=True)(resegmented_instances)
            except Exception as e:
                logger.warning(f"Failed to compute CA variants for latency scorers: {e}")

            if not fix_emission_ca_flag and ca_aware - ca_unaware > 2000:
                logger.warning(
                    f"CA-YAAL ({ca_aware:.1f}) is much higher than YAAL ({ca_unaware:.1f}). "
                    f"Consider using --fix_simuleval_emission_ca."
                )
        else:
            scores["ca_yaal"] = float("nan")
            # mark CA variants as NaN
            scores[f"ca_{prefix}al"] = float("nan")
            scores[f"ca_{prefix}laal"] = float("nan")
            scores[f"ca_{prefix}ap"] = float("nan")
            scores[f"ca_{prefix}dal"] = float("nan")
            logger.warning(
                "Not all instances have computation-aware timestamps. CA latency metrics set to NaN."
            )

    # Shortform-only degeneracy diagnostics
    if compute_latency and not is_longform:
        degeneracy = ShortformDegeneracyScorer()(resegmented_instances)
        scores.update(degeneracy)

    return scores
