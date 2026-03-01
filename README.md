# OmniSTEval

A tool for evaluating simultaneous speech/text translation systems — both **shortform** (segment-level) and **longform** (document-level with resegmentation).

For longform systems, OmniSTEval re-segments the translation outputs to match reference segmentation, enabling segment-level quality (BLEU, chrF, COMET) and latency (YAAL, LongYAAL, etc.) evaluation.

Implements YAAL, LongYAAL and the SoftSegmenter alignment algorithm from [*Better Late Than Never: Evaluation of Latency Metrics for Simultaneous Speech-to-Text Translation*](https://arxiv.org/abs/2509.17349).


## How It Works

### Shortform evaluation

For the shortform case (i.e., when the input speech is segmented into individual segments following a sentence boundary), OmniSTEval directly evaluates quality and latency metrics on the per-segment outputs without any resegmentation.

In addition to the standard quality metrics (BLEU, chrF, COMET) and latency metrics (YAAL, AL, LAAL, AP, DAL), shortform evaluation computes a set of **degeneracy diagnostics** that help detect whether a system has learned a degenerate simultaneous policy — see [Shortform Degeneracy Diagnostics](#shortform-degeneracy-diagnostics) below.

### Longform evaluation (with resegmentation)

Simultaneous speech translation systems (e.g., those evaluated via [SimulEval](https://github.com/facebookresearch/SimulEval)) produce a single long-form output per audio recording. However, most evaluation metrics (like BLEU and YAAL) are designed for segmented inputs. OmniSTEval takes the reference speech segmentation and aligns the hypothesis words (including their emission timestamps) to it, effectively re-segmenting the hypothesis according to the reference segments.

The pipeline consists of the following steps:

1. **Tokenization** — Reference and hypothesis words are optionally tokenized using the [Moses tokenizer](https://github.com/luismsgomes/mosestokenizer). For Chinese/Japanese (or when `--lang` is not set), tokenization is skipped and character-level units are used instead.
2. **Alignment** — A dynamic programming algorithm (similar to Needleman-Wunsch / DTW, but without gap penalties) aligns hypothesis words to reference words. The alignment maximizes a Jaccard-based character-set similarity score at word level, or exact match at character level. Punctuation is prevented from aligning with non-punctuation tokens, which helps to mitigate segmentation errors around sentence boundaries.
3. **Re-segmentation** — Aligned hypothesis words are grouped by their assigned reference segment IDs, producing one hypothesis segment per reference segment.
4. **Evaluation** — Each re-segmented instance is scored with the enabled metrics:
   - **Quality**: BLEU and chrF via [SacreBLEU](https://github.com/mjpost/sacrebleu), optionally [COMET](https://github.com/Unbabel/COMET)
   - **Latency**: LongYAAL (Yet Another Average Lagging) — both computation-aware and computation-unaware variants, plus LongAL, LongLAAL, LongAP, and LongDAL (adapted from [SimulEval](https://github.com/facebookresearch/SimulEval))


## Shortform Degeneracy Diagnostics

When running `omnisteval shortform`, three additional percent-scale metrics and a boolean flag are computed to help identify systems that have learned a **degenerate simultaneous policy** — i.e., a policy that appears simultaneous (outputs arrive before end-of-segment) without actually being driven by source content.

| Metric | Display name | Description |
|---|---|---|
| `swf` | Simultaneous Words Fraction (%) | Fraction of output words emitted before the end-of-segment signal. Corpus micro-average: `100 × Σ \|{d < src_len}\| / Σ \|delays\|`. |
| `efsw` | Expected Simul. Words Fraction (%) | Expected fraction of the segment duration that precedes the average latency per segment. `100 × Σ max(0, src_len − YAAL_i) / Σ src_len`. |
| `dsptv` | Degeneracy Test Value | Signed difference `EFSW − SWF`. A large positive value means that the system is emitting a few words with a significantly lower latency while translating the rest of the segment after the end-of-segment signal. |
| `degenerate_policy` | Likely Degenerate Simultaneous Policy | `YES` when `\|DSPTV\| > 20`, `NO` otherwise. When `YES`, an additional warning banner is also printed. |

**Interpretation:**
- Both SWF and EFSW measure the "simultaneous‑ness" of the system, but from different angles. SWF is purely empirical; EFSW is derived from YAAL.
- A well-behaved simultaneous system should have SWF ≈ EFSW, giving a DSPTV near 0.
- A large `|DSPTV|` (> 20 pp) suggests the system's emission pattern is inconsistent with its latency profile — a common signature of degenerate strategies such as outputting all words at the very start or end of each segment.

**Example report output (degenerate case):**

```
================================================================
OmniSTEval v0.1.1  |  Shortform evaluation
================================================================

Settings
----------------------------------------------------------------
  Hypothesis        instances.log
  Reference         references.txt
  BLEU tokenizer    13a
  Char-level        no
  Fix CA emissions  no
  Metrics           quality, latency
  Version           0.1.1

Scores
----------------------------------------------------------------
  BLEU                                   18.2271
  chrF                                   44.5324
  YAAL (CU)                              1135.6097
  AL (CU)                                1803.9192
  LAAL (CU)                              1857.7128
  AP (CU)                                0.7948
  DAL (CU)                               3532.4812
  YAAL (CA)                              1272.7485
  AL (CA)                                2021.1781
  LAAL (CA)                              2071.7031
  AP (CA)                                0.8903
  DAL (CA)                               3883.0303
  Simultaneous Words Fraction (%)        32.9060
  Expected Simul. Words Fraction (%)     81.1100
  Degeneracy Test Value                  48.2040
  Likely Degenerate Simultaneous Policy  YES

  *** Likely Degenerate Simultaneous Policy ***

================================================================
```


## Installation

```bash
pip install OmniSTEval
```

Or install from source:

```bash
git clone https://github.com/pe-trik/omnisteval.git
cd omnisteval
pip install -e .
```

For COMET scoring support:

```bash
pip install OmniSTEval[comet]
```

### Requirements

- Python 3.8+
- `mosestokenizer>=1.2.1`
- `PyYAML>=6.0.3`
- `sacrebleu>=2.5.1`
- (Optional) `unbabel-comet` — for COMET scoring


## Usage

OmniSTEval provides two subcommands: **`shortform`** and **`longform`**.

### Shortform evaluation

Evaluate a segment-level (shortform) SimulEval output directly:

```bash
omnisteval shortform \
  --hypothesis_file instances.log \
  --ref_sentences_file reference_sentences.txt \
  --bleu_tokenizer 13a \
  --output_folder evaluation_output
```

### Longform evaluation with speech resegmentation

Re-segment a long-form hypothesis to match the reference speech segmentation, then evaluate:

```bash
omnisteval longform \
  --speech_segmentation ref_segments.yaml \
  --ref_sentences_file reference_sentences.txt \
  --hypothesis_file simuleval_instance_file.log \
  --lang en \
  --bleu_tokenizer 13a \
  --output_folder segmentation_output
```

### Longform evaluation with text resegmentation

Re-segment based on text-level document/segment IDs (no latency metrics):

```bash
omnisteval longform \
  --text_segmentation text_segmentation.txt \
  --ref_sentences_file reference_sentences.txt \
  --hypothesis_file hypotheses.txt \
  --hypothesis_format text \
  --lang en \
  --output_folder segmentation_output
```

### Evaluate a pre-resegmented log

If you already have a resegmented JSONL file (e.g., from a previous longform run), you can evaluate it directly:

```bash
omnisteval longform \
  --resegmented_hypothesis instances.resegmented.jsonl \
  --bleu_tokenizer 13a \
  --output_folder evaluation_output
```

### With COMET scoring

```bash
omnisteval longform \
  --speech_segmentation ref_segments.yaml \
  --ref_sentences_file reference_sentences.txt \
  --hypothesis_file simuleval_instance_file.log \
  --source_sentences_file source_sentences.txt \
  --comet \
  --lang en \
  --output_folder segmentation_output
```

### Custom emission timestamp field names

If your JSONL hypothesis uses different keys for emission timestamps:

```bash
omnisteval longform \
  --speech_segmentation ref_segments.yaml \
  --ref_sentences_file reference_sentences.txt \
  --hypothesis_file hypothesis.log \
  --emission_cu_key my_delays \
  --emission_ca_key my_elapsed \
  --lang en \
  --output_folder segmentation_output
```

### Arguments

#### Common arguments (both subcommands)

| Argument | Required | Default | Description |
|---|---|---|---|
| `--output_folder` | No | — | Directory where output files will be written. When omitted, the evaluation report is printed to stdout only — no files are saved. |
| `--char_level` | No | `False` | Use character-level alignment and scoring instead of word-level. |
| `--no-quality` | No | `False` | Disable quality metrics (BLEU, chrF, COMET). |
| `--no-latency` | No | `False` | Disable latency metrics (YAAL). Automatically set for text-only hypotheses. |
| `--comet` | No | `False` | Enable COMET scoring. Requires `--source_sentences_file` and `unbabel-comet`. |
| `--comet_model` | No | `Unbabel/wmt22-comet-da` | COMET model name. |
| `--bleu_tokenizer` | No | `13a` | Tokenizer for SacreBLEU (e.g., `13a`, `intl`, `ja-mecab`, `zh`). |
| `--source_sentences_file` | No | — | Path to source sentences file (one per segment, for COMET scoring). |
| `--emission_cu_key` | No | `delays` | JSON key for computation-unaware emission timestamps in JSONL hypothesis. |
| `--emission_ca_key` | No | `elapsed` | JSON key for computation-aware emission timestamps in JSONL hypothesis. |
| `--fix_simuleval_emission_ca` | No | `False` | Fix computation-aware emission timestamps for CA-YAAL. |

#### `shortform` arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--hypothesis_file` | Yes | — | Path to the JSONL hypothesis file (one JSON per line with `prediction`, delays, etc.). |
| `--ref_sentences_file` | Yes | — | Path to the reference sentences file (one sentence per line). |

#### `longform` arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--speech_segmentation` | One of these | — | Path to a YAML/JSON speech segmentation file. Mutually exclusive with `--text_segmentation` and `--resegmented_hypothesis`. |
| `--text_segmentation` | or one of these | — | Path to a text segmentation file (`docid=DOC_ID,segid=SEG_ID` format). Mutually exclusive with `--speech_segmentation` and `--resegmented_hypothesis`. |
| `--resegmented_hypothesis` | or this | — | Path to a pre-resegmented JSONL file. Mutually exclusive with segmentation inputs. |
| `--ref_sentences_file` | For reseg. | — | Path to reference sentences file. Required for resegmentation mode. |
| `--hypothesis_file` | For reseg. | — | Path to the hypothesis file. Required for resegmentation mode. |
| `--hypothesis_format` | No | `jsonl` | Format of the hypothesis file: `jsonl` (SimulEval output) or `text`. |
| `--lang` | No | `None` | Language code for Moses tokenizer (e.g., `en`, `de`). |
| `--offset_delays` | No | `False` | Offset delays relative to the first segment of each recording. |


## Input Formats

### Text Segmentation

A plain-text file with one entry per line in the format `docid=DOC_ID,segid=SEG_ID`, where DOC_ID and SEG_ID are 0-based integers. One line per reference sentence. The number of unique document IDs must equal the number of hypothesis lines.

```
docid=0,segid=0
docid=0,segid=1
docid=0,segid=2
docid=1,segid=0
docid=1,segid=1
```

- `docid` — 0-based document index (maps to hypothesis line number)
- `segid` — 0-based segment index within the document

### Speech Segmentation (YAML/JSON)

A list of segments, each with the following fields:

```yaml
- {wav: recording.wav, offset: 2.433, duration: 9.05, speaker_id: spk1}
- {wav: recording.wav, offset: 15.003, duration: 9.675, speaker_id: spk1}
```

- `wav` — audio filename (used to group segments by recording)
- `offset` — segment start time in seconds
- `duration` — segment duration in seconds
- `speaker_id` — (optional) speaker identifier

### Reference Sentences

One sentence per line, aligned 1:1 with the segmentation entries:

```
Hello, this is Elena and I will present our work.
We will discuss what lexical borrowing is.
```

### Hypothesis File

**JSONL format** (`--hypothesis_format jsonl`, default) — One JSON object per line (SimulEval output format):

```json
{"source": "recording.wav", "prediction": "Hello this is Elena ...", "delays": [4067.0, 4067.0, ...], "elapsed": [4100.0, 4200.0, ...], "source_length": 220000}
```

- `source` — audio filename as a string (e.g., `"recording.wav"`), or an array with the recording name as the first element (e.g., `["recording.wav"]`) for backward compatibility with SimulEval logs
- `prediction` — the full hypothesis text
- `delays` — per-token computation-unaware emission timestamps (in ms); length must match the number of words (or characters if `--char_level`) in `prediction`
- `elapsed` — (optional) per-token computation-aware emission timestamps (in ms)
- `source_length` — (optional, but highly recommended) total recording length in ms

The key names for `delays` and `elapsed` can be customized with `--emission_cu_key` and `--emission_ca_key`.

**Text format** (`--hypothesis_format text`) — One hypothesis per line, matched by order to recordings in the segmentation file. Latency metrics are not available in this mode.

```
Hello this is Elena and I will present our work.
We will discuss what lexical borrowing is.
```

### Source Sentences (for COMET)

One source-language sentence per line, aligned 1:1 with the segmentation entries (same count as reference sentences):

```
Hola, soy Elena y presentaré nuestro trabajo.
Discutiremos qué es el préstamo léxico.
```


## Output

Each run prints a human-readable evaluation report to stdout:

```
================================================================
OmniSTEval v0.1.1  |  Longform evaluation (resegmentation)
================================================================

Settings
----------------------------------------------------------------
  Hypothesis         instances.log
  Hypothesis format  jsonl
  Reference          references.txt
  Segmentation       ref_segments.yaml
  Seg. type          speech
  Language           de
  BLEU tokenizer     13a
  Char-level         no
  Offset delays      no
  Fix CA emissions   yes
  Metrics            quality, latency
  Version            0.1.1

Scores
----------------------------------------------------------------
  BLEU           22.9418
  chrF           52.1659
  LongYAAL (CU)  2969.2223
  LongAL (CU)    3000.5570
  LongLAAL (CU)  3102.8672
  LongAP (CU)    1.0506
  LongDAL (CU)   4130.1372
  LongYAAL (CA)  5905.4040
  LongAL (CA)    6100.6042
  LongLAAL (CA)  6159.5048
  LongAP (CA)    1.7176
  LongDAL (CA)   7409.7906

================================================================
```

If `--output_folder` is provided, three files are also written:

### `evaluation_report.txt`

The same human-readable report shown on stdout, saved to a file for archival. Contains the version, all settings used, and the metric scores — sufficient to reproduce the reported values.

### `scores.tsv`

A tab-separated file with one score per line — useful for scripting:

```
metric	value
bleu	22.9418
chrf	52.1659
yaal	2969.2223
long_al	3000.5570
long_laal	3102.8672
long_ap	1.0506
long_dal	4130.1372
ca_yaal	5905.4040
ca_long_al	6100.6042
ca_long_laal	6159.5048
ca_long_ap	1.7176
ca_long_dal	7409.7906
```

### `instances.resegmented.jsonl` *(longform resegmentation only)*

A JSONL file with one re-segmented instance per line (one per reference segment):

```json
{"index": 0, "docid": 0, "segid": 0, "prediction": "Hello , this is Elena and I will present our work .", "reference": "Hello, this is Elena and I will present our work.", "source_length": 9050.0, "emission_cu": [4067.0, 4067.0, ...], "emission_ca": [4100.0, 4200.0, ...], "time_to_recording_end": 220000.0}
{"index": 1, "docid": 0, "segid": 1, ...}
```

Each entry contains the hypothesis words assigned to that segment, with emission timestamps offset relative to the segment start. In text-only mode, `emission_cu`, `emission_ca`, and `time_to_recording_end` fields are omitted.

### Score columns

#### Quality (both modes)

- `BLEU` — corpus-level SacreBLEU score
- `chrF` — corpus-level chrF score
- `COMET` — COMET score (only if `--comet` is enabled)

#### Latency (shortform — `is_longform=False`)

- `YAAL (CU)` / `YAAL (CA)` — Yet Another Average Lagging (computation-unaware / computation-aware)
- `AL (CU)` / `AL (CA)` — Average Lagging (adapted from SimulEval)
- `LAAL (CU)` / `LAAL (CA)` — Length-Adaptive Average Lagging (adapted from SimulEval)
- `AP (CU)` / `AP (CA)` — Average Proportion (adapted from SimulEval)
- `DAL (CU)` / `DAL (CA)` — Differentiable Average Lagging (adapted from SimulEval)

#### Latency (longform — `is_longform=True`)

Same metrics as above, prefixed with `Long`: `LongYAAL (CU)`, `LongAL (CU)`, `LongLAAL (CU)`, `LongAP (CU)`, `LongDAL (CU)`, and their `(CA)` counterparts.

#### Shortform degeneracy diagnostics (shortform only)

- `Simultaneous Words Fraction (%)` — see [Shortform Degeneracy Diagnostics](#shortform-degeneracy-diagnostics)
- `Expected Simul. Words Fraction (%)` — see above
- `Degeneracy Test Value` — signed difference EFSW − SWF
- `Likely Degenerate Simultaneous Policy` — `YES` / `NO`


## Examples

See the [examples/](examples/) directory for sample input files and expected output:

- **Shortform evaluation (with degeneracy diagnostics)**: `examples/short_form_degenerate_policy/`
- **Speech resegmentation**: `examples/speech_resegmentation_example/`
- **Text resegmentation**: `examples/text_resegmentation_example/`

Run an example with:

```bash
cd examples/speech_resegmentation_example
bash resegment.sh
```

Or evaluate a shortform system:

```bash
cd examples/short_form_degenerate_policy_example
bash evaluate.sh
```


## Citation

If you use this tool in your research, please cite it as follows:

```bibtex
@article{polak2025better,
  title={Better Late Than Never: Evaluation of Latency Metrics for Simultaneous Speech-to-Text Translation},
  author={Pol{\'a}k, Peter and Papi, Sara and Bentivogli, Luisa and Bojar, Ond{\v{r}}ej},
  journal={arXiv preprint arXiv:2509.17349},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.