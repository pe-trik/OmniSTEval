#!/bin/bash

omnisteval longform \
  --speech_segmentation ref_segments.yaml \
  --ref_sentences_file references.txt \
  --hypothesis_file simulstream_log.jsonl \
  --simulstream_config_file cfg.yaml \
  --hypothesis_format simulstream \
  --lang de \
  --bleu_tokenizer 13a \
  --output_folder segmentation_output
