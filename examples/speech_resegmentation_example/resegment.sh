#!/bin/bash

omnisteval longform \
  --speech_segmentation ref_segments.yaml \
  --ref_sentences_file references.txt \
  --hypothesis_file instances.log \
  --hypothesis_format jsonl \
  --lang de \
  --bleu_tokenizer 13a \
  --output_folder segmentation_output \
  --fix_simuleval_emission_ca