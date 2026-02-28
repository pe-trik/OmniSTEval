#!/bin/bash

omnisteval longform \
  --text_segmentation text_segmentation.txt \
  --ref_sentences_file references.txt \
  --hypothesis_file hypotheses.txt \
  --hypothesis_format text \
  --lang en \
  --bleu_tokenizer 13a \
  --output_folder segmentation_output 