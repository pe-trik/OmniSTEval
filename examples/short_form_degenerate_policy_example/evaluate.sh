#!/bin/bash

omnisteval shortform \
  --ref_sentences_file references.txt \
  --hypothesis_file instances.log \
  --bleu_tokenizer 13a \
  --output_folder evaluation_output 