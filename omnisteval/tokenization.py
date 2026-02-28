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
Tokenization utilities for SoftSegmenter.
"""

import unicodedata
from typing import List, Optional

from .alignment import Word


def unicode_normalize(text: str) -> str:
    """Normalize Unicode text to NFKC form."""
    return unicodedata.normalize("NFKC", text)


def tokenize_words(words: List[List[Word]], lang: Optional[str]) -> List[List[Word]]:
    """
    Tokenize words using Moses tokenizer.

    For Chinese/Japanese (or when lang is None), no tokenization is applied.
    Otherwise, the Moses tokenizer splits words into subtokens.

    Args:
        words: List of lists of Word objects grouped by recording.
        lang: Language code for Moses tokenizer (e.g., 'en', 'de').
            Use None, 'zh', or 'ja' to skip tokenization.

    Returns:
        Tokenized words with subtokens marked (main=False for non-first subtokens).
    """
    if lang is None or lang in ("zh", "ja"):
        tokenizer = lambda x: [x]
    else:
        from mosestokenizer import MosesTokenizer
        tokenizer = MosesTokenizer(lang=lang, no_escape=True)

    tokenized_words: List[List[Word]] = []
    for recording_words in words:
        tokenized_recording: List[Word] = []
        for word in recording_words:
            text = unicode_normalize(word.text).lower()
            tokens = tokenizer(text)
            main = True
            for token in tokens:
                tokenized_recording.append(
                    Word(
                        token,
                        emission_cu=word.emission_cu,
                        emission_ca=word.emission_ca,
                        seq_id=word.seq_id,
                        main=main,
                        original=word.text if main else None,
                        recording_length=word.recording_length,
                    )
                )
                main = False
        tokenized_words.append(tokenized_recording)
    return tokenized_words
