"""Numeric BM25 scoring, independent of tokenization and corpus selection."""
import math


def bm25_score(term_frequencies, document_frequencies, document_length,
               document_count, average_length, k1, b):
    """Score aligned, distinct query terms against one document.

    Uses positive log1p IDF and tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl)).
    Query terms have unit weight; callers choose the distinct terms and corpus.
    Counts are exact Python integers bounded by 2**53; document frequencies
    may be zero for unseen terms. Empty queries and zero-length documents score
    zero. No text, fitting, or mutable state is involved.

    Positive IDF convention: Apache Lucene BM25Similarity (10.2.2),
    https://lucene.apache.org/core/10_2_2/core/org/apache/lucene/search/similarities/BM25Similarity.html
    The explicit (k1+1) scale and exact lengths are this contract's convention;
    this does not promise bitwise agreement with Lucene's encoded norms.
    """
    def count(value):
        return type(value) is int and 0 <= value <= 2**53

    if type(term_frequencies) not in (list, tuple) or type(document_frequencies) not in (list, tuple):
        raise ValueError('Aligned count sequences required')
    if len(term_frequencies) != len(document_frequencies):
        raise ValueError('Aligned count sequences required')
    if not count(document_length) or not count(document_count) or document_count == 0:
        raise ValueError('Valid document length and positive corpus size required')
    if any(not count(t) for t in term_frequencies) or sum(term_frequencies) > document_length:
        raise ValueError('Distinct-term counts must fit document length')
    if any(not count(d) or d > document_count for d in document_frequencies):
        raise ValueError('Document frequencies must fit corpus size')
    if any(type(x) not in (int, float)
           or (type(x) is int and abs(x) > 2**53)
           or not math.isfinite(x) for x in (average_length, k1, b)):
        raise ValueError('Finite numeric controls required')
    if not 0 < average_length <= 2**53 or not 0 < k1 <= 2**53 or not 0 <= b <= 1:
        raise ValueError('Positive average length/k1 and b in [0,1] required')
    try:
        length_factor = k1 * (1 - b + b * document_length / average_length)
        score = math.fsum(
            math.log1p((document_count - df + 0.5) / (df + 0.5))
            * (tf * (k1 + 1) / (tf + length_factor))
            for tf, df in zip(term_frequencies, document_frequencies) if tf
        )
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError('BM25 arithmetic outside finite range') from exc
    if not math.isfinite(length_factor) or not math.isfinite(score):
        raise ValueError('BM25 arithmetic outside finite range')
    return score
