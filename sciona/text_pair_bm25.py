"""Frozen training-only BM25 statistics for the pending complete pair recipe."""
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
import re
import unicodedata

from sciona.bm25_scoring import bm25_score


def _pairs(pairs):
    # Accept already normalized outputs without applying spelling/expansion twice.
    if type(pairs) is not list or not pairs:
        raise ValueError('Nonempty normalized pair list required')
    for pair in pairs:
        if type(pair) not in (list, tuple) or len(pair) != 2:
            raise ValueError('Two normalized text fields required')
        for text in pair:
            if (type(text) is not str or not 0 < len(text) <= 512
                    or re.fullmatch(r'\w+(?: \w+)*', text) is None
                    or unicodedata.normalize('NFKC', text).casefold() != text):
                raise ValueError('Canonical normalized text required')
    return pairs


@dataclass(frozen=True, init=False)
class PairBM25:
    """Unique training texts define documents; repeated pair endpoints count once.

    Query terms are distinct, unseen terms have training document frequency zero,
    and held-out document lengths include every token. Pair score is the mean of
    both directional BM25 scores. It is a nonnegative feature, not a probability.
    Fitting never takes calibration/query populations. Transform never refits.
    """
    document_count: int
    average_length: float
    document_frequencies: object
    k1: float
    b: float

    def __init__(self, training_pairs, *, k1, b):
        docs = sorted({text for pair in _pairs(training_pairs) for text in pair})
        frequencies = Counter(token for text in docs for token in set(text.split()))
        average = sum(len(text.split()) for text in docs) / len(docs)
        bm25_score([], [], 0, len(docs), average, k1, b)
        for key, value in dict(document_count=len(docs), average_length=average,
                               document_frequencies=MappingProxyType(dict(frequencies)),
                               k1=float(k1), b=float(b)).items():
            object.__setattr__(self, key, value)

    def transform(self, normalized_pairs):
        result = []
        for left, right in _pairs(normalized_pairs):
            directional = []
            for query, document in ((left, right), (right, left)):
                terms = sorted(set(query.split()))
                counts = Counter(document.split())
                directional.append(bm25_score(
                    [counts[t] for t in terms], [self.document_frequencies.get(t, 0) for t in terms],
                    sum(counts.values()), self.document_count, self.average_length, self.k1, self.b))
            result.append(directional[0] / 2 + directional[1] / 2)
        return result
