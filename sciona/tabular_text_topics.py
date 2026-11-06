"""Training-only count vocabulary and LDA topics for explicit text fields."""
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from threadpoolctl import threadpool_limits

from sciona.atoms.ml.sklearn.decomposition.lda_shell.atoms import lda_normalize_document_topics


def _documents(documents):
    if type(documents) is not list or not documents or any(type(text) is not str for text in documents):
        raise ValueError('Nonempty list of text strings required')
    return documents


@dataclass(frozen=True)
class TextTopics:
    """Execution-local fitted vocabulary and topic model; queries never refit."""
    vectorizer: object
    model: object

    def transform(self, documents):
        counts = self.vectorizer.transform(_documents(documents))
        with threadpool_limits(limits=1):
            topics = self.model.transform(counts, normalize=False)
        result = lda_normalize_document_topics(topics)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite topic features')
        return result


def fit_text_topics(documents, *, topics, max_features, iterations, lowercase, seed):
    """Fit batch LDA on whitespace token counts, without labels.

Case folding is explicit. Punctuation remains part of each token; no stemming,
stop list or language inference occurs. Empty/all-unseen query documents yield
the symmetric topic prior. An entirely empty training vocabulary is rejected.
Topic indices have no intrinsic semantic labels. Vocabulary caps and iteration
counts are caller policy; large-corpus resource use is not qualified here.
"""
    documents = _documents(documents)
    for value in (topics, max_features, iterations):
        if type(value) is not int or value < 1:
            raise ValueError('Positive topic, vocabulary and iteration counts required')
    if type(lowercase) is not bool or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Explicit case policy and valid seed required')
    vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=lowercase,
        max_features=max_features, dtype=np.int64)
    counts = vectorizer.fit_transform(documents)
    model = LatentDirichletAllocation(n_components=topics, learning_method='batch',
        max_iter=iterations, random_state=seed, n_jobs=1, evaluate_every=-1)
    with threadpool_limits(limits=1):
        model.fit(counts)
    state = TextTopics(vectorizer, model)
    return state, state.transform(documents)
