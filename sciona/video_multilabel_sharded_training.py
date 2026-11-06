"""Full configured training and prediction over private indexed feature shards.

Frames and sparse source mappings are loaded per batch. Training labels, the
training CSR matrix, model state, shuffle indices and returned scores remain
resident; this is bounded frame loading, not a constant-total-memory promise.
"""
from dataclasses import dataclass
import warnings
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sciona.video_multilabel_network import pad_sequences
from sciona.video_multilabel_reusable_network import ReusableVideoHead
from sciona.video_multilabel_thresholds import label_matrix, score_matrix
from sciona.atoms.ml.calibration.empirical_f1 import fit_empirical_f1_thresholds, apply_column_thresholds
from sciona.atoms.ml.masked_attention import masked_attention_pool, sigmoid_context_gate
from sciona.video_multilabel_shards import FeatureShards, validate_populations


@dataclass
class ShardedVideoEnsemble:
    neural: object
    vectorizer: object
    baseline: list
    batch_size: int
    history: list

    def components(self, population):
        if not isinstance(population, FeatureShards) or population.width != self.neural.features:
            raise ValueError('Aligned feature shard store required')
        neural, linear = [], []
        self.neural.eval()
        with torch.no_grad():
            for batch in population.batches(self.batch_size):
                frames, mask = pad_sequences([row['frames'] for row in batch])
                neural.append(torch.sigmoid(self.neural(frames, mask)).numpy())
                features = self.vectorizer.transform(row['sparse'] for row in batch)
                linear.append(np.column_stack([model.predict_proba(features)[:,1] for model in self.baseline]))
        return score_matrix(np.concatenate(neural)), score_matrix(np.concatenate(linear))

    def predict(self, population):
        neural, linear = self.components(population)
        return score_matrix((neural + linear) / 2)


def fit(training, *, seed=12, hidden=8, epochs=80, batch_size=4, learning_rate=.02,
        pool_function=masked_attention_pool, gate_function=sigmoid_context_gate):
    if not isinstance(training, FeatureShards) or not training.labelled:
        raise ValueError('Labelled training shards required')
    for name, value in [('seed',seed), ('hidden',hidden), ('epochs',epochs), ('batch_size',batch_size)]:
        if type(value) is not int or value < (0 if name == 'seed' else 1):
            raise ValueError('Invalid training control')
    if seed >= 2**32:
        raise ValueError('Invalid seed')
    if type(learning_rate) not in (int,float) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid learning rate')
    y = label_matrix([list(row) for row in training.labels], (len(training), training.label_count))
    vectorizer = DictVectorizer(sparse=True, sort=True)
    features = vectorizer.fit_transform(row['sparse'] for row in training)
    if not features.shape[1]:
        raise ValueError('Training sparse vocabulary is empty')
    baseline = []
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for column in y.T:
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(features, column)
            baseline.append(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ReusableVideoHead(training.width, hidden, y.shape[1],
            pool_function=pool_function, gate_function=gate_function).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    target = torch.tensor(y, dtype=torch.float64)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs + 1):
        model.eval()
        total = 0.
        offset = 0
        with torch.no_grad():
            for batch in training.batches(batch_size):
                frames, mask = pad_sequences([row['frames'] for row in batch])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(frames,mask), target[offset:offset+len(batch)])
                total += float(loss) * len(batch)
                offset += len(batch)
        if offset != len(training):
            raise ValueError('Incomplete loss evaluation population')
        loss_value = total / len(training)
        if not np.isfinite(loss_value):
            raise ValueError('Nonfinite training loss')
        history.append(loss_value)
        if epoch == epochs:
            break
        model.train()
        order = rng.permutation(len(training))
        for start in range(0, len(training), batch_size):
            indices = order[start:start+batch_size]
            frames, mask = pad_sequences([training[int(i)]['frames'] for i in indices])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(frames,mask), target[indices])
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Invalid neural gradients')
            optimizer.step()
    model.eval()
    return ShardedVideoEnsemble(model, vectorizer, baseline, batch_size, history)


def execute(training, calibration, query, controls, *,
            pool_function=masked_attention_pool, gate_function=sigmoid_context_gate,
            threshold_fit_function=fit_empirical_f1_thresholds,
            threshold_apply_function=apply_column_thresholds):
    validate_populations(training, calibration, query)
    if type(controls) is not dict or set(controls) != {'seed','hidden','epochs','batch_size','learning_rate'}:
        raise ValueError('Explicit complete training controls required')
    # Validate calibration classes before fitting, without loading frame populations.
    labels = [list(row) for row in calibration.labels]
    label_matrix(labels, (len(calibration), calibration.label_count))
    fitted = fit(training, **controls, pool_function=pool_function, gate_function=gate_function)
    thresholds = threshold_fit_function(fitted.predict(calibration), np.asarray(labels, dtype=np.int64))
    scores = fitted.predict(query)
    predicted = threshold_apply_function(scores, thresholds)
    return dict(scores=scores.tolist(), labels=predicted.tolist(), thresholds=thresholds.tolist(),
        training_rows=len(training), calibration_rows=len(calibration), query_rows=len(query),
        label_count=len(thresholds), sparse_models=len(fitted.baseline), epochs=controls['epochs'],
        initial_training_loss=fitted.history[0], final_training_loss=fitted.history[-1])
