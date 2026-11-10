"""Sparse neural regression in an explicitly caller-selected response coordinate.

Adapted from pjankiewicz/mercari-solution, commit
374301ad1c32cbc93dcc40313d5d7bb9c5503746, mercari/tf_sparse.py.
Copyright (c) 2018 Paweł Jankiewicz, Konstantin Lopuhin; MIT license:
docs/licenses/mercari-solution-LICENSE.txt.

MSE and quantile classification are separate classes. The source's Huber-named
class inherited the MSE builder; callers map that recipe to SparseRegressor.
No application-specific logarithm, output clipping or inverse transform is
implied. TensorFlow must initialize before sklearn/pandas/Arrow on this host.
Owns one CPU-only session with single-thread operation/inter-operation pools.
This core is not yet a published provider or a portable model-state format.
"""
import tensorflow.compat.v1 as tf
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sciona.quantile_target_encoding import (
    quantile_boundaries, encode_quantile_targets, decode_quantile_probabilities,
)


def identity(x):
    return x


def sparse_linear(xs, shape, name: str, actfunc=identity):
    assert len(shape) == 2
    w = tf.get_variable(name, initializer=tf.glorot_normal_initializer(),
                        shape=shape)
    bias = tf.Variable(tf.zeros(shape[1]))
    return actfunc(tf.sparse_tensor_dense_matmul(xs, w) + bias), w


def linear(xs, shape, name: str, actfunc=identity):
    assert len(shape) == 2
    w = tf.get_variable(name, initializer=tf.glorot_normal_initializer(),
                        shape=shape)
    bias = tf.Variable(tf.zeros(shape[1]))
    output = tf.matmul(xs, w) + bias
    return actfunc(output)


class SparseMatrix:
    def __init__(self):
        self.indices = tf.placeholder(tf.int64, shape=[None, 2])
        self.values = tf.placeholder(tf.float32, shape=[None])
        self.shape = tf.placeholder(tf.int64, shape=[2])
        self.tensor = tf.SparseTensor(
            indices=self.indices,
            values=self.values,
            dense_shape=self.shape,
        )

    def feed_dict(self, X, binary_X=False):
        coo = X.tocoo()
        return {
            self.indices: np.stack([coo.row, coo.col]).T,
            self.values: coo.data if not binary_X else np.ones_like(coo.data),
            self.shape: np.array(X.shape),
        }


class SparseRegressor:
    def __init__(self, n_hidden=(16, 16),
                 learning_rate=1e-2, lr_decay=0.75,
                 n_epoch=4, batch_size=2 ** 10, bs_decay=2,
                 decay_epochs=2,
                 reg_l2=1e-5, reg_l1=0.0, use_target_scaling=True, n_bins=64, seed=0,
                 actfunc=tf.nn.relu, binary_X=False, quantile_precision=160.0,
                 classification_loss_multiplier=6.0):
        self.n_hidden = n_hidden
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.bs_decay = bs_decay
        self.decay_epochs = decay_epochs
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.use_target_scaling = use_target_scaling
        if self.use_target_scaling:
            self.target_scaler = StandardScaler()
        self.n_bins = n_bins
        self.seed = seed
        self.actfunc = actfunc
        self.binary_X = binary_X
        self.quantile_precision = quantile_precision
        self.classification_loss_multiplier = classification_loss_multiplier
        self.is_fitted = False

    def _build_model(self, n_features: int):
        self._ys = tf.placeholder(tf.float32, shape=[None])
        self._build_hidden(n_features)
        self._output = tf.squeeze(linear(self._hidden, [self.n_hidden[-1], 1], 'l_last'), axis=1)
        self._loss = tf.losses.mean_squared_error(self._output, self._ys)
        self._add_regularization()

    def _build_hidden(self, n_features: int):
        hidden, self._w_1 = sparse_linear(
            self._xs.tensor, [n_features, self.n_hidden[0]], 'l1', self.actfunc)
        if len(self.n_hidden) == 3:
            hidden = linear(hidden, [self.n_hidden[0], self.n_hidden[1]], 'l2', self.actfunc)
        self._hidden = linear(hidden, [self.n_hidden[-2], self.n_hidden[-1]], 'l_hidden', self.actfunc)

    def _add_regularization(self):
        if self.reg_l2:
            self._loss += self.reg_l2 * tf.nn.l2_loss(self._w_1)
        if self.reg_l1:
            self._loss += self.reg_l1 * tf.reduce_sum(tf.abs(self._w_1))

    def fit(self, X, y):
        """Fit in the caller's numeric response coordinate, on training rows only."""
        self.close()
        self._validate_controls()
        X = self._matrix(X)
        y = np.asarray(y)
        if (y.ndim != 1 or y.shape[0] != X.shape[0] or not len(y)
                or y.dtype.kind not in 'fiu' or not np.isfinite(y).all()):
            raise ValueError('Finite aligned nonempty target vector required')
        if hasattr(self, 'percentiles'):
            del self.percentiles
        self.n_features_in_ = X.shape[1]
        self.update_count_ = 0
        self.epoch_schedule_ = []
        if self.use_target_scaling:
            self._scaler_fit(y)
            y = self._scale_target(y)
        if not np.isfinite(y).all():
            raise ValueError('Target transformation produced nonfinite values')
        self._graph = tf.Graph()
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
            use_per_session_threads=1, allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 0})
        self._session = tf.Session(graph=self._graph, config=config)
        try:
            with self._graph.as_default():
                tf.set_random_seed(self.seed)
                random_state = np.random.RandomState(self.seed)
                self._xs = SparseMatrix()
                self._lr = tf.placeholder(tf.float32, shape=[])
                self._build_model(self.n_features_in_)
                self._train_op = tf.train.AdamOptimizer(self._lr).minimize(self._loss)
                self._session.run(tf.global_variables_initializer())
                bs, lr = self.batch_size, self.learning_rate
                for epoch in range(self.n_epoch):
                    indices = random_state.permutation(len(y))
                    batches = [indices[i:i+bs] for i in range(0, len(y), bs)]
                    self.epoch_schedule_.append(dict(epoch=epoch, batch_size=bs,
                        learning_rate=lr, updates=len(batches)))
                    for selected in batches:
                        feed = self._x_feed_dict(X[selected, :])
                        feed[self._ys] = y[selected]
                        feed[self._lr] = lr
                        _, loss = self._session.run([self._train_op, self._loss], feed_dict=feed)
                        if not np.isfinite(loss):
                            raise ValueError('Nonfinite training objective')
                        self.update_count_ += 1
                    if epoch < self.decay_epochs:
                        bs *= self.bs_decay
                        lr *= self.lr_decay
            self.is_fitted = True
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        """Release the owned TensorFlow session; fitted predictions then fail."""
        session = getattr(self, '_session', None)
        if session is not None:
            session.close()
            self._session = None
        self.is_fitted = False

    def _validate_controls(self):
        if (not isinstance(self.n_hidden, (list, tuple)) or len(self.n_hidden) not in (2, 3)
                or any(type(n) is not int or n < 1 for n in self.n_hidden)):
            raise ValueError('Two or three positive hidden widths required')
        for key in ('n_epoch', 'batch_size', 'bs_decay', 'n_bins'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError('Positive integer schedule control required: '+key)
        if type(self.decay_epochs) is not int or self.decay_epochs < 0:
            raise ValueError('Nonnegative integer decay epoch count required')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('Unsigned 32-bit seed required')
        for key in ('learning_rate', 'lr_decay', 'reg_l1', 'reg_l2'):
            value = getattr(self, key)
            if type(value) not in (float, int) or not np.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative numeric control required: '+key)
        if self.learning_rate == 0 or self.lr_decay == 0:
            raise ValueError('Positive learning rate and decay factor required')
        for value in (self.quantile_precision, self.classification_loss_multiplier):
            if type(value) not in (float, int) or not np.isfinite(value) or value <= 0:
                raise ValueError('Positive quantile precision and classification loss multiplier required')
        if type(self.use_target_scaling) is not bool or type(self.binary_X) is not bool:
            raise ValueError('Explicit boolean scaling and binary-feature policies required')
        if isinstance(self, QuantileSparseRegressor) and not self.use_target_scaling:
            raise ValueError('Quantile target encoding must be enabled for classification mode')

    @staticmethod
    def _matrix(X):
        if not isinstance(X, csr_matrix) or X.ndim != 2 or X.shape[1] < 1:
            raise ValueError('CSR feature matrix with positive width required')
        if X.dtype != np.float32 or not np.isfinite(X.data).all():
            raise ValueError('Finite float32 sparse values required')
        return X

    def _x_feed_dict(self, X):
        return self._xs.feed_dict(X, self.binary_X)

    def predict(self, X, batch_size=2 ** 13):
        if not self.is_fitted:
            raise ValueError('Model must be fitted and open')
        X = self._matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature width differs from training')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Positive prediction batch size required')
        if X.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        ys = []
        for start in range(0, X.shape[0], batch_size):
            ys.extend(self._session.run(self._output, self._x_feed_dict(X[start:start+batch_size])))
        result = self._invert_target(ys)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite model prediction')
        return result

    def _scaler_fit(self, y):
        self.target_scaler.fit(y.reshape(-1, 1))

    def _scale_target(self, y):
        y = np.array(y)
        if self.use_target_scaling:
            return self.target_scaler.transform(y.reshape(-1, 1))[:, 0]
        return y

    def _invert_target(self, y):
        y = np.array(y)
        if self.use_target_scaling:
            return self.target_scaler.inverse_transform(y.reshape(-1, 1))[:, 0]
        return y


class QuantileSparseRegressor(SparseRegressor):
    def _build_model(self, n_features: int):
        self._ys = tf.placeholder(tf.float32, shape=[None, self.n_bins])
        self._build_hidden(n_features)
        logits = linear(self._hidden, [self.n_hidden[-1], self.n_bins], 'l_last')
        self._output = tf.nn.softmax(logits)
        loss_scale = self.classification_loss_multiplier / self.n_bins
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self._ys)) * loss_scale
        self._add_regularization()

    def _scale_target(self, ys):
        return encode_quantile_targets(np.array(ys), self._get_percentiles(ys),
                                       soft=True, precision=self.quantile_precision)

    def _get_percentiles(self, ys):
        if not hasattr(self, 'percentiles'):
            self.percentiles = quantile_boundaries(ys, self.n_bins)
        return self.percentiles

    def _invert_target(self, ys):
        return decode_quantile_probabilities(np.asarray(ys), self.percentiles)


def prelu(_x):
    alphas = tf.get_variable('prelu_alpha_{}'.format(_x.get_shape()[-1]), _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg
