import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def read_data(fpath):
    pd_data = pd.read_csv(fpath, header=None)
    data = pd_data.to_numpy()
    x = data[:, :-1]
    y = data[:, -1].astype(np.int)
    return x, y


def split_cr(x, y, i):
    val_x = x[313 * i:313 * (i + 1)]
    train_x = np.delete(x, slice(313 * i, 313 * (i + 1)), 0)
    val_y = y[313 * i:313 * (i + 1)]
    train_y = np.delete(y, slice(313 * i, 313 * (i + 1)), 0)
    return train_x, train_y, val_x, val_y


def cross_validation(x, y, T, algo):
    algo_map = {
        'ada': AdaBoostClassifier,
        'logistic': LogisticBoostClassifier
    }
    BoostClassifier = algo_map[algo]
    perfs = np.empty(10)
    for i in range(10):
        train_x, train_y, val_x, val_y = split_cr(x, y, i)
        model = BoostClassifier(n_estimators=T)
        model.fit(train_x, train_y)
        acc = model.score(val_x, val_y)
        perfs[i] = acc
    return perfs


class LogisticBoostClassifier:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.base_estimators = []
        self.alpha = np.empty(n_estimators)
        self.rng = np.random.default_rng()

    def draw_samples(self, x, y, D_t, rng):
        n_samples = x.shape[0]
        indices = rng.choice(n_samples, n_samples, replace=True,
                             p=D_t)
        x_train = x[indices]
        y_train = y[indices]
        return x_train, y_train

    def fit(self, x, y):
        C_t = np.ones(x.shape[0]) / x.shape[0]
        for t in range(self.n_estimators):
            C_t = self._boost(t, x, y, C_t, self.rng)

    def _boost(self, t, x, y, C_t, rng):
        base_estimator = DecisionTreeClassifier(max_depth=1)
        D_t = 1 / (1 + C_t)
        D_t /= np.sum(D_t)
        train_x, train_y = self.draw_samples(x, y, D_t, rng)
        base_estimator.fit(train_x, train_y)
        pred_y = base_estimator.predict(train_x)
        err = np.mean(pred_y != train_y)
        self.alpha[t] = 1 / 2 * np.log((1 - err) / err)
        self.base_estimators.append(base_estimator)
        next_C_t = C_t * np.exp(-self.alpha[t] * train_y * pred_y)
        return next_C_t

    def predict(self, x):
        results = np.empty((x.shape[0], self.n_estimators))
        for i in range(self.n_estimators):
            results[:, i] = self.base_estimators[i].predict(x)
        final_logits = results @ self.alpha
        pred_y = 2 * (final_logits > 0) - 1
        return pred_y

    def score(self, x, y):
        pred_y = self.predict(x)
        score = np.mean(pred_y == y)
        return score


def choose_T(algo):
    x, y = read_data('abalone.train')
    T_range = 100 * np.arange(1, 11)
    perf_mean = np.empty_like(T_range).astype(np.float)
    perf_std = np.empty_like(T_range).astype(np.float)
    for i, T in enumerate(T_range):
        perfs = cross_validation(x, y, T, algo)
        perf_mean[i] = np.mean(perfs)
        perf_std[i] = np.std(perfs)
        print(f'algo:{algo}, T:{T}, mean:{perf_mean[i]}, std:{perf_std[i]}')
    plt.plot(T_range, perf_mean)
    plt.plot(T_range, perf_mean + perf_std)
    plt.plot(T_range, perf_mean - perf_std)
    plt.xticks(T_range)
    algo_name_map = {
        'ada': 'AdaBoost',
        'logistic': 'LogisticBoost'
    }
    plt.title(f'Cross Validation Error for {algo_name_map[algo]}')
    plt.legend(['mean', 'mean+std', 'mean-std'])
    plt.show()
    plt.close()
    return perf_mean, perf_std


def final_test(algo, T):
    algo_map = {
        'ada': AdaBoostClassifier,
        'logistic': LogisticBoostClassifier
    }
    BoostClassifier = algo_map[algo]
    train_x, train_y = read_data('abalone.train')
    test_x, test_y = read_data('abalone.test')
    model = BoostClassifier(n_estimators=T)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    return score


if __name__ == '__main__':
    # ada_mean, ada_std = choose_T('ada')
    # ada_T = 300
    # ada_score = final_test('ada', T=ada_T)
    # print(f'ada final score: {ada_score}')
    logistic_mean, logistic_std = choose_T('logistic')
    # logistic_T = 300
    # logistic_score = final_test('logistic', T=logistic_T)
    # print(f'logistic final score: {logistic_score}')
