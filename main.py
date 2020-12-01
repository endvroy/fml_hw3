import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
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
        model = BoostClassifier(n_estimators=T, algorithm='SAMME')
        model.fit(train_x, train_y)
        acc = model.score(val_x, val_y)
        perfs[i] = acc
    return perfs


class LogisticBoostClassifier(AdaBoostClassifier):
    def _boost(self, iboost, X, y, sample_weight, random_state):
        # implement logistic boost with adaboost with a trick
        # sample weight assumed to be C_t
        real_sample_weight = 1 / (1 + sample_weight)  # this is D_t
        real_sample_weight /= np.sum(real_sample_weight)
        last_real_sample_weight = real_sample_weight.copy()
        real_sample_weight, estimator_weight, estimator_error = super()._boost(
            iboost, X, y, real_sample_weight, random_state)  # boost with D_t
        multiplier = real_sample_weight / last_real_sample_weight
        next_C_t = sample_weight * multiplier  # update C_t
        next_C_t /= next_C_t.sum()
        return next_C_t, estimator_weight, estimator_error


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


if __name__ == '__main__':
    # ada_mean, ada_std = choose_T('ada')
    ada_T = 300
    logistic_mean, logistic_std = choose_T('logistic')
