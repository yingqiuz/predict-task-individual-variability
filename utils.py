import numpy as np
import scipy
import argparse
from sklearn.model_selection import KFold
from sklearn.decomposition import FastICA
WDIR = "/well/win-biobank/users/gbb787/ukbiobank/profumo20k/"


def parse_args():
    """parse argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--basis", default="profumo", dest="basis", type=str)
    parser.add_argument("-n1", "--numIC_bases", default='3000', dest="numIC_bases", type=str)
    parser.add_argument("-n2", "--numIC_tasks", default='4000', dest="numIC_tasks", type=str)
    parser.add_argument("-t", "--task_id", default=1, dest="task_id", type=int)  # indexing from 1, not 0
    parser.add_argument("-a", "--alpha", default=1, dest="alpha", type=float)
    parser.add_argument("-r", "--residualise", default=False, dest="residualise", type=bool)
    parser.add_argument("-j", "--n_jobs", default=24, dest="n_jobs", type=int)
    parser.add_argument("-f", "--fold", default=1, dest="fold", type=int)
    parser.add_argument("-nm", "--num_modes", default=50, dest="num_modes", type=int)
    parser.add_argument("-nc", "--num_pca", default=35, dest="nc", type=int)
    args = parser.parse_args()
    return args


def pearson_r(X, Y):
    """Pearson's correlation"""
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    Y_std = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    return X_std.T.dot(Y_std) / X_std.shape[0]


def score(X, Y):
    """
    return accuracy and discriminability
    :param X: voxel by subject array-like, predictions
    :param Y: voxel by subject array-like, truth
    :return: accuracy: (subject, ), array-like
    :return: discriminability: (subject, ), array_like
    """
    acc = pearson_r(X, Y)
    z = np.log((1 + acc) / (1 - acc)) / 2
    discriminability = np.diag(z) * (1 + 1 / X.shape[1]) \
        - z.mean(axis=1)
    accuracy = np.diag(acc)
    return accuracy, discriminability


def nets_svds(x, n):
    """
    efficient svd
    :param x: N by M ndarray
    :param n: int, dimension to reduce to
    :return u: (N, n) ndarray
    :return s: (n, ) ndarray
    :return v: (M, n) ndarray
    """
    if x.shape[0] < x.shape[1]:
        x_cov = x.dot(x.T)
        if n < x.shape[0]:
            s, u = scipy.linalg.eigh(
                x_cov,
                subset_by_index=(x_cov.shape[0] - n, x_cov.shape[0] - 1)
            )
        else:
            s, u = scipy.linalg.eigh(x_cov)
        # real numbers only
        u = np.real(u)
        s = np.real(s)
        sort_index = np.argsort(-s)
        u = u[:, sort_index]
        s = s[sort_index]
        s = np.sqrt(np.abs(s))
        v = x.T.dot(u).dot(np.diag(1 / s))

    else:
        x_cov = x.T.dot(x)
        if n < x.shape[1]:
            s, v = scipy.linalg.eigh(
                x_cov,
                subset_by_index=(x_cov.shape[0] - n. x_cov.shape[0] - 1),
            )
        else:
            s, v = scipy.linalg.eigh(x_cov)
        v = np.real(v)
        s = np.real(s)
        sort_index = np.argsort(-s)
        v = v[:, sort_index]
        s = s[sort_index]
        s = np.sqrt(np.abs(s))
        u = x.dot(v).dot(np.diag(1 / s))
    return u, s, v


def ica(train_data, test_data, n_components,
        fun='cube', max_iter=10000):
    """

    :param data:
    :param fold:
    :param residualise:
    :param n_components:
    :param fun:
    :param max_iter:
    :return:
    """
    # make individual maps zero-centered
    train_data -= train_data.mean(axis=1)[:, np.newaxis]
    test_data -= test_data.mean(axis=1)[:, np.newaxis]

    # conduct PCA first to reduce dimensions
    u, s, v = nets_svds(train_data, n=n_components)
    clf = FastICA(n_components=n_components, fun=fun, max_iter=max_iter)
    components = clf.fit_transform(X=v.dot(np.diag(s))).T
    train_mixing_matrix = u.dot(clf.mixing_)

    # estimate mixing matrix of test data
    u, s, v = nets_svds(components, n_components)  # to prevent svd convergence error
    test_mixing_matrix = test_data.dot(v).dot(np.diag(1 / s)).dot(u.T)
    return train_mixing_matrix, test_mixing_matrix, components


def pseudo_inverse(x):
    u, s, v = nets_svds(x, np.min(x.shape))
    if np.any(s < 1e-3):
        print("not full rank...is problematic")
        s[s<1e-3] = np.inf
    return v.dot(np.diag(1 / s)).dot(u.T)

