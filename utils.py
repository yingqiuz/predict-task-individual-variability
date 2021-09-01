import numpy as np
import scipy
import argparse
from sklearn.model_selection import KFold
from sklearn.decomposition import FastICA
WDIR = "/path/to/my/dir/"
HCP_CONTRASTS=[
    '01_EMOTION_FACES','02_EMOTION_SHAPES','03_EMOTION_FACES-SHAPES',
    '07_GAMBLING_PUNISH','08_GAMBLING_REWARD','09_GAMBLING_PUNISH-REWARD',
    '13_LANGUAGE_MATH','14_LANGUAGE_STORY','15_LANGUAGE_MATH-STORY',
    '19_MOTOR_CUE','20_MOTOR_LF','21_MOTOR_LH','22_MOTOR_RF','23_MOTOR_RH','24_MOTOR_T',
    '25_MOTOR_AVG','26_MOTOR_CUE-AVG','27_MOTOR_LF-AVG','28_MOTOR_LH-AVG','29_MOTOR_RF-AVG',
    '30_MOTOR_RH-AVG','31_MOTOR_T-AVG','45_RELATIONAL_MATCH','46_RELATIONAL_REL',
    '47_RELATIONAL_MATCH-REL','51_SOCIAL_RANDOM','52_SOCIAL_TOM','53_SOCIAL_RANDOM-TOM',
    '57_WM_2BK_BODY','58_WM_2BK_FACE','59_WM_2BK_PLACE','60_WM_2BK_TOOL','61_WM_0BK_BODY',
    '62_WM_0BK_FACE','63_WM_0BK_PLACE','64_WM_0BK_TOOL','65_WM_2BK','66_WM_0BK','67_WM_2BK-0BK',
    '71_WM_BODY','72_WM_FACE','73_WM_PLACE','74_WM_TOOL','75_WM_BODY-AVG','76_WM_FACE-AVG','77_WM_PLACE-AVG',
    '78_WM_TOOL-AVG'
]


def parse_args():
    """parse arguments"""
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
    """
    Calculation Pearson's correlation between spatial maps X and Y
    :param X: ndarray, voxel by subject
    :param Y: ndarray, voxel by subjectfisher
    """
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    Y_std = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    return X_std.T.dot(Y_std) / X_std.shape[0]


def score(X, Y):
    """
    return accuracy and discriminability
    :param X: voxel by subject array-like, the predicted maps
    :param Y: voxel by subject array-like, the actual maps
    :return: acc: ndarray, (subject, ), prediction accuracy for each subject
    :return: disc: ndarray (subject, ), prediction disc. for each subject
    """
    acc = pearson_r(X, Y)
    z = np.log((1 + acc) / (1 - acc)) / 2
    disc = np.diag(z) * (1 + 1 / X.shape[1]) - z.mean(axis=1)
    acc = np.diag(acc)
    return acc, disc


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
    conduct ica on the training data, and estimate the mixing matrices for the test data
    :param train_data: ndarray, subjects x voxels
    :param test_data: ndarray, subjects x voxels
    :param n_components: Int, number of independent components
    :param fun: function used to estimate neg entropy
    :param max_iter: maximum number of iterations
    :return train_mixing_matrix: ndarray, the mixing matrix of train_data (as output of ICA)
    :return test_mixing_matrix: ndarray, the estimated mixing matrix of test data (regress ICA components into test_data)
    :return components: n_components x voxels, the independent components
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
    """equivalent to np.linalg.pinv"""
    u, s, v = nets_svds(x, np.min(x.shape))
    if np.any(s < 1e-3):
        print("not full rank...")
        s[s<1e-3] = np.inf
    return v.dot(np.diag(1 / s)).dot(u.T)


def dual_regression(indiv_data, group_ica):
    """
    function to estimate dual regression maps
    :param indiv_data: ndarray, voxel by time
    :param group_ica: ndarray voxel by num_modes
    :return: voxel x num_modes ndarray, the dual regression maps
    """
    # if have multiple sessions, average the dual reg maps across sessions
    v, nt = indiv_data.shape
    _, d = group_ica.shape
    # time courses
    ts = pseudo_inverse(group_ica - group_ica.mean(axis=0)).dot(indiv_data - indiv_data.mean(axis=0))
    # dr maps
    indiv_data -= indiv_data.mean(axis=1)[:, np.newaxis]
    ts -= ts.mean(axis=1)[:, np.newaxis]
    pinv_ts = pseudo_inverse(ts)
    dr = indiv_data.dot(pinv_ts)
    sigsq = np.sum((indiv_data - dr.dot(ts))**2, axis=1) / (nt - d)
    varcope = np.dot(sigsq[:, np.newaxis], np.diag(np.dot(pinv_ts.T, pinv_ts))[np.newaxis, :])
    return dr / np.sqrt(varcope)
