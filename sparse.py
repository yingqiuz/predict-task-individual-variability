import numpy as np
from joblib import Parallel, delayed
from glmnet import ElasticNet
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
from utils import ica


def sparse(train_rest, train_task, test_rest, test_task,
           rest_ic, task_ic, n_jobs=1):
    """
    function to fit the baseline model
    :param train_rest:
    :param train_task:
    :param test_rest:
    :param test_task:
    :param n_jobs:
    :return:
    """
    num_modes = len(train_rest)  # number of functional modes
    n_train, n_voxels = train_task.shape  # number of training subjects, voxels
    n_test = test_task.shape[0]  # number of test subjects
    # ICA - rest data
    if isinstance(rest_ic, int):
        print("conduct ICA on rest data...")
        train_x, test_x = list(map(lambda x: np.zeros((x, rest_ic * num_modes)), [n_train, n_test]))
        for k in tqdm(range(num_modes)):
            train_x[:, k*rest_ic:(k+1)*rest_ic], test_x[:, k*rest_ic:(k+1)*rest_ic], _ = ica(
                train_rest[k], test_rest[k], rest_ic,
            )
        # rest_ic = {"train": train_x, "test": test_x}  # save the dict if needed
    else:
        train_x = rest_ic["train"]
        test_x = rest_ic["test"]
    # ICA - task data
    if isinstance(task_ic, int):
        print("conduct ICA on task data...")
        train_y, test_y, task_components = ica(
            train_task, test_task, task_ic,
        )
        # task_ic = {"train": train_y, "test": test_y, "IC": task_components}
    else:
        train_y = task_ic["train"]
        test_y = task_ic["test"]
        task_components = task_ic["IC"]
    # fit sparse model
    # standardise the data
    y_mean = train_y.mean(axis=0)
    train_y, test_y = list(map(lambda y: y - y_mean, [train_y, test_y]))
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    train_x, test_x = list(map(lambda x: (x - x_mean) / x_std, [train_x, test_x]))
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_sparse)(train_x, test_x, train_y[:, k])
        for k in tqdm(range(train_y.shape[1]))
    )
    # concatenate results
    pred_train = np.concatenate(
        [item[1][np.newaxis, :]
         for item in results],
        axis=0
    ) + y_mean
    pred_test = np.concatenate(
        [item[2][np.newaxis, :]
         for item in results],
        axis=0
    ) + y_mean
    sparse_predictions = {"train": pred_train.dot(task_components),
                          "test": pred_test.dot(task_components)}
    return sparse_predictions


def _fit_sparse(train_x, test_x, train_y, alpha):
    if alpha == 0:
        clf = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0), fit_intercept=False)
    else:
        clf = ElasticNet(alpha=alpha, fit_intercept=False)
    clf.fit(X=train_x, y=train_y)
    return clf.coef_, train_x.dot(clf.coef_), test_x.dot(clf.coef_)
