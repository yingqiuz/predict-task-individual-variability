import numpy as np
from joblib import Parallel, delayed
from glmnet import ElasticNet
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
from utils import ica
from copy import deepcopy


def sparse(bases_train, tasks_train, bases_test, tasks_test, rest_ic, task_ic, n_jobs=1, alpha=1.0):
    """
    function to fit the sparse model
    NB this function does not contain the residualisation step, thus either feed in the residualised data (both rest and
    task) or the original data. Please refer to ensemble.py for the residualisation step.
    :param bases_train: resting-state modes (training data)
        a list of (N, V) ndarray, N number of training subjects and V number of voxels
    :param tasks_train: task contrast maps (training data)
        (N, V) ndarray
    :param bases_test: resting-state modes (test data)
        a list of (N, V) ndarray, N number of training subjects and V #voxels
    :param tasks_test: task contrast maps (test data)
        (N, V) ndarray
    :param n_jobs: number of parallel jobs
    :param rest_ic: Int (the number of independent compoents) or
        Dict {"train": ndarray, "test" ndarray} (concatenated mixing matrices of rest ICA)
    :param task_ic: Int (the number of independent components) or
        Dict {"train": ndarray, "test": ndarray, "IC"}
    :param alpha: Float, ElasticNet mixing parameter
    :return: Dict, predictions for train and test data {"train": ndarray, "test": ndarray}
    """
    num_modes = len(bases_train)  # number of functional modes
    n_train, n_voxels = tasks_train.shape  # number of training subjects, voxels
    n_test = tasks_test.shape[0]  # number of test subjects
    # ICA - rest data
    if isinstance(rest_ic, int):
        print("conduct ICA on rest data...")
        train_x, test_x = list(map(lambda x: np.zeros((x, rest_ic * num_modes)), [n_train, n_test]))
        for k in tqdm(range(num_modes)):
            train_x[:, k*rest_ic:(k+1)*rest_ic], test_x[:, k*rest_ic:(k+1)*rest_ic], _ = ica(
                bases_train[k], bases_test[k], rest_ic,
            )
        # rest_ic = {"train": train_x, "test": test_x}  # save the dict if needed
    else:
        train_x = rest_ic["train"]
        test_x = rest_ic["test"]
    # ICA - task data
    if isinstance(task_ic, int):
        if task_ic != n_voxels:
            print("conduct ICA on task data...")
            train_y, test_y, task_components = ica(
                tasks_train, tasks_test, task_ic,
            )
            # task_ic = {"train": train_y, "test": test_y, "IC": task_components}
        else: # no ICA step
            train_y = deepcopy(tasks_train)
            test_y = deepcopy(tasks_test)
            task_components = None
    else:
        train_y = task_ic["train"]
        test_y = task_ic["test"]
        task_components = task_ic["IC"]
    # fit the sparse model
    # standardise the data
    y_mean = train_y.mean(axis=0)
    train_y -= y_mean
    test_y -= y_mean
    # train_y, test_y = list(map(lambda y: y - y_mean, [train_y, test_y]))
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    train_x, test_x = list(map(lambda x: (x - x_mean) / x_std, [train_x, test_x]))
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_sparse)(train_x, test_x, train_y[:, k], alpha)
        for k in tqdm(range(train_y.shape[1]))
    )
    # concatenate results
    pred_train = np.concatenate(
        [item[1][:, np.newaxis]
         for item in results],
        axis=1
    ) + y_mean
    pred_test = np.concatenate(
        [item[2][:, np.newaxis]
         for item in results],
        axis=1
    ) + y_mean
    if pred_train.shape[1] != n_voxels:
        sparse_predictions = {"train": pred_train.dot(task_components),
                              "test": pred_test.dot(task_components)}
    else:
        sparse_predictions = {"train": pred_train, "test": pred_test}
    return sparse_predictions


def _fit_sparse(train_x, test_x, train_y, alpha=1):
    if alpha == 0:
        clf = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0), fit_intercept=False)
    else:
        clf = ElasticNet(alpha=alpha, fit_intercept=False)
    clf.fit(X=train_x, y=train_y)
    return clf.coef_, train_x.dot(clf.coef_), test_x.dot(clf.coef_)
