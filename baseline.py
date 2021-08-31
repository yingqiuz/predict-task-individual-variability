"""Scripts for the baseline model"""
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def baseline(bases_train, tasks_train, bases_test, tasks_test, standardise=True, n_jobs=1):
    """
    function to fit the baseline model.
    NB this function does not contain the residualisation step, thus either feed in the residualised data (both rest and
    task) or the original data. Please refer to ensemble.py for the residualisation step.
    :param bases_train: resting-state modes (training data)
        a list of (N, V) ndarray, N number of training subjects and V number of voxels
    :param tasks_train: task contrast maps (training data)
        (N, V) ndarray
    :param bases_test: resting-state modes (test data)
        a list of (N, V) ndarray, N number of training subjects and V number of voxels
    :param tasks_test: task contrast maps (test data)
        (N, V) ndarray
    :param standardise: Boolean, whether standardise the resting-state modes (across voxels)
    :param n_jobs: Int, jobs in parallel
    :return: Dict, predictions for train and test data {"train": ndarray, "test": ndarray}
    """
    # make individual maps zero-centred
    if standardise:
        print("standardise the data...")
        for train, test in zip(bases_train, bases_test):
            train /= train.std(axis=1)[:, np.newaxis]
            test /= test.std(axis=1)[:, np.newaxis]
        tasks_train /= tasks_train.std(axis=1)[:, np.newaxis]
        tasks_test /= tasks_test.std(axis=1)[:, np.newaxis]
    # number of training subjects, voxels
    n_train, n_voxels = tasks_train.shape
    n_test = tasks_test.shape[0]  # number of test subjects
    # fit baseline model for each subject
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_baseline)(
            np.concatenate([el[i][:, np.newaxis] for el in bases_train], axis=1),
            tasks_train[i]
        ) for i in tqdm(range(n_train))
    )
    # concatenate results
    betas = np.concatenate([el[0][np.newaxis, :] for el in results], axis=0)
    pred_train = np.concatenate([el[1][np.newaxis, :] for el in results], axis=0)
    # make predictiosn for test subjects
    pred_test = np.concatenate(
        [
            item[np.newaxis, :]
            for item in Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_predict_baseline)(
                    np.concatenate([el[i][:, np.newaxis] for el in bases_test], axis=1),
                    betas.mean(axis=0)
                ) for i in tqdm(range(n_test))
            )
        ],
        axis=0
    )
    baseline_predictions = {"train": pred_train, "test": pred_test}
    return baseline_predictions


def _fit_baseline(rest, task):
    beta = np.linalg.pinv(rest).dot(task)
    return beta, rest.dot(beta)


def _predict_baseline(rest, betas):
    return rest.dot(betas)
