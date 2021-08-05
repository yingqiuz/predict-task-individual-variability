"""THIS SCRIPT REQUIRES A LOT MEMORY.
TO SAVE MEMORY, DO NOT LOAD ALL MODES AT THE SAME TIME"""
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

def baseline(train_rest, train_task, test_rest,
             test_task, standardise=True, n_jobs=1):
    """
    function to fit the baseline model
    :param train_rest: resting-state modes (training data)
        a list of (N, V) ndarray, N number of training subjects and V #voxels
    :param train_task: task contrast maps (training data)
        (N, V) ndarray
    :param test_rest: resting-state modes (test data)
        a list of (N, V) ndarray, N number of training subjects and V #voxels
    :param test_task: task contrast maps (test data)
        (N, V) ndarray
    :param resualise: Boolean, residualise the data or not
    :param normalise: Boolean, normalise each modes or not
    :return: baseline_predictions
    """
    # make individual maps zero-centred
    if standardise:
        print("standardise the data...")
        for train, test in zip(train_rest, test_rest):
            train /= train.std(axis=1)[:, np.newaxis]
            test /= test.std(axis=1)[:, np.newaxis]
        train_task /= train_task.std(axis=1)[:, np.newaxis]
        test_task /= test_task.std(axis=1)[:, np.newaxis]
    # number of training subjects, voxels
    n_train, n_voxels = train_task.shape
    n_test = test_task.shape[0]  # number of test subjects
    # fit baseline model for each subject
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_baseline)(
            np.concatenate([el[i][:, np.newaxis] for el in train_rest], axis=1),
            train_task[i]
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
                    np.concatenate([el[i][:, np.newaxis] for el in test_rest], axis=1),
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
