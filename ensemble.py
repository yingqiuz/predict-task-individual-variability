import numpy as np
from tqdm import tqdm
from baseline import baseline
from sparse import sparse


def ensemble2(train_rest, test_rest, train_task, test_task, rest_ic, task_ic,
             residualise=True, n_jobs=1, standardise=True):
    """
    :param train_rest: resting-state modes (training data)
        a list of (N, V) ndarray, N number of training subjects and V #voxels
    :param train_task: task contrast maps (training data)
        (N, V) ndarray
    :param test_rest: resting-state modes (test data)
        a list of (N, V) ndarray, N number of training subjects and V #voxels
    :param test_task: task contrast maps (test data)
        (N, V) ndarray
    :param resualise: Boolean, residualise the data or not
    :param n_jobs: int, #jobs in parallel
    :param standardise: Boolean, normalise each modes or not
    :return: (N, V) ndarray, predictions
    """
    # make individual maps zero-centered
    print("demean the data...")
    for train, test in zip(train_rest, test_rest):
        train -= train.mean(axis=1)[:, np.newaxis]
        test -= test.mean(axis=1)[:, np.newaxis]
    train_task -= train_task.mean(axis=1)[:, np.newaxis]
    test_task -= test_task.mean(axis=1)[:, np.newaxis]
    num_modes = len(train_rest)  # number of functional modes
    n_train, n_voxels = train_task.shape  # number of training subjects, voxels
    assert num_modes == len(test_rest) and n_voxels == test_task.shape[1], \
        "train and test data mismatch."
    #n_test = test_task.shape[1]  # number of test subjects
    # calculate group mean
    rest_mean = [item.mean(axis=0)[np.newaxis, :] for item in train_rest]
    task_mean = train_task.mean(axis=0)[np.newaxis, :]
    # calculate amplitude of rest and task
    rest_amp_train = []
    rest_amp_test = []
    print("calculating resting-state amplitude...")
    for k in range(num_modes):
        rest_amp_train.append(train_rest[k].dot(np.linalg.pinv(rest_mean[k])))
        rest_amp_test.append(test_rest[k].dot(np.linalg.pinv(rest_mean[k])))
    rest_amp_train, rest_amp_test = list(
        map(lambda x: np.concatenate(x, axis=1), [rest_amp_train, rest_amp_test])
    )
    print("calculating task amplitude...")
    task_amp_train, task_amp_test = list(
        map(lambda x: x.dot(np.linalg.pinv(task_mean)),
            [train_task, test_task])
    )
    #rest_amplitude = {"train": rest_amp_train, "test": rest_amp_test}
    #task_amplitude = {"train": task_amp_train, "test": task_amp_test}
    # residualise the data
    if residualise:
        print("residualise resting-state data...")
        for k in range(num_modes):
            train_rest[k] -= rest_amp_train[:, [k]].dot(rest_mean[k])
            test_rest[k] -= rest_amp_test[:, [k]].dot(rest_mean[k])
        print("residualising task data...")
        train_task -= task_amp_train.dot(task_mean)
        test_task -= task_amp_test.dot(task_mean)
    print("fitting baseline model...")
    baseline_predictions = baseline(
        train_rest, train_task, test_rest, test_task,
        standardise=standardise, n_jobs=n_jobs
    )
    print("fitting sparse model...")
    sparse_predictions = sparse(
        train_rest, train_task, test_rest, test_task,
        rest_ic, task_ic, n_jobs=n_jobs
    )
    # fit ensemble model
    ensemble_coeffs = np.zeros((3, n_voxels))
    for k in tqdm(range(n_voxels)):  # fit_ensemble(k):
        ensemble_coeffs[:, k] = np.linalg.pinv(
            np.c_[np.ones(n_train, ),
                  baseline_predictions["train"][:, k],
                  sparse_predictions["train"][:, k]]
        ).dot(train_task[:, k])
    pred = ensemble_coeffs[0, :] + \
        baseline_predictions["test"] * ensemble_coeffs[1, :] + \
        sparse_predictions["test"] * ensemble_coeffs[2, :]
    # add group average
    if residualise:
        # generate surrogate task amplitude
        predicted_amplitude = np.dot(
            rest_amp_test,
            np.linalg.pinv(rest_amp_train).dot(task_amp_train)
        )
        pred += predicted_amplitude.dot(task_mean)
    return pred
