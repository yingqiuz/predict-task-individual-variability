import numpy as np
from tqdm import tqdm
from baseline import baseline
from sparse import sparse


def ensemble(bases_train, tasks_train, bases_test, tasks_test, rest_ic, task_ic,
             residualise=True, n_jobs=1, standardise=True):
    """
    the ensemble model
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
    :param residualise: Boolean, whether run the model on residualised data
    :param n_jobs: Boolean, number of jobs in parallel
    :param standardise: Boolean, whether standardise the data
    :return: Dict, predictions for train and test data {"train": ndarray, "test": ndarray}
    """
    ### data preparation
    # make individual maps zero-centered (across voxels)
    print("demean the resting-state data...")
    for train, test in zip(bases_train, bases_test):
        train -= train.mean(axis=1)[:, np.newaxis]
        test -= test.mean(axis=1)[:, np.newaxis]
    print("dmean the task data...")
    tasks_train -= tasks_train.mean(axis=1)[:, np.newaxis]
    tasks_test -= tasks_test.mean(axis=1)[:, np.newaxis]
    num_modes = len(bases_train)  # number of functional modes
    n_train, n_voxels = tasks_train.shape  # number of training subjects, voxels
    assert num_modes == len(bases_test) and n_voxels == tasks_test.shape[1], \
        "Dimensions of data mismatch."
    #n_test = tasks_test.shape[1]  # number of test subjects
    # calculate group mean (be careful if there are missing values...)
    rest_mean = [item.mean(axis=0)[np.newaxis, :] for item in bases_train]
    task_mean = tasks_train.mean(axis=0)[np.newaxis, :]
    # calculate amplitude of rest and task
    rest_amp_train = []
    rest_amp_test = []
    print("calculating resting-state amplitude...")
    for k in range(num_modes):
        rest_amp_train.append(bases_train[k].dot(np.linalg.pinv(rest_mean[k])))
        rest_amp_test.append(bases_test[k].dot(np.linalg.pinv(rest_mean[k])))
    rest_amp_train, rest_amp_test = list(
        map(lambda x: np.concatenate(x, axis=1), [rest_amp_train, rest_amp_test])
    )
    print("calculating task amplitude...")
    task_amp_train, task_amp_test = list(
        map(lambda x: x.dot(np.linalg.pinv(task_mean)),
            [tasks_train, tasks_test])
    )
    # save the amplitude files if necessary
    #rest_amplitude = {"train": rest_amp_train, "test": rest_amp_test}
    #task_amplitude = {"train": task_amp_train, "test": task_amp_test}
    # residualise the data
    if residualise:
        print("residualise resting-state data...")
        for k in range(num_modes):
            bases_train[k] -= rest_amp_train[:, [k]].dot(rest_mean[k])
            bases_test[k] -= rest_amp_test[:, [k]].dot(rest_mean[k])
        print("residualising task data...")
        tasks_train -= task_amp_train.dot(task_mean)
        tasks_test -= task_amp_test.dot(task_mean)
    ### model fitting
    print("fitting baseline model...")
    baseline_predictions = baseline(
        bases_train, tasks_train, bases_test, tasks_test,
        standardise=standardise, n_jobs=n_jobs
    )
    print("fitting sparse model...")
    sparse_predictions = sparse(
        bases_train, tasks_train, bases_test, tasks_test,
        rest_ic, task_ic, n_jobs=n_jobs
    )
    # fit ensemble model
    ensemble_coeffs = np.zeros((3, n_voxels))
    for k in tqdm(range(n_voxels)):  # fit_ensemble(k):
        ensemble_coeffs[:, k] = np.linalg.pinv(
            np.c_[np.ones(n_train, ),
                  baseline_predictions["train"][:, k],
                  sparse_predictions["train"][:, k]]
        ).dot(tasks_train[:, k])
    pred = ensemble_coeffs[0, :] + \
        baseline_predictions["test"] * ensemble_coeffs[1, :] + \
        sparse_predictions["test"] * ensemble_coeffs[2, :]
    # add group average back if residualised
    if residualise:
        # generate surrogate task amplitude
        predicted_amplitude = np.dot(
            rest_amp_test,
            np.linalg.pinv(rest_amp_train).dot(task_amp_train)
        )
        pred += predicted_amplitude.dot(task_mean)
    return pred
