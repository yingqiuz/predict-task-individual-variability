import os
import numpy as np
from glmnet import ElasticNet
from scipy.io import loadmat
from utils import (WDIR, pearson_r, parse_args)
from sklearn.linear_model import RidgeCV
from joblib import Parallel, delayed


def rest_predict(basis, num_modes, fold, nidps, numIC_bases, residualise=True):
    print("loading training data...")
    train_X = []
    if residualise:
        for k in range(num_modes):
            bases = np.load(WDIR + '{1}.bases.{0}.residual2.numIC.{2}.A.fold.{3}.npy'
                            .format(k + 1, basis, numIC_bases, fold))
            train_X.append(bases)
    else:
        for k in range(num_modes):
            bases = np.load(WDIR + '{1}.bases.{0}.numIC.{2}.A.fold.{3}.npy'
                            .format(k + 1, basis, numIC_bases, fold))
            train_X.append(bases)
    train_X = np.concatenate(train_X, axis=1)
    # standardize
    train_X -= train_X.mean(axis=0)
    train_X /= train_X.std(axis=0)
    print("size of train_X is,", train_X.shape)


def task_predict(basis, fold, nidps,
                 numIC_bases, numIC_tasks, alpha,
                 residualise=True, n_jobs=24):
    numIC_tasks = int(numIC_tasks)
    train_folds = [f for f in range(1, 4) if f != fold]
    index_train = [np.loadtxt(WDIR + "sublist.fold.{}.test_index.txt"
                              .format(f), dtype=int) for f in train_folds]
    index_train = np.concatenate(index_train)
    print(index_train.shape)
    train_X = np.zeros((index_train.shape[0], numIC_tasks * 3))
    if residualise:
        for task_id in range(1, 4):
            predicted_task = []
            for f in train_folds:
                tmp = np.load(WDIR + "{0}.EN.numIC_bases.{1}."
                            "numIC_tasks.{2}.task_id.{3}.alpha.{4}.fold.{5}."
                            "residual2.lasso.ensemble.test.npy"
                            .format(basis, numIC_bases, numIC_tasks, task_id, alpha, f))
                predicted_task.append(tmp)
            train_X[:, (task_id-1)*numIC_tasks:task_id*numIC_tasks] = \
                np.concatenate(predicted_task, axis=0)
    else:
        for task_id in range(1, 4):
            predicted_task = []
            for f in train_folds:
                tmp = np.load(WDIR + "{0}.EN.numIC_bases.{1}."
                                     "numIC_tasks.{2}.task_id.{3}.alpha.{4}.fold.{5}."
                                     "lasso.ensemble.test.npy"
                              .format(basis, numIC_bases, numIC_tasks, task_id, alpha, f))
                predicted_task.append(tmp)
            train_X[:, (task_id - 1) * numIC_tasks:task_id * numIC_tasks] = \
                np.concatenate(predicted_task, axis=0)
    train_X -= train_X.mean(axis=0)
    train_X /= train_X.std(axis=0)
    print(" the size of training_X is, ", train_X.shape)

    # load vars
    nidps_train = nidps[index_train]
    nidps_mean = nidps_train.mean(axis=0)
    nidps_train -= nidps_mean

    if residualise:
        coeff_name = WDIR + "task_predicted_nidp.fold.{}." \
                            "residual2.coeff.npy".format(fold)
    else:
        coeff_name = WDIR + "task_predicted_nidp.fold.{}.coeff.npy".format(fold)
    if os.path.isfile(coeff_name):
        coeffs = np.load(coeff_name)
    else:
        print("coefficients not found... training...")
        def elasticnet(k):
            print("fitting for column {}".format(k))
            y = nidps_train[:, k]
            ind = ~np.isnan(y)
            if alpha == 0:
                clf = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0), fit_intercept=False)
            else:
                clf = ElasticNet(alpha=alpha, fit_intercept=False)
            clf.fit(X=train_X[ind, :], y=y[ind])
            return clf.coef_

        results = Parallel(n_jobs=n_jobs, prefer="threads") \
            (delayed(elasticnet)(i) for i in range(nidps_train.shape[1]))

        coeffs = np.concatenate([item[:, np.newaxis] for item in results], axis=1)
        print("size of coefficients is,", coeffs.shape)
        np.save(coeff_name, coeffs)

    test_X = []
    if residualise:
        for task_id in range(1, 4):
            predicted_task = np.load(WDIR + "{0}.EN.numIC_bases.{1}."
                                            "numIC_tasks.{2}.task_id.{3}.alpha.{4}.fold.{5}."
                                            "residual2.lasso.ensemble.test.npy"
                                     .format(basis, numIC_bases, numIC_tasks,
                                             task_id, alpha, fold))
            test_X.append(predicted_task)
    else:
        for task_id in range(1, 4):
            predicted_task = np.load(WDIR + "{0}.EN.numIC_bases.{1}."
                                            "numIC_tasks.{2}.task_id.{3}.alpha.{4}.fold.{5}."
                                            "lasso.ensemble.test.npy"
                                     .format(basis, numIC_bases, numIC_tasks,
                                             task_id, alpha, fold))
            test_X.append(predicted_task)
    test_X = np.concatenate(test_X, axis=1)
    test_X -= test_X.mean(axis=0)
    test_X /= test_X.std(axis=0)

    pred = test_X.dot(coeffs) + nidps_mean
    if residualise:
        np.save(WDIR + "task_predicted_nidp.fold.{}.residual2.npy".format(fold), pred)
    else:
        np.save(WDIR + "task_predicted_nidp.fold.{}.npy".format(fold), pred)


def task_predict2(basis, fold, nidps,
                 numIC_bases, numIC_tasks, alpha,
                 residualise=True, n_jobs=24):
    index_train = np.loadtxt(WDIR + "sublist.fold.{}.train_index.txt"
                             .format(fold), dtype=int)
    numIC_tasks = int(numIC_tasks)
    train_X = np.zeros((index_train.shape[0], numIC_tasks * 3))
    if residualise:
        for task_id in range(1, 4):
            train_X[:, (task_id-1)*numIC_tasks:task_id*numIC_tasks] = \
                np.load(WDIR + "tasks.{0}.residual2.numIC.{1}.A.fold.{2}.npy"
                        .format(task_id, numIC_tasks, fold))
    else:
        for task_id in range(1, 4):
            train_X[:, (task_id - 1) * numIC_tasks:task_id * numIC_tasks] = \
                np.load(WDIR + "tasks.{0}.numIC.{1}.A.fold.{2}.npy"
                        .format(task_id, numIC_tasks, fold))
    train_X -= train_X.mean(axis=0)
    train_X /= train_X.std(axis=0)
    print("the size of training_X is, ", train_X.shape)

    # load vars
    nidps_train = nidps[index_train]
    nidps_mean = nidps_train.mean(axis=0)
    nidps_train -= nidps_mean

    if residualise:
        coeff_name = WDIR + "task_predicted_nidp.fold.{}." \
                            "residual2.coeff.npy".format(fold)
    else:
        coeff_name = WDIR + "task_predicted_nidp.fold.{}.coeff.npy".format(fold)
    if os.path.isfile(coeff_name):
        coeffs = np.load(coeff_name)
    else:
        print("coefficients not found... training...")
        def elasticnet(k):
            print("fitting for column {}".format(k))
            y = nidps_train[:, k]
            ind = ~np.isnan(y)
            y = y[ind]
            if y.shape[0] < 2:
                return np.zeros((train_X.shape[1], ))
            if alpha == 0:
                clf = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0), fit_intercept=False)
            else:
                clf = ElasticNet(alpha=alpha, fit_intercept=False)
            clf.fit(X=train_X[ind, :], y=y)
            return clf.coef_

        results = Parallel(n_jobs=n_jobs, prefer="threads") \
            (delayed(elasticnet)(i) for i in range(nidps_train.shape[1]))

        coeffs = np.concatenate([item[:, np.newaxis] for item in results], axis=1)
        print("size of coefficients is,", coeffs.shape)
        np.save(coeff_name, coeffs)

    test_X = []
    if residualise:
        for task_id in range(1, 4):
            predicted_task = np.load(WDIR + "{0}.EN.numIC_bases.{1}."
                                            "numIC_tasks.{2}.task_id.{3}.alpha.{4}.fold.{5}."
                                            "residual2.lasso.ensemble.test.npy"
                                     .format(basis, numIC_bases, numIC_tasks,
                                             task_id, alpha, fold))
            test_X.append(predicted_task)
    else:
        for task_id in range(1, 4):
            predicted_task = np.load(WDIR + "{0}.EN.numIC_bases.{1}."
                                            "numIC_tasks.{2}.task_id.{3}.alpha.{4}.fold.{5}."
                                            "lasso.ensemble.test.npy"
                                     .format(basis, numIC_bases, numIC_tasks,
                                             task_id, alpha, fold))
            test_X.append(predicted_task)
    test_X = np.concatenate(test_X, axis=1)
    test_X -= test_X.mean(axis=0)
    test_X /= test_X.std(axis=0)

    pred = test_X.dot(coeffs) + nidps_mean
    if residualise:
        np.save(WDIR + "task_predicted_nidp.fold.{}.residual2.npy".format(fold), pred)
    else:
        np.save(WDIR + "task_predicted_nidp.fold.{}.npy".format(fold), pred)


if __name__ == "__main__":
    args = parse_args()
    basis = args.basis
    numIC_bases = args.numIC_bases
    numIC_tasks = args.numIC_tasks
    alpha = args.alpha
    residualise = args.residualise
    n_jobs = args.n_jobs
    fold = args.fold
    num_modes = args.num_modes
    basis += str(num_modes)
    nidps = np.load(WDIR + "vars_i_deconfnoage17560.npy")
    task_predict2(basis, fold, nidps,
                 numIC_bases, numIC_tasks, alpha,
                 residualise=residualise, n_jobs=n_jobs)