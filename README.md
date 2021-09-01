# predict-task-individual-variability
This is the repository for https://www.biorxiv.org/content/10.1101/2021.08.19.456783v1.

## the baseline model
[baseline.py](https://github.com/yingqiuz/predict-task-individual-variability/blob/main/baseline.py) contains the baseline model. For a given task contrast, it seeks to find a linear combination of functional modes to reconstruct the task contrast map for each individual.

#### example
```python
from baseline import baseline
baseline_predictions = baseline(train_x, train_y, test_x, test_y)
```
#### parameters and returns
 - `train_x` and `test_x` should be lists of ndarrays, e.g., if you have 50 functional modes, then `train_x` is a list of 50 (n_train, n_voxels) ndarrays.
 - `train_y` and `test_y` are individual task maps (of a given contrast), e.g., `train_y` is a (n_train, n_voxels) array.
 - It returns a dictionary of predicted task maps `{"train": pred_train, "test": pred_test}`, where `pred_train` and `pred_test` each is a n_subject x n_voxels array.
## the sparse model
[sparse.py](https://github.com/yingqiuz/predict-task-individual-variability/blob/main/sparse.py) contains the sparse model. It introduces more spatial complexity to the baseline mode.

#### example
```python
from sparse import sparse
sparse_predictions = sparse(train_x, train_y, test_x, test_y, rest_ic=800, task_ic=800, n_jobs=1, alpha=1)
```
#### parameters and returns
 - `train_x`, `train_y`, `test_x`, and `test_y` have the same types/formats as in `baseline.py`.
 - `rest_ic` can either be an Int, specifying the number of independent compoenents that each array in `train_x` should be reduced to, or be a dictionary `{"train": train_ic, "test": test_ic}`, where `train_ic` and `test_ic` are precomputed (and concatenated) mixing matrices.
 - `task_ic` can either be an Int, specifying the number of independent compoenents that `train_y` should be reduced to, or be a dictionary `{"train": train_ic, "test": test_ic}`, where `train_ic` and `test_ic` are precomputed mixing matrix of the given task contrast.
 - It returns a dictionary of predicted task maps `{"train": pred_train, "test": pred_test}`, where `pred_train` and `pred_test` each is a n_subject x n_voxels array.
## the ensemble model
[ensemble.py](https://github.com/yingqiuz/predict-task-individual-variability/blob/main/ensemble.py) runs the baseline model and the sparse model and combines them to give a single set of predictions.

#### examples
```python
from ensemble import ensemble
ensemble_predictions = ensemble(train_x, train_y, test_x, test_y, rest_ic=800, task_ic=800, residualise=True, n_jobs=-1)
```
#### parameters and returns
 - `train_x` and `test_x` should be lists of ndarrays, e.g., if you have 50 functional modes, then `train_x` is a list of 50 (n_train, n_voxels) ndarrays.
 - `train_y` and `test_y` are individual task maps (of a given contrast), e.g., `train_y` is a (n_train, n_voxels) array.
 - `rest_ic` can either be an Int, specifying the number of independent compoenents that each array in `train_x` should be reduced to, or be a dictionary `{"train": train_ic, "test": test_ic}`, where `train_ic` and `test_ic` are precomputed (and concatenated) mixing matrices.
 - `task_ic` can either be an Int, specifying the number of independent compoenents that `train_y` should be reduced to, or be a dictionary `{"train": train_ic, "test": test_ic}`, where `train_ic` and `test_ic` are precomputed mixing matrix of the given task contrast.
 - `residualise` specifies whether to run the model on residuals (and add the group-averaged back in at the end) or on the original data.
 - It returns a dictionary of predicted task maps `{"train": pred_train, "test": pred_test}`, where `pred_train` and `pred_test` each is a n_subject x n_voxels array.
