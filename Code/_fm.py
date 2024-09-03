###### File for Forecasting Model Functions ######
# Import
import os
import numpy as np

from joblib import Parallel, delayed
from itertools import combinations

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SequentialFeatureSelector

from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler
#from dwave.samplers import SteepestDescentSolver

from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

import gurobipy as gp
gp.setParam('OutputFlag', 0)

if os.name == 'nt':
    import dill
    dill.settings['recurse'] = True


############## ------------------- ##############
### BSSF
# Best Subset Selection of Forecasts
def bssf(y_train, cf_train, cf_pred, alpha, k, timeout, method):

    """
    Best Subset Selection of Forecasts (BSSF) based on given method.

    Parameters:
    - y_train: Training data targets.
    - cf_train: Candidate Model Prediction matrix for training.
    - cf_pred: Candidate Model Prediction matrix for prediction.
    - alpha: Regularization parameter.
    - k: Number of subsets.
    - timeout: Timeout for optimization models.
    - method: Optimization method to use ('dwave', 'qubo', 'qcbo').

    Returns:
    Tuple of prediction and solution vector.
    """

    # Validate inputs
    if any(map(lambda x: np.isnan(x).any() or x.size == 0, [y_train, cf_train, cf_pred])):
        raise ValueError("Input arrays must not contain NA values or be empty.")
    if y_train.shape[0] != cf_train.shape[0]:
        raise ValueError("Number of y_train and cf_train must match.")
    if cf_train.shape[1] != cf_pred.shape[1]:
        raise ValueError("Number of cf_train and cf_pred must match.")
    if alpha <= 0 or k <= 0 or timeout <= 0:
        raise ValueError("alpha, k, and timeout must be positive.")
    if method not in ["dwave", "gurobi", "qcbo"]:
        raise ValueError("Invalid method. Choose from 'dwave', 'gurobi', 'qcbo'.")

    # Adapt X-Matrix
    cf_train /= k
    cf_pred /= k

    # Generate Q-Matrix
    i_mat = np.ones((cf_train.shape[1], cf_train.shape[1]))
    aux_mat = y_train.T @ cf_train + alpha * k
    Q = - 2 * np.diag(aux_mat) + cf_train.T @ cf_train + alpha * i_mat

    # Optimization based on method
    if method == "dwave":
        solution = _solve_dwave(Q, timeout)
    elif method == "gurobi":
        solution = _solve_qubo(Q, timeout)
    elif method == "qcbo":
        solution = _solve_qcbo(cf_train, y_train, k, timeout)

    # Test Solution
    if np.sum(solution) != k:
        print(f"Warning: Number of selected features does not match --- {np.sum(solution)} instead of {k}!")

    # Prediction
    pred = solution @ cf_pred.T

    return pred, solution

# Optimation Function -- DWave
def _solve_dwave(Q, timeout):
    bqm = BinaryQuadraticModel('BINARY').from_qubo(Q)
    bqm.normalize()
    solver_qpu = SimulatedAnnealingSampler()
    #solver_qpu = SteepestDescentSolver()
    sampleset = solver_qpu.sample(bqm, num_reads = timeout, label = "BSSF", seed=123)
    return np.array(list(sampleset.first.sample.values()))

# Optimation Function -- Gurobi QUBO
def _solve_qubo(Q, timeout):
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = timeout
    model.Params.Threads = 1
    b = model.addMVar(shape=Q.shape[0], vtype=gp.GRB.BINARY, name="b")
    model.setObjective(b @ Q @ b, gp.GRB.MINIMIZE)
    model.optimize()
    return np.array(model.x)

# Optimation Function -- Gubrobi QCBO
def _solve_qcbo(cf_train, y_train, k, timeout):
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.params.timelimit = timeout
    model.Params.Threads = 1
    b = model.addMVar(shape=cf_train.shape[1], vtype=gp.GRB.BINARY, name="b")
    norm_0 = model.addVar(lb=k, ub=k, name="norm")
    model.setObjective(b.T @ cf_train.T @ cf_train @ b - 2 * y_train.T @ cf_train @ b + np.dot(y_train, y_train), gp.GRB.MINIMIZE)
    model.addGenConstrNorm(norm_0, b, which=0, name="budget")
    model.optimize()
    return np.array(model.x)[:-1]

# BSSF with Cross-Validation for k
def bssf_cv(y_train, cf_train, cf_pred, alpha, vec_k, timeout, method, kfolds, ran_st, n_jobs = 1):

    """
    Best Subset Selection of Forecasts (BSSF) with k-fold cross-validation for selecting k.

    Parameters:
    - y_train: Training data targets.
    - cf_train: Candidate Model Prediction matrix for training.
    - cf_pred: Candidate Model Prediction matrix for prediction.
    - alpha: Regularization parameter.
    - vec_k: List of values for k to cross-validate.
    - timeout: Timeout for optimization models.
    - method: Optimization method to use ('dwave', 'qubo', 'qcbo').
    - kfolds: Number of folds for cross-validation.
    - ran_st: Random state for cross-validation.
    - n_jobs: Number of jobs to run in parallel.

    Returns:
    Tuple of prediction and solution vector with the best k.
    """

    def _predict(y_train, cf_train, alpha, k, timeout, method, kf):
        scores = []
        for train_index, test_index in kf.split(cf_train):
            cf_train_fold, cf_train_val = cf_train[train_index], cf_train[test_index]
            y_train_fold, y_train_val = y_train[train_index], y_train[test_index]

            pred, _ = bssf(y_train_fold, cf_train_fold, cf_train_val, alpha, k, timeout, method)
            score = mean_squared_error(y_train_val, pred)
            scores.append(score)

        avg_score = np.mean(scores)
        return k, avg_score

    # Initialize
    kf = KFold(n_splits = kfolds, shuffle = True, random_state = ran_st)

    # Parallelize the cross-validation for each k
    results = Parallel(n_jobs = n_jobs)(delayed(_predict)(y_train, cf_train, alpha, k, timeout, method, kf) for k in vec_k)

    # Find the best k and its score
    best_k, _ = min(results, key=lambda x: x[1])

    # Train final model with the best k
    final_pred, final_solution = bssf(y_train, cf_train, cf_pred, alpha, best_k, timeout, method)
    return final_pred, final_solution, best_k

### Complete Subset Regression
# Complete Subset Regression
def csr(y_train, x_train, x_pred, k, sampling):

    """
    Fit and predict complete subset regressions according to Elliott et al. (2013).

    Parameters:
    - y_train: Training data targets.
    - x_train: Training data features.
    - x_pred: Prediction data features.
    - k (int): The number of features to include in each subset regression.
    - sampling (bool): Whether to use sampling for subset selection.

    Returns:
    - predictions: Predictions for x_pred.
    """

    # Validate inputs
    if k > x_train.shape[1]:
        raise ValueError("k cannot be greater than the number of features in x_train.")

    # Sampling Size
    upper_bound = 5000

    # Initialize
    pred_csr = []

    # Indices
    indices = np.arange(x_train.shape[1])

    # Complete Subset Regressions
    if not sampling:
        for comb in combinations(indices, k):
            model = LinearRegression().fit(x_train[:, comb], y_train)
            pred_csr.append(model.predict(x_pred[:, comb]))
    elif sampling:
        for u in range(upper_bound):
            np.random.seed(u)
            comb = tuple(np.random.choice(indices, k, replace = False))
            model = LinearRegression().fit(x_train[:, comb], y_train)
            pred_csr.append(model.predict(x_pred[:, comb]))
    else:
        raise ValueError("Invalid sampling parameter. Choose from True or False.")

    # Return
    return np.mean(pred_csr, axis = 0)

# CSR with Cross-Validation for k
def csr_cv(y_train, x_train, x_pred, vec_k, sampling, kfolds, ran_st, n_jobs = 1):

    """
    Cross-validate to find the best k for the csr function.

    Parameters:
    - y_train: Target data.
    - x_train: Feature data.
    - x_pred: Prediction data.
    - vec_k: A list of k values to test.
    - sampling: Whether to use sampling for subset selection in csr.
    - kfolds: The number of folds for cross-validation.
    - ran_st: Random state for cross-validation.
    - n_jobs: Number of jobs to run in parallel.

    Returns:
    - A dictionary mapping each k to its average cross-validation MSE.
    """

    def _predict(y_train, x_train, k, kf, sampling):
        scores = []
        for train_index, test_index in kf.split(x_train):
            x_train_fold, x_train_val = x_train[train_index], x_train[test_index]
            y_train_fold, y_train_val = y_train[train_index], y_train[test_index]

            pred = csr(y_train_fold, x_train_fold, x_train_val, k, sampling)
            score = mean_squared_error(y_train_val, pred)
            scores.append(score)

        avg_score = np.mean(scores)
        return k, avg_score

    # Initialize
    kf = KFold(n_splits = kfolds, shuffle = True, random_state = ran_st)

    # Parallelize the cross-validation for each k
    results = Parallel(n_jobs = n_jobs)(delayed(_predict)(y_train, x_train, k, kf, sampling) for k in vec_k)

    # Find the best k and its score
    best_k, _ = min(results, key = lambda x: x[1])

    # Train final model with the best k
    final_pred = csr(y_train, x_train, x_pred, best_k, sampling)
    return final_pred, best_k


### Average Best Forecast Combination
# Function to combine the average-best N models
def avg_best(y_train, cf_train, cf_pred, k):

    """
    Calculate the average prediction of the best k models based on mean squared error.

    Parameters:
    - y_train: Actual target values.
    - cf_train: Predictions from candidate models on training data.
    - cf_pred: Predictions from candidate models on prediction data.
    - k: Number of models to average.

    Returns:
    - pred: The average prediction of the k models.
    """

    # Validate inputs
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k > cf_train.shape[1]:
        raise ValueError("k cannot be greater than the number of models")

    # Calculate Squared Errors
    se = (y_train[:, np.newaxis] - cf_train) ** 2

    # Mean-Squared-Error
    mse = np.mean(se, axis=0)

    # Get indices of the average-best N candidate models
    ind = np.argsort(mse)

    # Average over Subset Forecasts
    pred = np.mean(cf_pred[:, ind[:k]], axis=1)

    return pred


# Average Best with Cross-Validation for k
def avg_best_cv(y_train, cf_train, cf_pred, vec_k, kfolds, ran_st, n_jobs = 1):

    """
    Cross-validate to find the best k for the avg_best function.

    Parameters:
    - y_train: Target data.
    - cf_train: Candidate model predictions for training data.
    - cf_pred: Candidate model predictions for prediction data.
    - vec_k: A list of k values to test.
    - kfolds: The number of folds for cross-validation.
    - ran_st: Random state for cross-validation.
    - n_jobs: Number of jobs to run in parallel.

    Returns:
    - A dictionary mapping each k to its average cross-validation MSE.
    """

    def _predict(y_train, cf_train, k, kf):
        scores = []
        for train_index, test_index in kf.split(cf_train):
            cf_train_fold, cf_train_val = cf_train[train_index], cf_train[test_index]
            y_train_fold, y_train_val = y_train[train_index], y_train[test_index]

            # Predict Average-Best Combination
            pred = avg_best(y_train_fold, cf_train_fold, cf_train_val, k)
            score = mean_squared_error(y_train_val, pred)
            scores.append(score)

        avg_score = np.mean(scores)
        return k, avg_score

    # Initialize
    kf = KFold(n_splits = kfolds, shuffle = True, random_state = ran_st)

    # Parallelize the loop over vec_k
    results = Parallel(n_jobs = n_jobs)(delayed(_predict)(y_train, cf_train, k, kf) for k in vec_k)

    # Find the best k and its score
    best_k, _ = min(results, key=lambda x: x[1])

    # Train final model with the best k
    final_pred = avg_best(y_train, cf_train, cf_pred, best_k)
    return final_pred, best_k


### Forward Selection
# Function to calculate Forward Selection Regression
def fss_cv(y_train, x_train, x_pred, kfolds, n_jobs = 1):

    """
    Forward Stepwise Selection (FSS) for feature selection and prediction.

    Parameters:
    - y_train: Training target values.
    - x_train: Training feature matrix.
    - x_pred: Prediction feature matrix.
    - kfolds: Number of cross-validation folds.
    - n_jobs: Number of jobs to run in parallel.

    Returns:
    - pred: Predictions for x_pred.
    """

    ### Model
    model = LinearRegression()
    sfs = SequentialFeatureSelector(model,
                                    n_features_to_select = "auto",
                                    direction = 'forward',
                                    cv = kfolds,
                                    n_jobs = n_jobs)
    active_set = sfs.fit(x_train, y_train).get_support(indices=True)
    model.fit(x_train[:, active_set], y_train)
    
    ### Prediction
    pred = model.predict(x_pred[:, active_set])

    # Return Prediction
    return pred, active_set


### Lasso
# Function to calculate the Lasso
def lasso_cv(y_train, x_train, x_pred, kfolds, n_jobs = 1):

    """
    LASSO to perform feature selection and prediction.

    Parameters:
    - y_train: Training target values.
    - x_train: Training feature matrix.
    - x_pred: Prediction feature matrix.
    - kfolds: Number of cross-validation folds.
    - n_jobs: Number of jobs to run in parallel.

    Returns:
    - pred: Predictions for x_pred.
    """

    ### Model
    model = LassoCV(fit_intercept = True,
                    n_alphas = 100,
                    max_iter = 2000,
                    cv = kfolds,
                    n_jobs = n_jobs)
    model.fit(x_train, y_train)
    active_indices = model.coef_ != 0
    
    ### Prediction
    pred = model.predict(x_pred)

    # Return Prediction
    return pred, model.coef_, np.sum(active_indices)


### Partially-Egalitarian Lasso
# Function to calculate the Partially-Egalitarian Lasso
def pelasso_cv(y_train, cf_train, cf_pred, kfolds, n_jobs = 1):

    """
    Partially-Egalitarian LASSO (peLASSO) for feature selection and prediction.

    Parameters:
    - y_train: Training target values.
    - cf_train: Training feature matrix.
    - cf_pred: Prediction feature matrix.
    - kfolds: Number of cross-validation folds.
    - n_jobs: Number of jobs to run in parallel.

    Returns:
    - pred: Predictions for cf_pred.
    """

    ### Step 1: Select to zero
    model_lasso = LassoCV(fit_intercept = False, #True,
                          n_alphas = 100,
                          max_iter = 2000,
                          cv = kfolds,
                          n_jobs = n_jobs)
    model_lasso.fit(cf_train, y_train)
    active_indices = model_lasso.coef_ != 0
    active_cf_train = cf_train[:, active_indices]
    active_cf_pred  = cf_pred[: , active_indices]

    ### Step 2: Shrink towards equality
    if active_cf_train.shape[1] > 0:
        mean_cf = active_cf_train.mean(axis = 1)
        model_elasso = LassoCV(fit_intercept = False, #True,
                               n_alphas = 100,
                               max_iter = 2000,
                               cv = kfolds,
                               n_jobs = n_jobs)
        model_elasso.fit(active_cf_train, (y_train - mean_cf))
        coefs = model_elasso.coef_ + (1.0 / active_cf_train.shape[1])
        #pred = model_elasso.intercept_ + np.dot(active_cf_pred, coefs)
        pred = np.dot(active_cf_pred, coefs)
    else:
        print("peLASSO: No active candidate models")
        pred = np.full(cf_pred.shape[0], y_train.mean())

    # Return Prediction
    return pred, model_elasso.coef_, np.sum(active_indices)


### Best-Split-Selection
# PSGD
def psgd_cv(y_train, x_train, x_pred, n_models, split_grid, size_grid, kfolds, ran_st = None, n_jobs = 1):

    """
    Best-Split-Selection (BSS) based on the PSGD algorithm.
    This function interfaces with the PSGD R package to perform cross-validation
    and model selection over a grid of hyperparameters.
    """

    # RPY2 Numpy Conversion
    numpy2ri.activate()

    try:

        # Load the PSGD R package
        psgd = importr('PSGD')

        # Set seed if provided
        if ran_st is not None:
            ro.r(f'set.seed({ran_st})')

        # Prepare data for R functions
        y_train_r = numpy2ri.py2rpy(y_train)
        x_train_r = numpy2ri.py2rpy(x_train)
        x_pred_r = numpy2ri.py2rpy(x_pred)
        #split_grid_r = numpy2ri.py2rpy(split_grid)
        #size_grid_r = numpy2ri.py2rpy(size_grid)

        # Ensure Y is a matrix
        y_train_r = ro.r.matrix(y_train_r, nrow=y_train.shape[0], ncol = 1)

        # Convert hyperparameter grids to R vectors
        split_grid_r = ro.FloatVector(split_grid)
        size_grid_r = ro.FloatVector(size_grid)

        # Define group indices for model coefficients
        group_index = ro.IntVector(range(1, n_models + 1))

        # Fast-Best-Split-Selection
        output = psgd.cv_PSGD(
            x = x_train_r, y = y_train_r, n_models = float(n_models),
            model_type = "Linear", include_intercept = True,
            split_grid = split_grid_r, size_grid = size_grid_r,
            max_iter = float(100), cycling_iter = float(0),
            n_folds = float(kfolds), n_threads = float(n_jobs)
        )

        # Extract coefficients and make predictions
        psgd_coef = ro.r['coef'](output, group_index = group_index)
        psgd_predictions = ro.r['predict'](output, newx = x_pred_r, group_index = group_index)

    except Exception as e:
        print(f"Error: {e}")
        psgd_predictions = np.full(x_pred.shape[0], y_train.mean())
        psgd_coef = np.zeros(x_train.shape[1])

    finally:
        # Deactivate numpy2ri
        numpy2ri.deactivate()

    return psgd_predictions, psgd_coef, np.sum(psgd_coef != 0)








#############################################
## CSR with Cross-Validation for k
#def csr_cv(y_train, x_train, x_pred, vec_k, sampling, kfolds, ran_st):
#
#    """
#    Cross-validate to find the best k for the csr function.
#
#    Parameters:
#    - y_train: Target data.
#    - x_train: Feature data.
#    - x_pred: Prediction data.
#    - vec_k: A list of k values to test.
#    - sampling: Whether to use sampling for subset selection in csr.
#    - kfolds: The number of folds for cross-validation.
#    - ran_st: Random state for cross-validation.
#
#    Returns:
#    - A dictionary mapping each k to its average cross-validation MSE.
#    """
#
#    # Initialize
#    kf = KFold(n_splits = kfolds, shuffle = True, random_state = ran_st)
#    best_k = None
#    best_score = float('inf')
#
#    for k in vec_k:
#        scores = []
#        for train_index, test_index in kf.split(x_train):
#            x_train_fold, x_train_val = x_train[train_index], x_train[test_index]
#            y_train_fold, y_train_val = y_train[train_index], y_train[test_index]
#
#            # Use the csr function from the active selection
#            pred = csr(y_train_fold, x_train_fold, x_train_val, k, sampling)
#            score = mean_squared_error(y_train_val, pred)
#            scores.append(score)
#
#        avg_score = np.mean(scores)
#        if avg_score < best_score:
#            best_score = avg_score
#            best_k = k
#
#    # Train final model with the best k
#    final_pred = csr(y_train, x_train, x_pred, best_k, sampling)
#    return final_pred, best_k

# # Average Best with Cross-Validation for k
# def avg_best_cv(y_train, cf_train, cf_pred, vec_k, kfolds, ran_st):
# 
#     """
#     Cross-validate to find the best k for the avg_best function.
# 
#     Parameters:
#     - y_train: Target data.
#     - cf_train: Candidate model predictions for training data.
#     - cf_pred: Candidate model predictions for prediction data.
#     - vec_k: A list of k values to test.
#     - kfolds: The number of folds for cross-validation.
#     - ran_st: Random state for cross-validation.
# 
#     Returns:
#     - A dictionary mapping each k to its average cross-validation MSE.
#     """
# 
#     # Initialize
#     kf = KFold(n_splits = kfolds, shuffle = True, random_state = ran_st)
#     best_k = None
#     best_score = float('inf')
# 
#     # Loop over Combination Size
#     for k in vec_k:
#         scores = []
#         for train_index, test_index in kf.split(cf_train):
#             cf_train_fold, cf_train_val = cf_train[train_index], cf_train[test_index]
#             y_train_fold, y_train_val = y_train[train_index], y_train[test_index]
# 
#             # Predict Average-Best Combination
#             pred = avg_best(y_train_fold, cf_train_fold, cf_train_val, k)
#             score = mean_squared_error(y_train_val, pred)
#             scores.append(score)
# 
#         avg_score = np.mean(scores)
#         if avg_score < best_score:
#             best_score = avg_score
#             best_k = k
# 
#     # Train final model with the best k
#     final_pred = avg_best(y_train, cf_train, cf_pred, best_k)
#     return final_pred, best_k
#
## BSSF with Cross-Validation for k
#def bssf_cv(y_train, cf_train, cf_pred, alpha, vec_k, timeout, method, kfolds, ran_st):
#
#    """
#    Best Subset Selection of Forecasts (BSSF) with k-fold cross-validation for selecting k.
#
#    Parameters:
#    - y_train: Training data targets.
#    - cf_train: Candidate Model Prediction matrix for training.
#    - cf_pred: Candidate Model Prediction matrix for prediction.
#    - alpha: Regularization parameter.
#    - vec_k: List of values for k to cross-validate.
#    - timeout: Timeout for optimization models.
#    - method: Optimization method to use ('dwave', 'qubo', 'qcbo').
#    - kfolds: Number of folds for cross-validation.
#    - ran_st: Random state for cross-validation.
#
#    Returns:
#    Tuple of prediction and solution vector with the best k.
#    """
#
#    # Initialize
#    kf = KFold(n_splits = kfolds, shuffle = True, random_state = ran_st)
#    best_k = None
#    best_score = float('inf')
#
#    for k in vec_k:
#        scores = []
#        for train_index, test_index in kf.split(cf_train):
#            cf_train_fold, cf_train_val = cf_train[train_index], cf_train[test_index]
#            y_train_fold, y_train_val = y_train[train_index], y_train[test_index]
#
#            # Adjust the function call to include cross-validation fold data
#            pred, solution = bssf(y_train_fold, cf_train_fold, cf_train_val, alpha, k, timeout, method)
#            score = mean_squared_error(y_train_val, pred)
#            scores.append(score)
#
#        avg_score = np.mean(scores)
#        if avg_score < best_score:
#            best_score = avg_score
#            best_k = np.sum(solution)
#
#    # Train final model with the best k
#    final_pred, final_solution = bssf(y_train, cf_train, cf_pred, alpha, best_k, timeout, method)
#    return final_pred, final_solution, best_k