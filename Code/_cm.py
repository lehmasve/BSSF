###### File for Candidate Model Functions ######
### Import
import numpy as np
#import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn import random_projection
from sklearn.model_selection import KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Lars
import statsmodels.api as sm
from abess.linear import LinearRegression as abessLR
#from itertools import combinations

### Elastic Net
# Function to create the L1-Sequence
def lambda_path(y, X, n_lambda):

    """
    Generates a sequence of lambda values for Lasso-regularization.
    
    Parameters:
    - y: Array of target values.
    - X: 2D array of features.
    - n_lambda: Number of lambda values to generate.
    
    Returns:
    - A numpy array of lambda values.
    """

    #### Solution 1
    ## Validate inputs
    #if n_lambda <= 0 or not isinstance(n_lambda, int):
    #    raise ValueError("n_lambda must be a positive integer")
    #if X.size == 0 or y.size == 0:
    #    raise ValueError("Input arrays cannot be empty")
    #
    ## Standardize X
    #scaler = StandardScaler()
    #sX = scaler.fit_transform(X)
    #
    ## Calculate Lambda Max
    #lambda_max = max(abs(np.dot(y, sX))) / y.shape[0]
    #
    ## Ratio
    #epsilon = 0.001
    #
    ## Lambda Min
    #lambda_min = lambda_max * epsilon
    #
    ## Lambda Path
    #lambdapath = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), num = n_lambda))

    ### Solution 2
    model = ElasticNetCV(n_alphas = n_lambda,
                         l1_ratio = 0.5)
    model.fit(X, y)
    lambdapath = model.alphas_

    #### Solution 3
    ## Equally spaced lambda values
    #lambdapath = np.exp(np.linspace(-15, 15, num = n_lambda))

    # Return
    return np.round(lambdapath, 10)

# Fit and Predict Elastic Nets
def eln(y_train, x_train, x_pred, lambda_vec, alpha_vec, n_jobs=1):

    """
    Fit and predict using Elastic Net for a range of alpha and lambda values.
    
    Parameters:
    - y_train: Array of target values for training.
    - x_train: Training feature matrix.
    - x_pred: Feature matrix for which predictions are to be made.
    - lambda_vec: Array of lambda values to iterate over.
    - alpha_vec: Array of alpha values to iterate over.
    - n_jobs: Number of parallel jobs to run. Defaults to 1 (no parallelism).
    
    Returns:
    - predictions: Array of predictions for each combination of lambda and alpha.
    """

    # Validate inputs
    if lambda_vec.size == 0:
        raise ValueError("lambda_vec must be a non-empty iterable")

    def _predict(l, a):
        if l == 0:
            model = LinearRegression()
            model.fit(x_train, y_train)
            return model, model.predict(x_pred)
        if l != 0 and a == 0:
            model = Ridge(alpha = l, max_iter = 10000, random_state = 42)
            model.fit(x_train, y_train)
            return model, model.predict(x_pred)
        model = ElasticNet(alpha = l, l1_ratio = a, max_iter = 10000, random_state = 42)
        model.fit(x_train, y_train)
        return model, model.predict(x_pred)

    # Generate all combinations of lambda and alpha values
    parameter_combinations = [(l, a) for l in lambda_vec for a in alpha_vec]

    # Parallel execution
    if n_jobs == 1:
        # Execute sequentially if only one job is specified
        models, predictions = zip(*[_predict(l, a) for l, a in parameter_combinations])
    else:
        try:
            models, predictions = zip(*Parallel(n_jobs=n_jobs)(delayed(_predict)(l, a) for l, a in parameter_combinations))
        except Exception as e:
            raise RuntimeError(f"Failed to execute parallel predictions: {e}") from e

    # Combine results
    predictions = np.column_stack(predictions)
    #models = dict(zip(parameter_combinations, models))
    return models, predictions

### Best Subset Selection
# Fit and Predict (fast) Best Subset Selections
def bss(y_train, x_train, x_pred, k_vec, n_jobs=1):

    """
    Fit and predict using (Fast) Best Subset Selection for a range of support values.
    
    Parameters:
    - y_train: Array of target values for training.
    - x_train: Train feature matrix.
    - x_pred: Feature matrix for which predictions are to be made.
    - k_vec: Int array of support values to iterate over (subset size).
    - n_jobs: Number of parallel jobs to run. Defaults to 1 (no parallelism).
    
    Returns:
    - predictions: Array of predictions for each subset size.
    """

    # Validate inputs
    if k_vec.size == 0:
        raise ValueError("k_vec must be a non-empty iterable")

    def _predict(k):
        model = abessLR(fit_intercept = True,
                        support_size = k,
                        thread = 1)
        model.fit(x_train, y_train)
        return model, model.predict(x_pred)

    # Parallel execution
    if n_jobs == 1:
        # Execute sequentially if only one job is specified
        models, predictions = zip(*[_predict(k) for k in k_vec])
    else:
        try:
            models, predictions = zip(*Parallel(n_jobs=n_jobs)(delayed(_predict)(k) for k in k_vec))
        except Exception as e:
            raise RuntimeError(f"Failed to execute parallel predictions: {e}") from e

    # Combine results
    predictions = np.column_stack(predictions)
    #models = dict(zip(k_vec, models))
    return models, predictions

### LARS
# Least Angle Regression
def lars(y_train, x_train, x_pred, k_vec, n_jobs=1):

    """
    Fit and predict using Least Angle Regression.
    
    Parameters:
    - y_train: Array of target values for training.
    - x_train: Train feature matrix.
    - x_pred: Feature matrix for which predictions are to be made.
    - k_vec: Int array of feature numbers to iterate over.
    - n_jobs: Number of parallel jobs to run. Defaults to 1 (no parallelism).
    
    Returns:
    - predictions: Array of predictions for each subset size.
    """

    def _predict(k):
        model = Lars(fit_intercept = True,
                     fit_path = False,
                     n_nonzero_coefs = k)
        model.fit(x_train, y_train)
        return model, model.predict(x_pred)

    # Parallel execution
    if n_jobs == 1:
        # Execute sequentially if only one job is specified
        models, predictions = zip(*[_predict(k) for k in k_vec])
    else:
        try:
            models, predictions = zip(*Parallel(n_jobs=n_jobs)(delayed(_predict)(k) for k in k_vec))
        except Exception as e:
            raise RuntimeError(f"Failed to execute parallel predictions: {e}") from e

    # Combine results
    predictions = np.column_stack(predictions)
    #models = dict(zip(k_vec, models))
    return models, predictions

### Compressed Regression
# Compressed Regression (Gaussian random projection)
def crr(y_train, x_train, x_pred, comp_vec, rep_range, n_jobs=1):

    """
    Perform compressed regression using Gaussian random projection, with optional parallelization.
    
    Parameters:
    - y_train: Target variable for training data.
    - x_train: Feature matrix for training data.
    - x_pred: Feature matrix for prediction data.
    - comp_vec: Vector of component numbers for projection.
    - rep_range: Range of random states for reproducibility.
    - n_jobs: Number of parallel jobs to run. Defaults to 1 (no parallelism).
    
    Returns:
    - predictions: Predicted values for x_pred.
    
    Raises:
    - ValueError: If NA values are found in input arrays or if input arrays are empty.
    """

    def _predict(n_comp, ran_st):

        # Set up Random-Projection-Matrix
        projector = random_projection.GaussianRandomProjection(n_components = n_comp,
                                                               random_state = ran_st)

        # Transform
        x_train_proj = projector.fit_transform(x_train)
        x_pred_proj = projector.fit_transform(x_pred)

        # Add Constant to Projected Data
        rp_train = sm.add_constant(x_train_proj)
        rp_pred = sm.add_constant(x_pred_proj, has_constant = "add")

        # Fit Model
        model = sm.OLS(y_train, rp_train).fit()

        # Predict
        return model, model.predict(rp_pred)

    # Generate all combinations of component numbers and random states
    parameter_combinations = [(n_comp, ran_st) for n_comp in comp_vec for ran_st in rep_range]

    # Parallel execution
    if n_jobs == 1:
        # Execute sequentially if only one job is specified
        models, predictions = zip(*[_predict(n_comp, ran_st) for n_comp, ran_st in parameter_combinations])
    else:
        try:
            models, predictions = zip(*Parallel(n_jobs=n_jobs)(delayed(_predict)(n_comp, ran_st) for n_comp, ran_st in parameter_combinations))
        except Exception as e:
            raise RuntimeError(f"Failed to execute parallel predictions: {e}") from e

    # Combine results
    predictions = np.column_stack(predictions)
    #models = dict(zip(parameter_combinations, models))
    return models, predictions


### Decision Trees
# Simple Decision Tree Regressions
def dtr(y_train, x_train, x_pred, vec_depth, n_jobs=1):

    """
    Fits a Decision Tree to the train data and predicts the outcome for x_pred using the tree.
    
    Parameters:
    - y_train (array-like): The target values for the training data.
    - x_train (array-like): The feature set for the training data.
    - x_pred (array-like): The feature set for which predictions are to be made.
    - vec_depth (list or array-like): The maximum depths of the trees to be explored.
    - n_jobs (int): Number of parallel jobs to run. Defaults to 1 (no parallelism)
    
    Returns:
    A matrix of predictions where each row corresponds to a tree's prediction for x_pred.
    """

    def _predict(max_depth):
        model = DecisionTreeRegressor(criterion = "squared_error",
                                      splitter = "best",
                                      max_depth = max_depth,
                                      max_features = None,
                                      min_samples_split = 12,
                                      min_samples_leaf = 4,
                                      random_state = 42)
        model.fit(x_train, y_train)
        return model, model.predict(x_pred)

    # Parallel execution
    if n_jobs == 1:
        # Execute sequentially if only one job is specified
        models, predictions = zip(*[_predict(depth) for depth in vec_depth])
    else:
        try:
            models, predictions = zip(*Parallel(n_jobs=n_jobs)(delayed(_predict)(depth) for depth in vec_depth))
        except Exception as e:
            raise RuntimeError(f"Failed to execute parallel predictions: {e}") from e

    # Combine results
    predictions = np.column_stack(predictions)
    #models = dict(zip(vec_depth, models)) # list(models)
    return models, predictions

### Univariate Models
# Tree Stump Regressions
def dtrst(y_train, x_train, x_pred, n_jobs=1):

    """
    Fits a univariate Tree Stump to the train data and predicts the outcome for x_pred.
    
    Parameters:
    - y_train (array-like): The target values for the training data.
    - x_train (array-like): The feature set for the training data.
    - x_pred (array-like): The feature set for which predictions are to be made.
    - n_jobs (int): Number of parallel jobs to run. Defaults to 1 (no parallelism)
    
    Returns:
    A matrix of predictions where each row corresponds to a tree's prediction for x_pred.
    """

    def _predict(j):
        model = DecisionTreeRegressor(criterion = "squared_error",
                                      splitter = "best",
                                      max_depth = 1, ###
                                      max_features = None,
                                      random_state = 42)
        model.fit(x_train[:, [j]], y_train)
        return model, model.predict(x_pred[:, [j]])

    # Predictor-Indices
    predictors = list(range(x_train.shape[1]))

    # Parallel execution
    if n_jobs == 1:
        # Execute sequentially if only one job is specified
        models, predictions = zip(*[_predict(j) for j in predictors])
    else:
        try:
            models, predictions = zip(*Parallel(n_jobs=n_jobs)(delayed(_predict)(j) for j in predictors))
        except Exception as e:
            raise RuntimeError(f"Failed to execute parallel predictions: {e}") from e

    # Combine results
    predictions = np.column_stack(predictions)
    #models = dict(zip(predictors, models)) # list(models)
    return models, predictions

### Wrapper
# Wrapper Function
def candidate_models(y_train, x_train, x_pred, models_params, lambda_vec):

    """
    Wrapper function to fit and predict multiple candidate models.
    
    Parameters:
    - y_train: Array of target values for training.
    - x_train: Training feature matrix.
    - x_pred: Feature matrix for which predictions are to be made.
    - models_params: List of tuples, each containing the name of the model and its parameters dictionary.
    - lambda_vec: Array of lambda values for Elastic Net model (...).
    
    Returns:
    - combined_predictions: Array of combined predictions from all models.
    
    Raises:
    - ValueError: If NA values are found in input arrays or if input arrays are empty.
    """

    # Validate inputs
    if any(map(lambda x: np.isnan(x).any() or x.size == 0, [y_train, x_train, x_pred])):
        raise ValueError("Input arrays must not contain NA values or be empty.")
    if x_train.shape[0] != y_train.shape[0] or x_train.shape[1] != x_pred.shape[1]:
        raise ValueError("Mismatch in dimensions of input arrays.")

    # Pre-Process Data
    scaler = StandardScaler()
    sx_train = scaler.fit_transform(x_train)
    sx_pred = scaler.transform(x_pred)

    # Initialize list to store predictions
    all_predictions = []
    all_models = []
    all_descriptions = []

    # Loop over Models and Parameters
    for model_name, params in models_params:

        if model_name == "eln":
            alpha_vec = params.get("alpha_vec", None)
            n_jobs_eln = params.get("n_jobs", 1)
            if lambda_vec is None or alpha_vec is None:
                raise ValueError("Missing required parameters for Elastic Net model")
            if lambda_vec.size == 0 or alpha_vec.size == 0:
                raise ValueError("lambda_vec and alpha_vec must be non-empty iterables")
            descriptions = [f"ELN_L{l}_A{a}" for l in lambda_vec for a in alpha_vec]
            models, predictions = eln(y_train, sx_train, sx_pred, lambda_vec, alpha_vec, n_jobs_eln)

        elif model_name == "bss":
            k_vec = params.get("k_vec", None)
            n_jobs_bss = params.get("n_jobs", 1)
            if k_vec is None:
                raise ValueError("Missing required parameters for Best Subset Selection model")
            if k_vec.size == 0:
                raise ValueError("k_vec must be a non-empty iterable")
            descriptions = [f"BSS_K{k}" for k in k_vec]
            models, predictions = bss(y_train, sx_train, sx_pred, k_vec, n_jobs_bss)

        elif model_name == "lars":
            k_vec = params.get("k_vec", None)
            n_jobs_lars = params.get("n_jobs", 1)
            if k_vec is None:
                raise ValueError("Missing required parameters for Least Angle Regression model")
            if k_vec.size == 0:
                raise ValueError("k_vec must be a non-empty iterable")
            descriptions = [f"LARS_K{k}" for k in k_vec]
            models, predictions = lars(y_train, sx_train, sx_pred, k_vec, n_jobs_lars)

        elif model_name == "crr":
            comp_vec = params.get("comp_vec", None)
            rep_range = params.get("rep_range", None)
            n_jobs_crr = params.get("n_jobs", 1)
            if comp_vec is None or rep_range is None:
                raise ValueError("Missing required parameters for Compressed Regression model")
            if comp_vec.any() <= 0 or len(comp_vec) == 0:
                raise ValueError("comp_vec must be a non-empty iterable.")
            if rep_range.any() <= 0 or len(rep_range) == 0:
                raise ValueError("rep_range must be a non-empty iterable.")
            descriptions = [f"CRR_K{n_comp}_N{ran_st}" for n_comp in comp_vec for ran_st in rep_range]
            models, predictions = crr(y_train, sx_train, sx_pred, comp_vec, rep_range, n_jobs_crr)

        elif model_name == "dtr":
            vec_depth = params.get("vec_depth", None)
            n_jobs_dtr = params.get("n_jobs", 1)
            if vec_depth is None:
                raise ValueError("Missing required parameters for Decision Tree model")
            if vec_depth.any() <= 0 or len(vec_depth) == 0:
                raise ValueError("vec_depth must be a non-empty iterable.")
            descriptions = [f"TREE_D{depth}" for depth in vec_depth]
            models, predictions = dtr(y_train, sx_train, sx_pred, vec_depth, n_jobs_dtr)

        elif model_name == "dtrst":
            n_jobs_dtrst = params.get("n_jobs", 1)
            descriptions = [f"TREEST_J{j}" for j in range(sx_train.shape[1])]
            models, predictions = dtrst(y_train, sx_train, sx_pred, n_jobs_dtrst)

        else:
            raise ValueError(f"Model {model_name} is not supported.")

        # Append predictions
        all_models.extend(models) # all_models.update(models)
        all_predictions.append(predictions)
        all_descriptions.extend(descriptions)

    # Combine predictions
    return all_models, np.column_stack(all_predictions), all_descriptions

### K-Fold Candidate Models
# Function to generate predictions for each fold
def candidate_models_kf(y, X, kfolds, models_params, ran_st, n_jobs=1):

    """
    Function to generate predictions for each fold using candidate models.
    
    Parameters:
    - y: Target variable.
    - X: Feature matrix.
    - kfolds: Positive integer, number of folds for cross-validation.
    - models_params: List of tuples, each containing the name of the model and its parameters dictionary.
    - ran_st: Random state for reproducibility.
    - n_jobs: Integer, number of parallel jobs to run. Defaults to 1 (no parallelism).
    
    Returns:
    - y_test: Array of true values for each fold.
    - predictions: Array of predictions for each fold.
    """

    # Validate kfolds
    if not isinstance(kfolds, int) or kfolds < 1:
        raise ValueError("kfolds must be a positive integer")

    # Validate models_params
    if not models_params or not all(isinstance(mp, tuple) and len(mp) == 2 for mp in models_params):
        raise ValueError("models_params must be a non-empty list of tuples (model_name, params_dict)")

    # Validate Cores
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")

    # Set up K-Fold
    kf = KFold(n_splits = kfolds, shuffle = True, random_state = ran_st)

    # Get Lambda-Path
    lambda_vec = None
    if any(model_name == "eln" for model_name, _ in models_params):
        lambda_vec = next((lambda_path(y, X, params.get("n_lambda")) for model_name, params in models_params if model_name == "eln"), None)

    # Function to predict for a single fold
    def _predict_fold(train_index, test_index):
        y_train, y_test = y[train_index], y[test_index]
        x_train, x_test = X[train_index], X[test_index]
        _, predictions, _ = candidate_models(y_train, x_train, x_test, models_params, lambda_vec)
        return y_test, predictions

    # Execute predictions in parallel
    if n_jobs == 1:
        # Execute sequentially if only one job is specified
        y_test, predictions = zip(*[_predict_fold(train_index, test_index) for train_index, test_index in kf.split(X)])
    else:
        try:
            y_test, predictions = zip(*Parallel(n_jobs=n_jobs)(delayed(_predict_fold)(train_index, test_index) for train_index, test_index in kf.split(X)))
        except Exception as e:
            raise RuntimeError(f"Failed to execute parallel predictions: {e}") from e

    # Combine Results
    y_test = np.concatenate(y_test, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    return y_test, predictions, lambda_vec



#### --- END --- ####
#-------------------------------------------------#
#### Forward Selection
## Forward Selection
#def fss(y_train, x_train, x_pred, k_vec, n_jobs=1):
#
#    """
#    Fit and predict using Forward Selection for a range of feature numbers.
#    
#    Parameters:
#    - y_train: Array of target values for training.
#    - x_train: Train feature matrix.
#    - x_pred: Feature matrix for which predictions are to be made.
#    - k_vec: Int array of feature numbers to iterate over.
#    - n_jobs: Number of parallel jobs to run. Defaults to 1 (no parallelism).
#    
#    Returns:
#    - predictions: Array of predictions for each subset size.
#    """
#
#    # Validate inputs
#    if k_vec.size == 0:
#        raise ValueError("k_vec must be a non-empty iterable")
#
#    def _predict(k):
#        model = LinearRegression()
#        sfs = SequentialFeatureSelector(model,
#                                        n_features_to_select = k,
#                                        direction = 'forward')
#        active_set = sfs.fit(x_train, y_train).get_support()
#        model.fit(x_train[:, active_set], y_train)
#        return model, model.predict(x_pred[:, active_set])
#
#    # Parallel execution
#    if n_jobs == 1:
#        # Execute sequentially if only one job is specified
#        models, predictions = zip(*[_predict(k) for k in k_vec])
#    else:
#        try:
#            models, predictions = zip(*Parallel(n_jobs=n_jobs)(delayed(_predict)(k) for k in k_vec))
#        except Exception as e:
#            raise RuntimeError(f"Failed to execute parallel predictions: {e}") from e
#
#    # Combine results
#    predictions = np.column_stack(predictions)
#    models = dict(zip(k_vec, models))
#    return models, predictions
#
### Elasti Net
# Elastic Net
#def eln(y_train, x_train, x_pred, lambda_vec, alpha_vec, n_iter):
#    
#    # Check NA-Values
#    if np.isnan(y_train).any() or np.isnan(x_train).any() or np.isnan(x_pred).any():
#        raise ValueError("NA values found in input arrays")
#    
#    # Set up Array
#    predictions = np.full((len(lambda_vec) * len(alpha_vec), x_pred.shape[0]), np.nan)
#    
#    # Init Counter
#    i = 0 
#
#    # Loop over Parameters
#    for l in lambda_vec:
#        for a in alpha_vec:
#            
#            # Define Model
#            model = ElasticNet(fit_intercept = True,
#                               alpha = l,
#                               l1_ratio = a, 
#                               max_iter = n_iter,
#                               warm_start = False, 
#                               random_state = 123)
#            
#            # Fit Model
#            model.fit(x_train, y_train)
#            
#            # Predict
#            predictions[i] = model.predict(x_pred)
#            
#            # Update
#            i += 1
#    
#    # Return
#    return predictions
#
### Compressed Regression
# def cr_reg(y_train, x_train, x_pred, comp_vec, rep_range):
#     
#     """
#     Perform compressed regression using Gaussian random projection.
#     
#     Parameters:
#     - y_train: Target variable for training data.
#     - x_train: Feature matrix for training data.
#     - x_pred: Feature matrix for prediction data.
#     - comp_vec: Vector of component numbers for projection.
#     - rep_range: Range of random states for reproducibility.
#     
#     Returns:
#     - predictions: Predicted values for x_pred.
#     """
#     
#     # Validate inputs
#     if any(map(lambda x: np.isnan(x).any() or x.size == 0, [y_train, x_train, x_pred])):
#         raise ValueError("Input arrays must not contain NA values or be empty.")
#     if x_train.shape[0] != y_train.shape[0]:
#         raise ValueError("x_train and y_train must have the same number of rows.")
#     
#     # Initialize predictions array
#     num_predictions = len(comp_vec) * len(rep_range)
#     predictions = np.full((num_predictions, x_pred.shape[0]), np.nan)
#     
#     # Init Counter
#     ctr = 0 
# 
#     # Loop over Tuning Parameters
#     for n_comp in comp_vec:
#         for ran_st in rep_range:
#     
#             # Set up Random-Projection-Matrix
#             projector = random_projection.GaussianRandomProjection(n_components = n_comp, random_state = ran_st)
# 
#             # Transform
#             x_train_proj = projector.fit_transform(x_train)
#             x_pred_proj  = projector.fit_transform(x_pred)
# 
#             # Add Constant to Projected Data
#             rp_train = sm.add_constant(x_train_proj)
#             rp_pred  = sm.add_constant(x_pred_proj, has_constant = 'add')
# 
#             # Fit Model
#             model  =  sm.OLS(y_train, rp_train).fit()
# 
#             # Predict
#             predictions[ctr] = model.predict(rp_pred)
#             
#             # Update Counter
#             ctr += 1
#             
#     # Return Predictions
#     return predictions
# #
# ### Subset Regressions
# # Function to return array of all subsets of length k
# def complete_sub(arr, k):
#     
#     """
#     Elements are treated as unique based on their position, not on their value.
#     So if the input elements are unique, there will be no repeated values in each combination.
#     """
#     
#     # Get all subsets of size k
#     subset = list(combinations(arr, k)) 
#     
#     # Return 
#     return subset 
# 
# # Function to calculate number of models
# def n_models(K, k):
#     
#     """
#     Function to calculate the number of models
#     """
#     
#     return math.factorial(K) / (math.factorial(k) * math.factorial(K-k))
# 
# # Function to randomly select n_max items from array
# def random_select(arr, n_max, ran_st):
#     
#     """
#     Function to randomly select n_max items from array
#     """
#     
#     # Set random state
#     random.seed(ran_st)
#     
#     # Set upper Boundary
#     upper_bound  =  len(arr) if len(arr) < n_max else n_max
#     
#     # Randomly select items without repetition
#     rand_arr  =  random.sample(arr, k = upper_bound)
#     
#     # Return 
#     return rand_arr
# 
# # Function to produce Subset Regression Forecasts
# def ssf(y_train, x_train, x_pred, feature):
#     
#     """
#     Function to fit and predict subset regression models
#     """
#     
#     # Subset Feature Space (incl. constant)
#     x_train_subset = x_train[:, list(range(0, 1)) + list(feature)]
#     x_pred_subset  = x_pred[:, list(range(0, 1)) + list(feature)]
#     
#     # Fit Model
#     model =  sm.OLS(y_train, x_train_subset) 
#     regr  =  model.fit()
#     
#     # Predict
#     pred = regr.predict(x_pred_subset)
#     
#     return(pred, regr.params[1:])