############## ---------------------------
### HELPER FUNCTION
# Importing Libraries
import numpy as np

from sklearn.metrics import mean_squared_error

from _cm import candidate_models, candidate_models_kf
from _fm import bssf_cv, lasso_cv, csr_cv, avg_best_cv, pelasso_cv, psgd_cv

# Function to get relevant predictor from model
def relevant_predictor(models, weights):
    """
    Extracts the relevant predictors from the models.
    
    Parameters:
    - models: List of lists of candidate models.
    - weights: List of lists of weights.
    
    Returns:
    - relevant_indices: Dictionary with the indices of the relevant predictors.
    """
    # Initialize dict
    relevant_indices = {}

    for i, sublist in enumerate(models):
        for j, model in enumerate(sublist):
            if weights[i][j] == 1:
                if hasattr(model, 'coef_'):
                    # For linear models
                    indices = np.nonzero(model.coef_)[0].tolist() # [k for k, coef in enumerate(model.coef_) if coef != 0]
                elif hasattr(model, 'feature_importances_'):
                    # For tree-based models
                    indices = np.nonzero(model.feature_importances_)[0].tolist() #[k for k, importance in enumerate(model.feature_importances_) if importance > 0]
                else:
                    indices = []
                relevant_indices[(i, j)] = indices
    return relevant_indices

############## ---------------------------
# Function to split data
def train_test_split(train_size, x, y, ran_st=None):
    """
    Splits the data into training and test sets and prepares the corresponding subsets.
    
    Parameters:
    - train_size: Number of observations to include in the training set.
    - x: DataFrame of predictors.
    - y: Series or array of response variable.
    - ran_st: Seed for random number generator.
    
    Returns:
    - x_train: Training set predictors.
    - y_train: Training set response variable.
    - x_test: Test set predictors.
    - y_test: Test set response variable.
    """
    # Set seed for reproducibility
    np.random.seed(ran_st)

    # Full-Sample-Indices
    indices = np.arange(0, y.shape[0])

    # Randomly select indices for training set
    train_indices = np.random.choice(indices, train_size, replace=False)

    # Remaining indices are used for test set
    test_indices = np.setdiff1d(indices, train_indices)
    return x[train_indices].copy(), y[train_indices].copy(), x[test_indices].copy(), y[test_indices].copy()

############## ---------------------------
### Function to generate results
def run_results(N, x, y, train, cm_params, bssf_timeout, n_threads):
    """
    Function for real data analysis simulation.
    
    Parameters:
    - N: Number of runs.
    - x: DataFrame of predictors.
    - y: Series or array of response variable.
    - train: Number of observations to include in the training set.
    - cm_params: List of tuples with the candidate models and their parameters.
    - bssf_timeout: Timeout for BSSF model.
    - n_threads: Number of threads to use.
    
    Returns:
    - output_pred: List of predictions for each model.
    - output_mse: List of mean squared errors for each model.
    - output_bestk: List of best k values for each model.
    - output_weights: List of weights for BSSF model.
    - output_cfmodels: List of candidate models.
    - cf_descriptions: Descriptions of candidate models.
    - fmodel_names: List of model names.
    """

    # Forecasting Model Names
    fmodel_names = ["PHM", "LASSO", "PELASSO", "AVG_BEST", "CSR", "PSGD", "BSSF"]

    # Initialize results dictionary
    results = {
        "y_test": [None] * N,
        "predictions": [None] * N,
        "mse": [None] * N,
        "best_k": [None] * N,
        "bssf_weights": [None] * N,
        "cf_models": [None] * N,
        "cf_descriptions": None,
        "fmodel_names": fmodel_names,
        "bssf_alpha": [None] * N,
        "bssf_timeout": bssf_timeout
    }

    ### Re-Sampling
    for r in range(1, N + 1):

        if N != 1:
            print(f"Iteration: {r}")

        # Split Data
        x_train, y_train, x_test, y_test = train_test_split(train, x, y, r)

        # Check
        assert x_train.shape[0] == train
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]

        ### Candidate Models
        lambda_vec = None
        kfolds = train ### -> LOOCV

        # Candidate models -- Train
        target_train, cf_train, lambda_vec = candidate_models_kf(y_train, x_train, kfolds, cm_params, r, n_threads)

        # Candidate models -- Test
        cf_models, cf_test, cf_descriptions = candidate_models(y_train, x_train, x_test, cm_params, lambda_vec)

        # Check
        assert cf_train.shape[0] == train
        assert cf_train.shape[0] == x_train.shape[0]
        assert cf_test.shape[0] == x_test.shape[0]

        ### Forecasting Methods
        kfolds = 10

        # PHM
        pred_phm = np.full(y_test.shape[0], target_train.mean())

        ## Forward Stepwise Regression
        #pred_fss, fss_k = fss_cv(y_train, x_train, x_test, kfolds, n_threads)

        # Lasso
        pred_lasso, lasso_coef, lasso_k = lasso_cv(y_train, x_train, x_test, kfolds, n_threads)

        # peLasso
        pred_pelasso, pelasso_coef, pelasso_k = pelasso_cv(target_train, cf_train, cf_test, kfolds, n_threads)

        # Best Average
        vec_k = np.array([1, 2, 3, 4, 5])
        pred_avg_best, avg_best_k = avg_best_cv(target_train, cf_train, cf_test, vec_k, kfolds, r, n_threads)

        # Complete Subset Regression
        vec_k = np.arange(1, 10)
        sampling = True
        pred_csr, csr_k = csr_cv(y_train, x_train, x_test, vec_k, sampling, kfolds, r, n_threads)

        # Fast-Best-Split-Selection - Simple Signals
        n_models = 5
        split_grid = np.array([1, 2, 3, 4, 5])
        size_grid = np.array([9, 12, 15]) # np.floor(np.array([0.3 * x_train.shape[0], 0.4 * x_train.shape[0], 0.5 * x_train.shape[0]]))
        pred_psgd, psgd_coef, psgd_k = psgd_cv(y_train, x_train, x_test, n_models, split_grid, size_grid, kfolds, r, n_threads)

        # BSSF
        vec_k = np.array([1, 2, 3, 4, 5])
        alpha = max(0.01, np.round(np.max(vec_k) * np.var(target_train), 1)) # bssf_alpha # 1e9
        timeout = bssf_timeout # 10
        method = "gurobi"
        pred_bssf, bssf_weights, bssf_k = bssf_cv(target_train, cf_train, cf_test, alpha, vec_k, timeout, method, kfolds, r, n_threads)

        ### Evaluation
        # Mean-Squared Error
        mse_phm = mean_squared_error(y_test, pred_phm)
        #mse_fss = mean_squared_error(y_test, pred_fss)
        mse_lasso = mean_squared_error(y_test, pred_lasso)
        mse_pelasso = mean_squared_error(y_test, pred_pelasso)
        mse_avg_best = mean_squared_error(y_test, pred_avg_best)
        mse_csr = mean_squared_error(y_test, pred_csr)
        mse_psgd = mean_squared_error(y_test, pred_psgd)
        mse_bssf = mean_squared_error(y_test, pred_bssf)

        # Fill Results
        results["y_test"][r - 1] = y_test
        results["predictions"][r - 1] = [y_test, pred_phm, pred_lasso, pred_pelasso, pred_avg_best, pred_csr, pred_psgd, pred_bssf]
        results["mse"][r - 1] = [mse_phm, mse_lasso, mse_pelasso, mse_avg_best, mse_csr, mse_psgd, mse_bssf]
        results["best_k"][r - 1] = [None, lasso_k, pelasso_k, avg_best_k, csr_k, psgd_k, bssf_k]
        results["bssf_weights"][r - 1] = bssf_weights
        results["cf_models"][r - 1] = cf_models
        results["cf_descriptions"] = cf_descriptions
        results["bssf_alpha"][r - 1] = alpha

    return results