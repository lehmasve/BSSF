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
def train_test_split(indices, train_size, x, y):
    """
    Splits the data into training and test sets and prepares the corresponding subsets.
    
    Parameters:
    - indices: Array of indices to split.
    - train_size: Number of observations to include in the training set.
    - x: DataFrame of predictors.
    - y: Series or array of response variable.
    
    Returns:
    - x_train: Training set predictors.
    - y_train: Training set response variable.
    - x_test: Test set predictors.
    - y_test: Test set response variable.
    """
    # Randomly select indices for training set
    train_indices = np.random.choice(indices, train_size, replace=False)

    # Remaining indices are used for test set
    test_indices = np.setdiff1d(indices, train_indices)
    return x[train_indices].copy(), y[train_indices].copy(), x[test_indices].copy(), y[test_indices].copy()

############## ---------------------------
### Function to generate results
def run_results(N, x, y, train, cm_params, bssf_timeout, bssf_alpha, ran_st):
    """
    Function for real data analysis simulation.
    
    Parameters:
    - N: Number of runs.
    - x: DataFrame of predictors.
    - y: Series or array of response variable.
    - train: Number of observations to include in the training set.
    - cm_params: List of tuples with the candidate models and their parameters.
    - ran_st: Seed for random number generator 
    
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

    # Set seed for reproducibility
    np.random.seed(ran_st)

    # Full Index
    indices = np.arange(0, x.shape[0])

    # Initialize results dictionary
    results = {
        "predictions": [None] * N,
        "mse": [None] * N,
        "best_k": [None] * N,
        "bssf_weights": [None] * N,
        "cf_models": [None] * N,
        "cf_descriptions": None,
        "fmodel_names": fmodel_names,
        "bssf_timeout": bssf_timeout,
        "bssf_alpha": bssf_alpha
    }

    ### Simulation replications
    for rep_ind in range(1, N + 1):
        print(f"Iteration: {rep_ind}")

        # Split Data
        x_train, y_train, x_test, y_test = train_test_split(indices, train, x, y)

        # Check
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        assert x_train.shape[0] == train

        ### Candidate Models
        lambda_vec = None
        kfolds = train ### LOOCV

        # Candidate models -- Train
        target_train, cf_train, lambda_vec = candidate_models_kf(y_train, x_train, kfolds, cm_params, ran_st = rep_ind, n_jobs = 5)

        # Candidate models -- Test
        cf_models, cf_test, cf_descriptions = candidate_models(y_train, x_train, x_test, cm_params, lambda_vec)

        # Check
        assert cf_train.shape[0] == x_train.shape[0]
        assert cf_test.shape[0] == x_test.shape[0]
        assert cf_train.shape[0] == train

        ### Benchmark Methods
        kfolds = 5

        # PHM
        pred_phm = np.full(y_test.shape[0], target_train.mean())

        ## Forward Stepwise Regression
        #pred_fss, fss_k = fss_cv(y_train, x_train, x_test, kfolds, n_jobs = 4)
        
        # Lasso
        pred_lasso, lasso_coef, lasso_k = lasso_cv(y_train, x_train, x_test, kfolds, n_jobs = 4)

        # peLasso
        pred_pelasso, pelasso_coef, pelasso_k = pelasso_cv(target_train, cf_train, cf_test, kfolds, n_jobs = 4)

        # Best Average
        vec_k = np.array([1, 2, 3, 4, 5])
        pred_avg_best, avg_best_k = avg_best_cv(target_train, cf_train, cf_test, vec_k, kfolds, ran_st = rep_ind, n_jobs = 1)

        # Complete Subset Regression
        vec_k = np.arange(1, 10)
        sampling = True
        pred_csr, csr_k = csr_cv(y_train, x_train, x_test, vec_k, sampling, kfolds, ran_st = rep_ind, n_jobs = 4)
     
        # Fast-Best-Split-Selection - Simple Signals
        n_models = 5
        split_grid = np.array([1, 2, 3, 4, 5])
        size_grid = np.array([9, 12, 15]) # np.floor(np.array([0.3 * x_train.shape[0], 0.4 * x_train.shape[0], 0.5 * x_train.shape[0]]))
        pred_psgd, psgd_coef, psgd_k = psgd_cv(y_train, x_train, x_test, n_models, split_grid, size_grid, kfolds, n_jobs = 4)

        # BSSF
        alpha = bssf_alpha # 1e9
        vec_k = np.array([1, 2, 3, 4, 5])
        timeout = bssf_timeout # 10
        method = "gurobi"
        pred_bssf, bssf_weights, bssf_k = bssf_cv(target_train, cf_train, cf_test, alpha, vec_k, timeout, method, kfolds, ran_st = rep_ind)

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
        results["predictions"][rep_ind - 1] = [y_test, pred_phm, pred_lasso, pred_pelasso, pred_avg_best, pred_csr, pred_psgd, pred_bssf]
        results["mse"][rep_ind - 1] = [mse_phm, mse_lasso, mse_pelasso, mse_avg_best, mse_csr, mse_psgd, mse_bssf]
        results["best_k"][rep_ind - 1] = [None, lasso_k, pelasso_k, avg_best_k, csr_k, psgd_k, bssf_k]
        results["bssf_weights"][rep_ind - 1] = bssf_weights
        results["cf_models"][rep_ind - 1] = cf_models
        results["cf_descriptions"] = cf_descriptions

    return results