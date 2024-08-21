##### File for running the simulation #####
# Import
import numpy as np

# Simulate linear, additive data
def sim_linear(n, p, s, snr=1.0):
    """
    Parameters:
    n (int): Number of observations
    p (int): Number of predictors
    s (int): Number of non-zero coefficients
    snr (float): Signal-to-noise ratio

    Returns:
    X (np.ndarray): Design matrix of shape (n, p)
    y (np.ndarray): Response variable of shape (n,)
    beta (np.ndarray): True coefficients of shape (p,)
    """
    # Step 1: Generate covariance matrix Sigma
    Sigma = np.array([[0.5 ** abs(i - j) for j in range(p)] for i in range(p)])

    # Step 2: Generate design matrix X
    X = np.random.multivariate_normal(np.zeros(p), Sigma, n)

    # Step 3: Generate true coefficients beta
    lower_bound = -1.5
    upper_bound = 1.5
    beta = np.zeros(p)
    non_zero_indices = np.arange(s) # np.random.choice(p, s, replace=False)
    beta[non_zero_indices] = np.random.uniform(lower_bound, upper_bound, s)

    # Step 4: Generate response variable y
    y_signal = X @ beta

    # Step 5: Calculate SNR-Adjustment
    signal_variance = np.var(y_signal)
    noise_variance = signal_variance / snr
    sigma = np.sqrt(noise_variance)

    # Step 6: Generate noise and add it to signal
    epsilon = np.random.normal(0, sigma, n)
    y = y_signal + epsilon

    return X, y, np.sort(non_zero_indices)

# Simulate non-linear, additive data
def sim_frd1(n, p, snr=1.0):
    """
    Simulate data according to Friedman (1991) "Multivariate Adaptive Regression Splines".

    Parameters:
    n (int): Number of observations
    p (int): Number of predictors
    snr (float): Signal-to-noise ratio

    Returns:
    X (np.ndarray): Design matrix of shape (n, p)
    y (np.ndarray): Response variable of shape (n,)
    """
    # Step 1: Generate design matrix X
    X = np.random.uniform(0, 1, (n, p))

    ## Step 1: Generate covariance matrix Sigma
    #Sigma = np.array([[0.5 ** abs(i - j) for j in range(p)] for i in range(p)])

    ## Step 2: Generate design matrix X
    #X = np.random.multivariate_normal(np.zeros(p), Sigma, n)

    # Step 2: Generate response variable y without noise
    y_signal = (0.1 * np.exp(4.0 * X[:, 0]) +
                4.0 / (1.0 + np.exp(-10.0 * (X[:, 1] - 0.5))) +
                0.5 ** X[:, 2] +
                X[:, 3] ** 0.5 +
                X[:, 4] ** 2.0)

    # Step 3: Calculate SNR-Adjustment
    signal_variance = np.var(y_signal)
    noise_variance = signal_variance / snr
    sigma = np.sqrt(noise_variance)

    # Step 4: Generate noise and add it to signal
    epsilon = np.random.normal(0, sigma, n)
    y = y_signal + epsilon

    return X, y, np.array([0, 1, 2, 3, 4])

# Simulate non-linear, interactive data
def sim_frd2(n, p, snr=1.0):
    """
    Simulate data according to Friedman (1991) "Multivariate Adaptive Regression Splines".

    Parameters:
    n (int): Number of observations
    p (int): Number of predictors
    snr (float): Signal-to-noise ratio

    Returns:
    X (np.ndarray): Design matrix of shape (n, p)
    y (np.ndarray): Response variable of shape (n,)
    """
    # Step 1: Generate design matrix X
    X = np.random.uniform(0, 1, (n, p))

    ## Step 1: Generate covariance matrix Sigma
    #Sigma = np.array([[0.5 ** abs(i - j) for j in range(p)] for i in range(p)])

    ## Step 2: Generate design matrix X
    #X = np.random.multivariate_normal(np.zeros(p), Sigma, n)

    # Step 2: Generate response variable y without noise
    y_signal = (5.0 * np.sin(np.pi * X[:, 0] * X[:, 1]) +
                10.0 * (X[:, 2] - 0.5) ** 2.0 +
                X[:, 3] ** 0.5 * X[:, 4] ** 2.0)

    # Step 3: Calculate SNR-Adjustment
    signal_variance = np.var(y_signal)
    noise_variance = signal_variance / snr
    sigma = np.sqrt(noise_variance)

    # Step 4: Generate noise and add it to signal
    epsilon = np.random.normal(0, sigma, n)
    y = y_signal + epsilon

    return X, y, np.array([0, 1, 2, 3, 4])

# ...
def sim_comb1(n, p, s, snr=1.0):
    """
    Simulate data where the first 5 signals are generated according to sim_frd1
    and the remaining signals follow the structure of sim_linear.

    Parameters:
    n (int): Number of observations
    p (int): Number of predictors
    s (int): Number of non-zero coefficients for linear simulation
    snr (float): Signal-to-noise ratio

    Returns:
    X_combined (np.ndarray): Combined design matrix of shape (n, p)
    y_combined (np.ndarray): Combined response variable of shape (n,)
    non_zero_indices (np.ndarray): Indices of non-zero coefficients for linear part
    """

    ### Check
    if p < 6:
        raise ValueError("The number of predictors must be at least 6.")
    if s >= p - 5:
        raise ValueError("The number of non-zero coefficients must be less than p - 5.")

    ### Part 1: Friedman 1 for the first 5 signals
    # Generate design matrix
    X1 = np.random.uniform(0, 1, (n, 5))

    # Generate response variable
    y_signal1 = (0.1 * np.exp(4.0 * X1[:, 0]) +
                 4.0 / (1.0 + np.exp(-10.0 * (X1[:, 1] - 0.5))) +
                 0.5 ** X1[:, 2] +
                 X1[:, 3] ** 0.5 +
                 X1[:, 4] ** 2.0)

    ### Part 2: Linear Regression for the remaining signals
    # Generate design matrix
    p_remaining = p - 5
    Sigma = np.array([[0.5 ** abs(i - j) for j in range(p_remaining)] for i in range(p_remaining)])
    X2 = np.random.multivariate_normal(np.zeros(p_remaining), Sigma, n)

    # Generate true coefficients beta
    lower_bound = -1.5
    upper_bound = 1.5
    beta = np.zeros(p_remaining)
    non_zero_indices = np.arange(s) # np.random.choice(p_remaining, s, replace=False)
    beta[non_zero_indices] = np.random.uniform(lower_bound, upper_bound, s)

    # Generate response variable y
    y_signal2 = X2 @ beta

    ### Part 3: Combination
    # Design matrices
    X = np.hstack((X1, X2))

    # Response variables
    y_signal = y_signal1 + y_signal2

    # SNR-Adjustment
    signal_variance = np.var(y_signal)
    noise_variance = signal_variance / snr
    sigma = np.sqrt(noise_variance)

    # Generate noise and add it to signal
    epsilon = np.random.normal(0, sigma, n)
    y = y_signal + epsilon

    # Combine indices
    non_zero_indices = np.concatenate((np.array([0, 1, 2, 3, 4]), non_zero_indices + 5))

    return X, y, np.sort(non_zero_indices)

# ...
def sim_comb2(n, p, s, snr=1.0):
    """
    Simulate data where the first 5 signals are generated according to sim_frd2
    and the remaining signals follow the structure of sim_linear.

    Parameters:
    n (int): Number of observations
    p (int): Number of predictors
    s (int): Number of non-zero coefficients for linear simulation
    snr (float): Signal-to-noise ratio

    Returns:
    X_combined (np.ndarray): Combined design matrix of shape (n, p)
    y_combined (np.ndarray): Combined response variable of shape (n,)
    non_zero_indices (np.ndarray): Indices of non-zero coefficients for linear part
    """

    ### Check
    if p < 6:
        raise ValueError("The number of predictors must be at least 6.")
    if s >= p - 5:
        raise ValueError("The number of non-zero coefficients must be less than p - 5.")

    ### Part 1: Friedman 2 for the first 5 signals
    # Generate design matrix
    X1 = np.random.uniform(0, 1, (n, 5))

    # Generate response variable
    y_signal1 = (5.0 * np.sin(np.pi * X1[:, 0] * X1[:, 1]) +
                 10.0 * (X1[:, 2] - 0.5) ** 2.0 +
                 X1[:, 3] ** 0.5 * X1[:, 4] ** 2.0)

    ### Part 2: Linear Regression for the remaining signals
    # Generate design matrix
    p_remaining = p - 5
    Sigma = np.array([[0.5 ** abs(i - j) for j in range(p_remaining)] for i in range(p_remaining)])
    X2 = np.random.multivariate_normal(np.zeros(p_remaining), Sigma, n)

    # Generate true coefficients beta
    lower_bound = -1.5
    upper_bound = 1.5
    beta = np.zeros(p_remaining)
    non_zero_indices = np.random.choice(p_remaining, s, replace=False)
    beta[non_zero_indices] = np.random.uniform(lower_bound, upper_bound, s)

    # Generate response variable y
    y_signal2 = X2 @ beta

    ### Part 3: Combination 
    # Design matrices
    X = np.hstack((X1, X2))

    # Response variables
    y_signal = y_signal1 + y_signal2

    # SNR-Adjustment
    signal_variance = np.var(y_signal)
    noise_variance = signal_variance / snr
    sigma = np.sqrt(noise_variance)

    # Generate noise and add it to signal
    epsilon = np.random.normal(0, sigma, n)
    y = y_signal + epsilon

    # Combine indices
    non_zero_indices = np.concatenate((np.array([0, 1, 2, 3, 4]), non_zero_indices + 5))

    return X, y, np.sort(non_zero_indices)

########
def sim_data(sim_type, n, p, s=None, snr=1.0):
    """
    Run the specified simulation.

    Parameters:
    sim_type (str): Type of simulation ('linear', 'frd1', 'frd2', 'comb1', 'comb2')
    n (int): Number of observations
    p (int): Number of predictors
    s (int, optional): Number of non-zero coefficients (only for 'linear')
    snr (float): Signal-to-noise ratio

    Returns:
    X (np.ndarray): Design matrix of shape (n, p)
    y (np.ndarray): Response variable of shape (n,)
    indices (np.ndarray): Indices of non-zero coefficients or relevant predictors
    """
    if sim_type == 'linear':
        if s is None:
            raise ValueError("Parameter 's' must be provided for linear simulation.")
        return sim_linear(n, p, s, snr)
    elif sim_type == 'frd1':
        return sim_frd1(n, p, snr)
    elif sim_type == 'frd2':
        return sim_frd2(n, p, snr)
    elif sim_type == 'comb1':
        if s is None:
            raise ValueError("Parameter 's' must be provided for linear simulation.")
        return sim_comb1(n, p, s, snr)
    elif sim_type == 'comb2':
        if s is None:
            raise ValueError("Parameter 's' must be provided for linear simulation.")
        return sim_comb2(n, p, s, snr)
    else:
        raise ValueError("Invalid simulation type. Choose from 'linear', 'frd1', or 'frd2'.")
