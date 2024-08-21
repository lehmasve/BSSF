############## ---------------------------
### VISUALIZATION FUNCTION
# Import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from collections import Counter

from _helpers import relevant_predictor

# Function to plot the BSSF-Subset-Size Distribution
def plot_subsetsize(best_k, max_k, N):
    """
    Plots the distribution of the best subset sizes.

    Parameters:
    best_k (list of lists): List containing the best subset sizes.
    max_k (int): Maximum subset size.
    N (int): Number of repetitions.
    """
    # Extract the last element from each list in best_k
    bssf_k = [lst[-1] for lst in best_k]

    # Count the frequency of each subset size
    freq = Counter(bssf_k)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(freq.keys(), freq.values(), color='skyblue', width=0.5)
    plt.xlabel('Subset Size (k)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'BSSF - Subset Size Distribution ({N} reps)', fontsize=16)
    plt.xlim(0, max_k + 1)
    plt.xticks(range(1, max_k + 1))
    plt.ylim(0, N + 1)
    #plt.yticks(range(1, N + 1))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plot the BSSF-Selected Candidate Models
def plot_cm(bssf_weights, cf_descriptions, N):
    """
    Plots the frequency of selected candidate models.

    Parameters:
    bssf_weights (numpy array): Array containing the weights of the best subset sizes.
    cf_descriptions (list of str): List containing the descriptions of candidate models.
    N (int): Number of repetitions.
    """
    # Sum the weights for each candidate model
    sum_bssf_weights = np.sum(bssf_weights, axis=0)
    frequencies = [sum_bssf_weights[i] for i in range(len(cf_descriptions))]

    # Define a set of colors
    color_set = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple', 'orange', 'pink', 'brown']

    # Assign a color to each group
    color_index = 0
    group_colors = {}
    for model in cf_descriptions:
        group = model.split('_')[0]
        if group not in group_colors:
            group_colors[group] = color_set[color_index % len(color_set)]
            color_index += 1
    colors = [group_colors.get(name.split('_')[0], 'gray') for name in cf_descriptions]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(cf_descriptions, frequencies, color=colors, width=0.5)
    plt.xlabel('Candidate Models', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'BSSF - Selected Candidate Models ({N} reps)', fontsize=16)
    plt.ylim(0, N + 1)
    #plt.yticks(range(1, N+1))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Rotate xtick labels
    plt.xticks(rotation=90)

    # Create legend
    legend_patches = [mpatches.Patch(color=color, label=group) for group, color in group_colors.items()]
    plt.legend(handles=legend_patches, title='Candidate Models')

    plt.show()

# Function to plot the BSSF-Selected Predictors
def plot_preds(cf_models, bssf_weights, p, N):
    """
    Plots the frequency of selected predictors.

    Parameters:
    cf_models (list): List of candidate models.
    bssf_weights (numpy array): Array containing the weights of the best subset sizes.
    p (int): Number of predictors.
    N (int): Number of repetitions.
    """
    # Get the relevant predictors
    idx = relevant_predictor(cf_models, bssf_weights)
    # Flatten the list of indices
    list_idx = [index for sublist in idx.values() for index in sublist]
    # Count the frequency of each predictor
    freq = Counter(list_idx)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(freq.keys(), freq.values(), color='skyblue', width=0.75)
    plt.xlabel('Predictor', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'Selected Predictors ({N} reps)', fontsize=16)
    plt.xlim(0, p)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
