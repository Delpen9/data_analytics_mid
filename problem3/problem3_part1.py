# Standard Libraries
import os
import numpy as np

# Loading .mat files
import scipy.io

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Imports
from CPDecomposition import cp_decomposition
from AIC import relative_error, AIC, RSS

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    matlab_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'heat.mat'))

    X = scipy.io.loadmat(matlab_file_path)['X'][0][0][0]
    N = 50
    
    ranks = np.arange(1, 20, 1)

    relative_errors = []
    aic_scores = []
    for rank in ranks:
        core, factors = cp_decomposition(X, rank, N)
        relative_errors.append(relative_error(X, core))
        aic_scores.append(AIC(X, core, rank))

    relative_errors = np.array(relative_errors)
    aic_scores = np.array(aic_scores)

    # Plotting: Plot 1
    # =================
    ax = sns.lineplot(x = ranks, y = relative_errors)

    plt.xlabel('Rank of Reconstructred Tensor')
    plt.ylabel('Relative Error')
    plt.title('Reconstruction Rank Vs. Relative Error')
    plt.xticks(ranks)
    
    output_filepath_kfold = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'reconstruction_rank_vs_relative_error.png'))
    plt.savefig(output_filepath_kfold)

    plt.clf()
    plt.cla()
    # =================


    # Plotting: Plot 1
    # =================
    ax = sns.lineplot(x = ranks, y = aic_scores)

    plt.xlabel('Rank of Reconstructred Tensor')
    plt.ylabel('AIC')
    plt.title('Reconstruction Rank Vs. AIC')
    plt.xticks(ranks)
    
    output_filepath_kfold = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'reconstruction_rank_vs_AIC.png'))
    plt.savefig(output_filepath_kfold)

    plt.clf()
    plt.cla()
    # =================