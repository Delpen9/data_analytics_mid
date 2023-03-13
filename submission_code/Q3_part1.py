# Standard Libraries
import os
import numpy as np

# Loading .mat files
import scipy.io
from scipy.linalg import khatri_rao

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

def RSS(
    X : np.ndarray,
    core : np.ndarray
) -> float:
    '''
    '''
    inner = (X - core)**2
    _RSS = np.sum(inner.flatten())
    return _RSS

def k(
    X : np.ndarray,
    rank : int
) -> float:
    '''
    '''
    _k = rank * np.sum(X.shape)
    return _k

def n_aic(
    X : np.ndarray
) -> float:
    '''
    '''
    _naic = np.prod(np.array(list(X.shape)))
    return _naic

def AIC(
    X : np.ndarray,
    core : np.ndarray,
    rank : int
) -> float:
    '''
    '''
    _n_aic = n_aic(X)
    _RSS = RSS(X, core)
    _k = k(X, rank)

    _AIC = _n_aic * np.log(_RSS / _n_aic) + 2*_k

    return _AIC

def relative_error(
    X : np.ndarray,
    core : np.ndarray
) -> float:
    '''
    '''
    assert X.ndim <= 3 and core.ndim <= 3

    X_to_norm = X.reshape((X.shape[0], X.shape[1] * X.shape[2])).copy()
    core_to_norm = core.reshape((core.shape[0], core.shape[1] * core.shape[2])).copy()

    X_frobenius_norm = np.linalg.norm(X_to_norm, 'fro')
    core_frobenius_norm = np.linalg.norm(core_to_norm, 'fro')

    numerator = np.linalg.norm(X_to_norm - core_to_norm, 'fro')
    denominator = X_frobenius_norm

    relative_error = numerator / denominator

    return relative_error

def cp_decomposition(
    X : np.ndarray,
    rank : int = 5,
    max_iteration : int = 50
) -> tuple[np.ndarray, list[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    '''
    I, J, K = X.shape
    A1 = np.random.randn(I, rank)
    A2 = np.random.randn(J, rank)
    A3 = np.random.randn(K, rank)

    A1 = A1 / np.linalg.norm(A1, axis = 0)
    A2 = A2 / np.linalg.norm(A2, axis = 0)
    A3 = A3 / np.linalg.norm(A3, axis = 0)

    factors = [A1, A2, A3]

    for iteration in range(max_iteration):
        for k in range(3):
            before = k - 1
            after = (k + 1) % 3
            V = np.multiply(
                factors[before].T @ factors[before],
                factors[after].T @ factors[after]
            )

            # Get kth mode of the tensor
            X_k = np.moveaxis(X, k, 0)
            X_k = X_k.reshape((X.shape[k], -1))
            
            # Get updates of factor matrices
            khatri_rao_product = khatri_rao(factors[before], factors[after])
            inverse = np.linalg.pinv(V.T @ V)
            factors[k] = X_k @ khatri_rao_product @ inverse @ V.T

    A1, A2, A3 = factors

    # Get the 1st mode unfolding and refold
    core_1st_mode = A1 @ khatri_rao(A2, A3).T
    core = core_1st_mode.reshape(X.shape)

    return core, factors

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

    plt.xlabel('Rank of Reconstructed Tensor')
    plt.ylabel('Relative Error')
    plt.title('Reconstruction Rank Vs. Relative Error')
    plt.xticks(ranks)
    
    output_filepath_kfold = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'reconstruction_rank_vs_relative_error.png'))
    plt.savefig(output_filepath_kfold)

    plt.clf()
    plt.cla()
    # =================


    # Plotting: Plot 2
    # =================
    ax = sns.lineplot(x = ranks, y = aic_scores)

    plt.xlabel('Rank of Reconstructed Tensor')
    plt.ylabel('AIC')
    plt.title('Reconstruction Rank Vs. AIC')
    plt.xticks(ranks)
    
    output_filepath_kfold = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'reconstruction_rank_vs_AIC.png'))
    plt.savefig(output_filepath_kfold)

    plt.clf()
    plt.cla()
    # =================