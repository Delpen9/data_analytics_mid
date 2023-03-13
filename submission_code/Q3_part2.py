# Standard Libraries
import os
import numpy as np

# Loading .mat files
import scipy.io

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

from tensorly.tenalg import mode_dot

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

def cur_decomposition(
    X : np.ndarray,
    rc : int = 5,
    rr : int = 5
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    '''
    N1, N2, N3 = X.shape

    slices = np.random.choice(N3, rc, replace = True)
    C = X[:, :, slices]

    D_c = np.eye(rc) * np.sqrt(N3 / rc)

    fibers = np.random.choice(N1 * N2, rr, replace = True)

    X_3 = np.reshape(np.moveaxis(X, 2, 0), (X.shape[2], -1))

    R = X_3[:, fibers].T

    D_r = np.eye(rr) * np.sqrt(N1 * N2 / rr)

    W = np.expand_dims(X_3[slices, :][:, fibers], axis = 0).T
    
    tensor_product = np.dot(np.dot(D_r, W).T, D_c)

    moore_penrose_pseudoinverse = np.linalg.pinv(tensor_product)

    U = np.dot(np.dot(D_c, moore_penrose_pseudoinverse), D_r)
    U = U.reshape((rc, rr))

    X_reconstructed = mode_dot(C, np.dot(U, R).T, mode = 2)
    factors = (C, U, R)

    return X_reconstructed, factors

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    matlab_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'heat.mat'))

    X = scipy.io.loadmat(matlab_file_path)['X'][0][0][0]

    ranks = np.arange(1, 20, 1)

    relative_errors = []
    aic_scores = []
    for rank in ranks:
        X_reconstructed, factors = cur_decomposition(X, rank, rank)
        relative_errors.append(relative_error(X, X_reconstructed))
        aic_scores.append(AIC(X, X_reconstructed, rank))

    relative_errors = np.array(relative_errors)
    aic_scores = np.array(aic_scores)

    # Plotting: Plot 1
    # =================
    ax = sns.lineplot(x = ranks, y = relative_errors)

    plt.xlabel('Rank of Reconstructed Tensor')
    plt.ylabel('Relative Error')
    plt.title('Reconstruction Rank Vs. Relative Error')
    plt.xticks(ranks)
    
    output_filepath_kfold = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'cur_decomposition_reconstruction_rank_vs_relative_error.png'))
    plt.savefig(output_filepath_kfold)

    plt.clf()
    plt.cla()
    # =================


    # Plotting: Plot 2
    # =================
    ax = sns.lineplot(x = ranks, y = aic_scores)
    ax.set_ylim([-30000, 5000])

    plt.xlabel('Rank of Reconstructed Tensor')
    plt.ylabel('AIC')
    plt.title('Reconstruction Rank Vs. AIC')
    plt.xticks(ranks)
    
    output_filepath_kfold = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'cur_decomposition_reconstruction_rank_vs_AIC.png'))
    plt.savefig(output_filepath_kfold)

    plt.clf()
    plt.cla()
    # =================