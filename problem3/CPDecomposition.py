import numpy as np
from scipy.linalg import khatri_rao

def cp_decomposition(
    X : np.ndarray,
    rank : int = 3,
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
            
            # Get updates factor matrix
            khatri_rao_product = khatri_rao(factors[before], factors[after])
            inverse = np.linalg.pinv(V.T @ V)
            factors[k] = X_k @ khatri_rao_product @ inverse @ V.T

    A1, A2, A3 = factors

    # Get the 1st mode unfolding and refold
    core_1st_mode = A1 @ khatri_rao(A2, A3).T
    core = core_1st_mode.reshape(21, 21, 10)

    return core, factors
