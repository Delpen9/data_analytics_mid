import numpy as np
from tensorly.tenalg import mode_dot

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
