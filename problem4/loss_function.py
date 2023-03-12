import numpy as np

def _sigma(
    x : np.ndarray
) -> np.ndarray:
    '''
    '''
    _sigma = 1 / (1 + np.e**(-x))
    return _sigma

def _partial_w(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray
) -> float:
    '''
    '''
    _z_i1 = w[0] * np.expand_dims(_sigma(alpha * x), axis = 0).T
    _z_i2 = w[1] * np.expand_dims(_sigma(beta * x), axis = 0).T
    inside_sigma = np.sum(np.hstack((_z_i1, _z_i2)), axis = 0)

    y_new = np.vstack((y, y)).T

    dLdw = 2 *\
            (y_new - _sigma(inside_sigma)) *\
            _sigma(inside_sigma) *\
            [1 - _sigma(inside_sigma)] *\
            inside_sigma

    I, J = dLdw.shape

    dLdw = dLdw.reshape((I, J)) 

    dLdw = -np.sum(dLdw, axis = 0)

    return tuple(dLdw)

def _partial_alpha(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray
) -> float:
    '''
    '''
    _z_i1 = w[0] * np.expand_dims(_sigma(alpha * x), axis = 0).T
    _z_i2 = w[1] * np.expand_dims(_sigma(beta * x), axis = 0).T
    inside_sigma = np.sum(np.hstack((_z_i1, _z_i2)), axis = 1)

    dLda = 2 *\
            (y - _sigma(inside_sigma)) *\
            _sigma(inside_sigma) *\
            [1 - _sigma(inside_sigma)] *\
            w[0] *\
            _sigma(alpha * x) *\
            [1 - _sigma(alpha * x)] *\
            x

    dLda = -np.sum(dLda)
    return dLda

def _partial_beta(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray
) -> float:
    '''
    '''
    _z_i1 = w[0] * np.expand_dims(_sigma(alpha * x), axis = 0).T
    _z_i2 = w[1] * np.expand_dims(_sigma(beta * x), axis = 0).T
    inside_sigma = np.sum(np.hstack((_z_i1, _z_i2)), axis = 1)

    dLdb = 2 *\
            (y - _sigma(inside_sigma)) *\
            _sigma(inside_sigma) *\
            [1 - _sigma(inside_sigma)] *\
            w[1] *\
            _sigma(beta * x) *\
            [1 - _sigma(beta * x)] *\
            x

    dLdb = -np.sum(dLdb)
    return dLdb

def _gradient(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray
) -> np.ndarray:
    '''
    '''
    gradient = np.array([
        *_partial_w(w, alpha, beta, x, y),
        _partial_alpha(w, alpha, beta, x, y),
        _partial_beta(w, alpha, beta, x, y)
    ])
    return gradient

def loss_function(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray
) -> np.ndarray:
    '''
    '''
    _z_i1 = w[0] * np.expand_dims(_sigma(alpha * x), axis = 0).T
    _z_i2 = w[1] * np.expand_dims(_sigma(beta * x), axis = 0).T
    inside_sigma = np.sum(np.hstack((_z_i1, _z_i2)), axis = 1)

    loss_values = (y - _sigma(inside_sigma))**2

    _loss = -np.sum(loss_values)
    return _loss
