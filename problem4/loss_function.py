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
    _z_i1 = np.expand_dims(_sigma(alpha * x), axis = 0).T
    _z_i2 = np.expand_dims(_sigma(beta * x), axis = 0).T
    _z = np.hstack((_z_i1, _z_i2))

    w = np.expand_dims(w, axis = 0)

    inside_sigma = (w @ _z.T).flatten()

    dLdw = 2 *\
            _sigma(inside_sigma) *\
            _sigma(inside_sigma) *\
            [1 - _sigma(inside_sigma)] *\
            np.sum(_z, axis = 1).flatten()
    
    dLdw = -np.sum(dLdw)
    return dLdw

def _partial_alpha(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray
) -> float:
    '''
    '''
    _z_i1 = np.expand_dims(_sigma(alpha * x), axis = 0).T
    _z_i2 = np.expand_dims(_sigma(beta * x), axis = 0).T
    _z = np.hstack((_z_i1, _z_i2))

    w = np.expand_dims(w, axis = 0)

    inside_sigma = (w @ _z.T).flatten()

    dLda = 2 *\
            _sigma(inside_sigma) *\
            _sigma(inside_sigma) *\
            [1 - _sigma(inside_sigma)] *\
            w[0][0] *\
            _sigma(alpha * _z_i1.flatten()) *\
            [1 - _sigma(alpha * _z_i1.flatten())] *\
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
    _z_i1 = np.expand_dims(_sigma(alpha * x), axis = 0).T
    _z_i2 = np.expand_dims(_sigma(beta * x), axis = 0).T
    _z = np.hstack((_z_i1, _z_i2))

    w = np.expand_dims(w, axis = 0)

    inside_sigma = (w @ _z.T).flatten()

    dLdb = 2 *\
            _sigma(inside_sigma) *\
            _sigma(inside_sigma) *\
            [1 - _sigma(inside_sigma)] *\
            w[0][1] *\
            _sigma(beta * _z_i2.flatten()) *\
            [1 - _sigma(beta * _z_i2.flatten())] *\
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
        _partial_w(w, alpha, beta, x, y),
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
    _z_i1 = np.expand_dims(_sigma(alpha * x), axis = 0).T
    _z_i2 = np.expand_dims(_sigma(beta * x), axis = 0).T
    _z = np.hstack((_z_i1, _z_i2))

    w = np.expand_dims(w, axis = 0)

    inside_sigma = (w @ _z.T).flatten()

    loss_values = (y - _sigma(inside_sigma))**2

    _loss = np.sum(loss_values)
    return _loss
