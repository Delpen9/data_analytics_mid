import numpy as np

def sigma(
    x : np.ndarray
) -> float:
    _sigma = 1 / (1 + np.e**(-x))
    return _sigma

def loss_function(
    w : np.ndarray,
    alpha : np.ndarray,
    beta : np.ndarray,
    y : np.ndarray,
    x : np.ndarray
) -> np.ndarray:
    '''
    '''
    inside_sigma = w[0] * z[:, 0].flatten() + w[1] * z[:, 1].flatten()
    loss_values = y - sigma(inside_sigma)
    _loss = np.sum(loss_values)
    return _loss