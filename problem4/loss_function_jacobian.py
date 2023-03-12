import numpy as np
import math

def loss_function(
    x : np.ndarray,
    y : np.ndarray,
    beta_1 : float,
    beta_2 : float
) -> float:
    '''
    '''
    loss = (y - np.sin(beta_1 * x) / (beta_2 * x + 1))**2
    loss = np.sum(loss)
    return loss

def decomposed_loss_function(
    x : float,
    y : float,
    beta_1 : float,
    beta_2 : float
) -> np.ndarray:
    '''
    '''
    _g = y - np.sin(beta_1 * x) / (beta_2 * x + 1)
    return _g.T

def right_term(
    x : np.ndarray,
    beta_1 : float,
    beta_2 : float 
) -> float:
    '''
    '''
    _right_term = np.sin(beta_1 * x) / (beta_2 * x + 1)
    return _right_term

def _partial_beta1(
    xi : float,
    yi : float,
    beta_1 : float,
    beta_2 : float
) -> float:
    '''
    '''
    numerator = -xi * np.cos(beta_1 * xi)
    denominator = beta_2 * xi + 1

    partial_beta1 = numerator / denominator
    return partial_beta1

def _partial_beta2(
    xi : float,
    yi : float,
    beta_1 : float,
    beta_2 : float
) -> float:
    '''
    '''
    numerator = xi * np.sin(beta_1 * xi)
    denominator = (beta_2 * xi + 1)**2

    partial_beta2 = numerator / denominator
    return partial_beta2

def jacobian(
    x : np.ndarray,
    y : np.ndarray,
    beta_1 : float,
    beta_2 : float
) -> np.ndarray:
    '''
    '''
    jacobian = []
    for xi, yi in zip(x, y):
        partial_beta1 = _partial_beta1(xi, yi, beta_1, beta_2)
        partial_beta2 = _partial_beta2(xi, yi, beta_1, beta_2)
        jacobian.append([partial_beta1, partial_beta2])
    jacobian = np.array(jacobian)
    return jacobian
