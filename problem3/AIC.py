import numpy as np

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