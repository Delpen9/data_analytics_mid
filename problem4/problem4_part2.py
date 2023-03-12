# Standard libraries
import os
import numpy as np

# Imports
from loss_function import loss_function, _gradient

def exact_line_search(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray,
    gradient : np.ndarray
) -> np.ndarray[float, float]:
    '''
    '''
    log_values = np.logspace(-2, 2, num = 4, base = 10.0)
    incremental_values = np.arange(0.1, 1.01, 0.1)
    step_sizes = np.outer(log_values, incremental_values).flatten()

    exact_lines = np.array([
        (
            w[0] + step_size * gradient[0],
            w[1] + step_size * gradient[1],
            alpha + step_size * gradient[2],
            beta + step_size * gradient[3],
        )
        for step_size in step_sizes
    ])

    loss = np.array([
        loss_function(
            np.array([exact_line[0], exact_line[1]]),
            exact_line[2],
            exact_line[3],
            x,
            y
        ) for exact_line in exact_lines
    ])

    best_step = step_sizes[np.argmin(loss)]
    return best_step

def gradient_descent(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray,
    tolerance : float = 1e-2
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    params = np.array([*w, alpha, beta])

    loss = loss_function(w, alpha, beta, x, y)
    gradient = _gradient(w, alpha, beta, x, y)

    while (np.dot(gradient.T, gradient) > tolerance):
        new_w = np.array([params[0], params[1]])
        gradient = _gradient(new_w, params[2], params[3], x, y)
        step = exact_line_search(new_w, params[2], params[3], x, y, gradient)
        
        params = params + step * gradient

        loss_old = loss
        new_w = np.array([params[0], params[1]])
        loss = loss_function(new_w, params[2], params[3], x, y)

        print(np.dot(gradient.T, gradient))
    return params

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    csv_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question4Part2-1.csv'))
    data = np.genfromtxt(csv_file_path, delimiter = ',')
    x = data[:, 0]
    y = data[:, 1]

    w = np.array([0.4, 4])
    alpha = 1
    beta = 5

    best_params = gradient_descent(w, alpha, beta, x, y)