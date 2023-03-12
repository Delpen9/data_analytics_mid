# Standard libraries
import os
import numpy as np

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

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
    tolerance : float = 1e-8
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    loss_history = []

    params = np.array([*w, alpha, beta])

    loss = loss_function(w, alpha, beta, x, y)
    loss_history.append(loss)
    gradient = _gradient(w, alpha, beta, x, y)

    while (np.dot(gradient.T, gradient) > tolerance):
        new_w = np.array([params[0], params[1]])
        gradient = _gradient(new_w, params[2], params[3], x, y)
        step = exact_line_search(new_w, params[2], params[3], x, y, gradient)
        
        params = params + step * gradient

        loss_old = loss
        new_w = np.array([params[0], params[1]])
        loss = loss_function(new_w, params[2], params[3], x, y)
        loss_history.append(loss)
    return params, loss_history

def accelerated_gradient_descent(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray,
    tolerance : float = 1e-8
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    k = 1
    loss_history = []

    params_x = np.array([*w, alpha, beta])
    params_x0 = params_x.copy()
    params_x00 = params_x.copy()
    params_y = params_x.copy()

    loss = loss_function(w, alpha, beta, x, y)
    loss_history.append(loss)
    gradient = _gradient(w, alpha, beta, x, y)

    while (np.dot(gradient.T, gradient) > tolerance):
        new_w = np.array([params_x[0], params_x[1]])
        gradient = _gradient(new_w, params_x[2], params_x[3], x, y)
        step = exact_line_search(new_w, params_x[2], params_x[3], x, y, gradient)
        
        params_x = params_y + step * gradient
        params_y = params_x0 + (k - 1) / (k + 2) * (params_x0 - params_x00)

        loss_old = loss
        new_w = np.array([params_x[0], params_x[1]])
        loss = loss_function(new_w, params_x[2], params_x[3], x, y)
        loss_history.append(loss)

        params_x00 = params_x0.copy()
        params_x0 = params_x.copy()

        k += 1
    return params_x, loss_history

def stochastic_gradient_descent(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray,
    tolerance : float = 1e-8
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    '''
    loss_history = []

    params = np.array([*w, alpha, beta])

    x_batch = np.random.choice(x, size = 1)
    y_batch = np.random.choice(y, size = 1)

    loss = loss_function(w, alpha, beta, x, y)
    loss_history.append(loss)
    gradient = _gradient(w, alpha, beta, x_batch, y_batch) * len(y)

    while (np.dot(gradient.T, gradient) > tolerance):
        new_w = np.array([params[0], params[1]])

        x_batch = np.random.choice(x, size = 1)
        y_batch = np.random.choice(y, size = 1)

        gradient = _gradient(new_w, params[2], params[3], x_batch, y_batch) * len(y)
        step = exact_line_search(new_w, params[2], params[3], x_batch, y_batch, gradient)
        
        params = params + step * gradient

        loss_old = loss
        new_w = np.array([params[0], params[1]])
        loss = loss_function(new_w, params[2], params[3], x, y)
        loss_history.append(loss)

    return params, loss_history
 

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    csv_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question4Part2-1.csv'))
    data = np.genfromtxt(csv_file_path, delimiter = ',')
    x = data[:, 0]
    y = data[:, 1]

    w = np.array([0.4, 4])
    alpha = 1
    beta = 5

    best_params, loss_history = gradient_descent(w, alpha, beta, x, y)

    # =================
    # Plotting: Plot 1
    # =================
    iterations = np.arange(1, len(loss_history) + 1)

    ax = sns.lineplot(x = iterations, y = loss_history)

    plt.xlabel('Iteration')
    plt.ylabel('Loss Function')
    plt.title('Loss Function Vs. Iteration for Gradient Descent')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'gradient_descent.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================

    best_params, loss_history = accelerated_gradient_descent(w, alpha, beta, x, y)

    # =================
    # Plotting: Plot 2
    # =================
    iterations = np.arange(1, len(loss_history) + 1)

    ax = sns.lineplot(x = iterations, y = loss_history)

    plt.xlabel('Iteration')
    plt.ylabel('Loss Function')
    plt.title('Loss Function Vs. Iteration for Accelerated Gradient Descent')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'accelerated_gradient_descent.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================

    best_params, loss_history = stochastic_gradient_descent(w, alpha, beta, x, y)

    # =================
    # Plotting: Plot 3
    # =================
    iterations = np.arange(1, len(loss_history) + 1)

    ax = sns.lineplot(x = iterations, y = loss_history)

    plt.xlabel('Iteration')
    plt.ylabel('Loss Function')
    plt.title('Loss Function Vs. Iteration for Stochastic Gradient Descent')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'stochastic_gradient_descent.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================