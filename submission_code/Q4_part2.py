# Standard libraries
import os
import numpy as np

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

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

def right_term(
    w0 : float,
    w1 : float,
    alpha : float,
    beta : float,
    x : np.ndarray
) -> np.ndarray:
    '''
    '''
    _z_i1 = w0 * np.expand_dims(_sigma(alpha * x), axis = 0).T
    _z_i2 = w1 * np.expand_dims(_sigma(beta * x), axis = 0).T
    inside_sigma = np.sum(np.hstack((_z_i1, _z_i2)), axis = 1)

    loss_values = _sigma(inside_sigma)

    _loss = -np.sum(loss_values)
    return _loss

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
) -> tuple[list, list, list]:
    '''
    '''
    loss_history = []
    param_history = []

    params = np.array([*w, alpha, beta])
    param_history.append(params)

    loss = loss_function(w, alpha, beta, x, y)
    loss_history.append(loss)
    gradient = _gradient(w, alpha, beta, x, y)

    while (np.dot(gradient.T, gradient) > tolerance):
        new_w = np.array([params[0], params[1]])
        gradient = _gradient(new_w, params[2], params[3], x, y)
        step = exact_line_search(new_w, params[2], params[3], x, y, gradient)
        
        params = params + step * gradient
        param_history.append(params)

        loss_old = loss
        new_w = np.array([params[0], params[1]])
        loss = loss_function(new_w, params[2], params[3], x, y)
        loss_history.append(loss)
    param_history = np.array(param_history)
    return params, loss_history, param_history

def accelerated_gradient_descent(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray,
    tolerance : float = 1e-8
) -> tuple[list, list, list]:
    '''
    '''
    k = 1
    loss_history = []
    param_history = []

    params_x = np.array([*w, alpha, beta])
    param_history.append(params_x)
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
        param_history.append(params_x)
        params_y = params_x0 + (k - 1) / (k + 2) * (params_x0 - params_x00)

        loss_old = loss
        new_w = np.array([params_x[0], params_x[1]])
        loss = loss_function(new_w, params_x[2], params_x[3], x, y)
        loss_history.append(loss)

        params_x00 = params_x0.copy()
        params_x0 = params_x.copy()

        k += 1
    param_history = np.array(param_history)
    return params_x, loss_history, param_history

def stochastic_gradient_descent(
    w : np.ndarray,
    alpha : float,
    beta : float,
    x : np.ndarray,
    y : np.ndarray,
    tolerance : float = 1e-8
) -> tuple[list, list, list]:
    '''
    '''
    loss_history = []
    param_history = []

    params = np.array([*w, alpha, beta])
    param_history.append(params)

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
        param_history.append(params)

        loss_old = loss
        new_w = np.array([params[0], params[1]])
        loss = loss_function(new_w, params[2], params[3], x, y)
        loss_history.append(loss)
    param_history = np.array(param_history)
    return params, loss_history, param_history
 

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    csv_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question4Part2-1.csv'))
    data = np.genfromtxt(csv_file_path, delimiter = ',')
    x = data[:, 0]
    y = data[:, 1]

    w = np.array([0.4, 4])
    alpha = 1
    beta = 5

    best_params, loss_history, param_history = gradient_descent(w, alpha, beta, x, y)
    print(fr'''
The best parameters given by gradient descent are:
w0 = {best_params[0]},
w1 = {best_params[1]},
alpha = {best_params[2]},
beta = {best_params[3]}

    ''')

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

    # =================
    # Plotting: Plot 2
    # =================
    iterations = np.arange(1, len(loss_history) + 1)

    ax = sns.lineplot(x = iterations, y = param_history[:, 0])
    ax = sns.lineplot(x = iterations, y = param_history[:, 1])
    ax = sns.lineplot(x = iterations, y = param_history[:, 2])
    ax = sns.lineplot(x = iterations, y = param_history[:, 3])

    plt.xlabel('Iteration')
    plt.ylabel('Parameter Histories')
    plt.title('Parameter Histories Vs. Iteration for Gradient Descent')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'gradient_descent_parameter_histories.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================

    # =================
    # Plotting: Plot 3
    # =================
    _right_term_data = right_term(*best_params, x)

    ax = sns.lineplot(x = x, y = _right_term_data)
    ax = sns.lineplot(x = x, y = y)

    plt.xlabel('Iteration')
    plt.ylabel('Right Term and Y')
    plt.title('Right Term and Y Vs. Iteration for Gradient Descent')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'gradient_descent_right_term_and_y.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================

    best_params, loss_history, param_history = accelerated_gradient_descent(w, alpha, beta, x, y)
    print(fr'''
The best parameters given by accelerated gradient descent are:
w0 = {best_params[0]},
w1 = {best_params[1]},
alpha = {best_params[2]},
beta = {best_params[3]}

    ''')

    # =================
    # Plotting: Plot 4
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

    # =================
    # Plotting: Plot 5
    # =================
    iterations = np.arange(1, len(loss_history) + 1)

    ax = sns.lineplot(x = iterations, y = param_history[:, 0])
    ax = sns.lineplot(x = iterations, y = param_history[:, 1])
    ax = sns.lineplot(x = iterations, y = param_history[:, 2])
    ax = sns.lineplot(x = iterations, y = param_history[:, 3])

    plt.xlabel('Iteration')
    plt.ylabel('Parameter Histories')
    plt.title('Parameter Histories Vs. Iteration for Accelerated Gradient Descent')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'accelerated_gradient_descent_parameter_histories.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================

    # =================
    # Plotting: Plot 6
    # =================
    _right_term_data = right_term(*best_params, x)

    ax = sns.lineplot(x = x, y = _right_term_data)
    ax = sns.lineplot(x = x, y = y)

    plt.xlabel('Iteration')
    plt.ylabel('Right Term and Y')
    plt.title('Right Term and Y Vs. Iteration for Accelerated Gradient Descent')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'accelerated_gradient_descent_right_term_and_y.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================

    best_params, loss_history, param_history = stochastic_gradient_descent(w, alpha, beta, x, y)
    print(fr'''
The best parameters given by stochastic gradient descent are:
w0 = {best_params[0]},
w1 = {best_params[1]},
alpha = {best_params[2]},
beta = {best_params[3]}

    ''')

    # =================
    # Plotting: Plot 7
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

    # =================
    # Plotting: Plot 8
    # =================
    iterations = np.arange(1, len(loss_history) + 1)

    ax = sns.lineplot(x = iterations, y = param_history[:, 0])
    ax = sns.lineplot(x = iterations, y = param_history[:, 1])
    ax = sns.lineplot(x = iterations, y = param_history[:, 2])
    ax = sns.lineplot(x = iterations, y = param_history[:, 3])

    plt.xlabel('Iteration')
    plt.ylabel('Parameter Histories')
    plt.title('Parameter Histories Vs. Iteration for Stochastic Gradient Descent')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'stochastic_gradient_descent_parameter_histories.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================

    # =================
    # Plotting: Plot 9
    # =================
    _right_term_data = right_term(*best_params, x)

    ax = sns.lineplot(x = x, y = _right_term_data)
    ax = sns.lineplot(x = x, y = y)

    plt.xlabel('Iteration')
    plt.ylabel('Right Term and Y')
    plt.title('Right Term and Y Vs. Iteration for Stochastic Gradient Descent')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'stochastic_gradient_descent_right_term_and_y.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================