# Standard libraries
import os
import numpy as np

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

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


def gauss_newton_adaptive_step_size(
    x : np.ndarray,
    y : np.ndarray,
    beta_1 : float,
    beta_2 : float,
    tolerance : float = 1e-10,
    damping_factor : float = 1e-10,
    max_iterations : float = 100
) -> tuple[float, float, np.ndarray, np.ndarray]:
    '''
    '''
    loss_history = []
    alpha = 1e+3

    _jacobian = jacobian(x, y, beta_1, beta_2)
    _g = decomposed_loss_function(x, y, beta_1, beta_2)

    omega = -np.linalg.solve(
        np.linalg.inv(np.dot(_jacobian.T, _jacobian) + damping_factor * np.eye(2)),
        np.dot(_jacobian.T, _g)
    )

    iteration = 1
    while (np.amax(np.abs(omega)) > tolerance) and (iteration < max_iterations):
        _jacobian = jacobian(x, y, beta_1, beta_2)
        _g = decomposed_loss_function(x, y, beta_1, beta_2)

        omega = -np.linalg.solve(
            np.linalg.inv(np.dot(_jacobian.T, _jacobian) + damping_factor * np.eye(2)),
            np.dot(_jacobian.T, _g)
        )

        _loss = loss_function(x, y, beta_1, beta_2)
        loss_history.append(_loss)
        new_loss = _loss + 1

        while new_loss > _loss:
            beta_1_new = beta_1 + alpha * omega[0]
            beta_2_new = beta_2 + alpha * omega[1]

            new_loss = loss_function(x, y, beta_1_new, beta_2_new)
            alpha *= 0.1

        beta_1 = beta_1_new
        beta_2 = beta_2_new

        alpha = alpha**0.5

        iteration += 1

    best_beta_1 = beta_1
    best_beta_2 = beta_2

    loss_history = np.array(loss_history)

    return (best_beta_1, best_beta_2, loss_history)

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    csv_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question4Part3-1.csv'))
    data = np.genfromtxt(csv_file_path, delimiter = ',')
    x = data[:, 0]
    y = data[:, 1]

    beta_1 = 0.5
    beta_2 = 0.5
    best_beta_1, best_beta_2, loss_history = gauss_newton_adaptive_step_size(x, y, beta_1, beta_2)

    print(best_beta_1, best_beta_2)

    # =================
    # Plotting: Plot 1
    # =================
    iterations = np.arange(1, len(loss_history) + 1)

    ax = sns.lineplot(x = iterations, y = loss_history)

    plt.xlabel('Iteration')
    plt.ylabel('Loss Function')
    plt.title('Loss Function Vs. Iteration for Gauss Newton Method')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'gauss_newton_method.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================

    # =================
    # Plotting: Plot 2
    # =================
    ax = sns.lineplot(x = x, y = y)

    plt.xlabel('Xi')
    plt.ylabel('Loss Function')
    plt.title('X Vs. Y')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Q4_part3_x_vs_y.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================

    # =================
    # Plotting: Plot 3
    # =================
    right_term_values = right_term(x, best_beta_1, best_beta_2)
    ax = sns.lineplot(x = x, y = right_term_values)

    plt.xlabel('Xi')
    plt.ylabel('Right Term')
    plt.title('X Vs. Right Term')
    
    output_filepath = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'Q4_part3_x_vs_right_term.png'))
    plt.savefig(output_filepath)

    plt.clf()
    plt.cla()
    # =================