# Standard Libraries
import os
import numpy as np

# Loading .mat files
import scipy.io

# Create Splines
from scipy.interpolate import BSpline, splrep

# Error Metric
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

def compute_mse(
    x : np.ndarray,
    y : np.ndarray,
    spline : object
) -> float:
    '''
    compute_mse():
        Compute the mean squared error of a spline
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
        spline: spline object
    '''
    y_pred = spline(x)
    error = mean_squared_error(y, y_pred)
    return error

def b_spline(
    x : np.ndarray,
    y : np.ndarray,
    knot_count : int
) -> object:
    '''
    b_spline():
        Perform a B-spline interpolation with knots as an input parameter
        x: array of x-coordinates of data points
        y: array of y-coordinates of data points
        knot_count: the number of knots expected for the spline object
    '''
    assert x.shape == y.shape

    t, c, k = splrep(
        x,
        y,
        k = 3,
        task = -1,
        t = np.linspace(x.min(), x.max(), knot_count + 2)[1 : -1]
    )

    spline = BSpline(t, c, k, extrapolate = True)

    return spline

def k_fold_cross_validation(
    x : np.ndarray,
    y : np.ndarray,
    knots : int,
    n : int = 5
) -> float:
    '''
    '''
    folds = KFold(n_splits = n, shuffle = True, random_state = 1)

    errors = []
    for train_index, test_index in folds.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        spline = b_spline(x_train, y_train, knots)
        error = compute_mse(x_test, y_test, spline)

        errors.append(error)
    errors = np.array(errors)
    mean_error = np.mean(errors)
    return mean_error

def get_best_knot_count(
    x : np.ndarray,
    y : np.ndarray
) -> np.ndarray:
    '''
    '''
    mean_errors = []
    for knot_count in range(5, 51):
        mean_error = k_fold_cross_validation(x, y, knot_count, n = 5)
        mean_errors.append(mean_error)
    mean_errors = np.array(mean_errors)
    return mean_errors

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    matlab_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Q1Part1.mat'))
    q1_part_1_data = scipy.io.loadmat(matlab_file_path)['Q1Part1']
    
    y = q1_part_1_data.flatten()
    x = np.arange(0, len(y)) / (len(y) - 1)

    # Plotting: Plot 1
    # =================
    mean_errors = get_best_knot_count(x, y)
    plot_x = np.arange(5, len(mean_errors) + 5)

    ax = sns.lineplot(x = plot_x, y = mean_errors)
    ax.set_yscale('log')

    plt.xlabel('Knot Count')
    plt.ylabel('Mean Error (5 Folds)')
    plt.title('Best Knot Count Using 5-Fold Cross Validation')
    
    output_filepath_kfold = os.path.abspath(os.path.join(current_path, '..', '..', 'output', '5fold_cross_validation_splines.png'))
    plt.savefig(output_filepath_kfold)

    plt.clf()
    plt.cla()
    # =================

    # Plotting: Plot 2
    # =================
    mean_errors = get_best_knot_count(x, y)
    plot_x = np.arange(5, len(mean_errors) + 5)

    ax = sns.lineplot(x = plot_x[-10:], y = mean_errors[-10:])
    ax.set_yscale('log')

    plt.xlabel('Knot Count')
    plt.ylabel('Mean Error (5 Folds)')
    plt.title('Best Knot Count Using 5-Fold Cross Validation')
    
    output_filepath_kfold = os.path.abspath(os.path.join(current_path, '..', '..', 'output', '5fold_cross_validation_splines_final_10.png'))
    plt.savefig(output_filepath_kfold)

    plt.clf()
    plt.cla()
    # =================

    best_knot_count = plot_x[np.argmin(mean_errors)]
    best_spline = b_spline(x, y, best_knot_count)
    # best_spline = b_spline(x, y, 1)

    y_pred = best_spline(x)

    # Plotting: Plot 3
    # =================
    sns.lineplot(x = x, y = y_pred)
    sns.scatterplot(x = x, y = y, color = 'orange')

    plt.xlabel('X-Values')
    plt.ylabel('Y-Values')
    plt.title('Best Spline Vs. Data')
    
    output_filepath_kfold = os.path.abspath(os.path.join(current_path, '..', '..', 'output', 'best_spline_plot.png'))
    plt.savefig(output_filepath_kfold)

    plt.clf()
    plt.cla()
    # =================
