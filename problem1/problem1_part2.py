# Standard Libraries
import os
import numpy as np

# Loading .mat files
import scipy.io

# Create Splines
from scipy.interpolate import BSpline, splrep

# Error Metric
from sklearn.metrics import accuracy_score, confusion_matrix

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Modeling
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def b_spline_coefficients(
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

    _, c, _ = splrep(
        x,
        y,
        k = 3,
        task = -1,
        t = np.linspace(x.min(), x.max(), knot_count + 2)[1 : -1]
    )

    return c

def convert_x_data(
    x : np.ndarray,
    best_knot_count : int = 47
) -> np.ndarray:
    '''
    '''
    new_x = []
    for i in range(len(x)):
        y_row = x[i]
        x_row = np.arange(0, len(y_row)) / (len(y_row) - 1)
        coefficients = b_spline_coefficients(x_row, y_row, best_knot_count)
        new_x.append(coefficients)
    new_x = np.array(new_x)
    return new_x

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    matlab_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Q1Part2.mat'))
    q1_part_2_data = scipy.io.loadmat(matlab_file_path)['Q1Part2']

    y = q1_part_2_data[:, 0]
    x = q1_part_2_data[:, 1:]

    # =====================================================
    # Part A
    # =====================================================
    # Pre-processing
    new_x = convert_x_data(x)[:, :-4]

    # Training
    x_train = new_x[:50, :].copy()
    y_train = y[:50].copy()

    x_test = new_x[50:, :].copy()
    y_test = y[50:].copy()

    svm = SVC(kernel = 'linear')
    svm.fit(x_train, y_train)

    # Model Evaluation
    y_pred = svm.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print(fr'The accuracy of the model is: {accuracy}')
    print(f'The confusion matrix for the model is: \n{confusion}')

    # =====================================================
    # Part B
    # =====================================================
    pca = PCA(n_components = 5)
    x_train_pca = pca.fit_transform(x_train)

    explain_variance_data = pca.explained_variance_ratio_.cumsum()

    print(f'The explained variance of the data after each added component: \n{explain_variance_data}')

    pca = PCA(n_components = 3)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.fit_transform(x_test)

    # Training    
    svm = SVC(kernel = 'linear')
    svm.fit(x_train_pca, y_train)

    # Model Evaluation
    y_pred = svm.predict(x_test_pca)

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print(fr'The accuracy of the model with fPCA is: {accuracy}')
    print(f'The confusion matrix for the model with fPCA is: \n{confusion}')
