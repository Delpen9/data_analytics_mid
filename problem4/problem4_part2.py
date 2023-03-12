# Standard libraries
import os
import numpy as np

# Imports
from loss_function import loss_function, _gradient

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    csv_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'Question4Part2-1.csv'))
    data = np.genfromtxt(csv_file_path, delimiter = ',')
    x = data[:, 0]
    y = data[:, 1]

    w = np.array([0.4, 4])
    alpha = 1
    beta = 5

    print(loss_function(w, alpha, beta, x, y))
    print(_gradient(w, alpha, beta, x, y))