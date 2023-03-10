# Standard Libraries
import os
import numpy as np

# Loading .mat files
import scipy.io

# Imports
from CPDecomposition import cp_decomposition

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    matlab_file_path = os.path.abspath(os.path.join(current_path, '..', '..', 'data', 'heat.mat'))

    X = scipy.io.loadmat(matlab_file_path)['X'][0][0][0]
    rank = 5
    N = 50
    core, factors = cp_decomposition(X, rank, N)

    print(core.shape)