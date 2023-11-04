import itertools

import numpy as np

eigen_counts = [3, 10, 50, 100, 200]
norms = {
    'L1 Norm': lambda x1, x2: np.linalg.norm(x1 - x2, ord=1),
    'L2 Norm': lambda x1, x2: np.linalg.norm(x1 - x2, ord=2),
}
test_parameters = [(eigen_count, name, norms[name]) for eigen_count, name in
                   list(itertools.product(eigen_counts, norms))]

context = {
    'face_row': 46,
    'face_column': 56,
    'eigen_counts': eigen_counts,
    'norms': norms,
    'test_parameters': test_parameters,
}

config = {
    'train_test_split_seed': 35,
}
