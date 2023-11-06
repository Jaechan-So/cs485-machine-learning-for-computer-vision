import itertools

import numpy as np

eigen_counts = [3, 10, 50, 100, 200]
norms = {
    'L1 Norm': lambda x: np.linalg.norm(x, ord=1),
    'L2 Norm': lambda x: np.linalg.norm(x, ord=2),
}
pca_test_parameters = [(eigen_count, name, norms[name]) for eigen_count, name in
                       list(itertools.product(eigen_counts, norms))]

m_pca_candidates = [3, 10, 30, 100, 200]
m_lda_candidates = [3, 5, 10, 30, 50]
pca_lda_test_parameters = [(m_pca, m_lda, norm_name, norms[norm_name]) for m_pca, m_lda, norm_name in
                           list(itertools.product(m_pca_candidates, m_lda_candidates, norms)) if m_pca > m_lda]

context = {
    'face_row': 46,
    'face_column': 56,
    'eigen_value_tolerance': 1e-10,
    'eigen_counts': eigen_counts,
    'norms': norms,
    'pca_test_parameters': pca_test_parameters,
    'pca_lda_test_parameters': pca_lda_test_parameters,
}

config = {
    'train_test_split_seed': 35,
}
