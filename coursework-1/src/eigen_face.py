import numpy as np

from common.decorators import measure_time
from face_data import FaceData

reconstruction_choice_count = 3
reconstruction_eigen_counts = [3, 10, 50, 100, 200]


def pca(matrix, data_count):
    covariance_matrix_train = (matrix @ matrix.T) / data_count
    return np.linalg.eig(covariance_matrix_train)


class EigenFace:
    def __init__(self, face_data: FaceData):
        self._face_data = face_data

    @measure_time(tag='Q1 - Eigenfaces - a')
    def _compute_eig_d_with_measure_time(self):
        return self._face_data.compute_eig_d()

    @measure_time(tag='Q1 - Eigenfaces - b')
    def _compute_eig_n_with_measure_time(self):
        return self._face_data.compute_eig_n()

    def test_dimensionality(self):
        eigen_value_d, _ = self._compute_eig_d_with_measure_time()
        eigen_value_n, _ = self._compute_eig_n_with_measure_time()
        print('Therefore, using (1/N)(A^T)A is more efficient with respect to time and space.')
        print(
            f'# of eigenvalues using (1/N)(A^T)A: {len(eigen_value_n)}, # of eigenvalues which are also in the set of eigenvalues using (1/N)A(A^T): {len([x in set(eigen_value_d) for x in eigen_value_n])}'
        )
        print('Hence, the set of eigenvalues from (1/N)(A^T)A is the subset of the set of eigenvalues from (1/N)A(A^T)')
