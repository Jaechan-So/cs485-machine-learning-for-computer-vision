import numpy as np

from common.decorators import measure_time
from face_data import FaceData


def pca(matrix, data_count):
    covariance_matrix_train = (matrix @ np.transpose(matrix)) / data_count
    return np.linalg.eig(covariance_matrix_train)


class EigenFace:
    def __init__(self, face_data: FaceData):
        self._eigen_vector = None
        self._eigen_value = None
        self._face_data = face_data

    def _compute_eig_d(self):
        return pca(self._face_data.feature_train, self._face_data.data_count)

    def _compute_eig_n(self):
        return pca(np.transpose(self._face_data.feature_train), self._face_data.data_count)

    @measure_time(measure_time_tag='Q1-a')
    def _compute_eig_d_with_measure_time(self):
        return self._compute_eig_d()

    @measure_time(measure_time_tag='Q1-b')
    def _compute_eig_n_with_measure_time(self):
        return self._compute_eig_n()

    def test_dimensionality(self):
        eigen_value_d, _ = self._compute_eig_d_with_measure_time()
        eigen_value_n, _ = self._compute_eig_n_with_measure_time()
        print('Therefore, using (1/N)(A^T)A is more efficient with respect to time and space.')
        print(
            f'# of eigenvalues using (1/N)(A^T)A: {len(eigen_value_n)}, # of eigenvalues which are also in the set of eigenvalues using (1/N)A(A^T): {len([x in set(eigen_value_d) for x in eigen_value_n])}'
        )
        print('Hence, the set of eigenvalues from (1/N)(A^T)A is the subset of the set of eigenvalues from (1/N)A(A^T)')

    def compute_low_dimension_eigen(self):
        self.eigen_value_n, self.eigen_vector_n = self.compute_eig_n()
