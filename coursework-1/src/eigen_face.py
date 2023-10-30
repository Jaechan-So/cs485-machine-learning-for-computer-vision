import numpy as np

from common.decorators import measure_time
from face_data import FaceData
from face_plot import FacePlot, display, get_face_for_plot


def pca(matrix, data_count):
    covariance_matrix_train = (matrix @ matrix.T) / data_count
    return np.linalg.eig(covariance_matrix_train)


class EigenFace:
    def __init__(self, face_data: FaceData):
        self._eigen_vector = None
        self._eigen_value = None
        self._face_data = face_data
        self._mean_face = None
        self._face_plot = None

    def _compute_eig_d(self):
        return pca(self._face_data.feature_train, self._face_data.data_count)

    def _compute_eig_n(self):
        return pca(self._face_data.feature_train.T, self._face_data.data_count)

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

    def _compute_low_dimension_eigen(self):
        low_dimension_matrix = self._face_data.feature_train if self._face_data.feature_train.shape[0] < \
                                                                self._face_data.feature_train.shape[
                                                                    1] else self._face_data.feature_train.T
        self._eigen_value, self._eigen_vector = pca(low_dimension_matrix, self._face_data.data_count)

    def _sort_eigen(self):
        if self._eigen_value is None:
            raise ValueError('Eigenvalue is not computed yet')
        if self._eigen_vector is None:
            raise ValueError('Eigenvector is not computed yet')

        eigen = zip(self._eigen_value, self._eigen_vector)
        sorted_eigen = sorted(eigen, key=lambda e: -e[0])

        self._eigen_value = [e[0] for e in sorted_eigen]
        self._eigen_vector = [e[1] for e in sorted_eigen]

    def _compute_mean_face(self):
        self._mean_face = np.mean(self._face_data.feature_train, axis=1)

    def preprocess_eigen(self):
        self._compute_low_dimension_eigen()
        self._sort_eigen()

    def _reconstruct(self, face, num_of_eigen):
        assert num_of_eigen <= len(num_of_eigen)

        reconstructed = self._mean_face

        for i in range(num_of_eigen):
            a = (face - self._mean_face) * self._eigen_vector[i]
            reconstructed += (a * self._eigen_vector[i])

        return reconstructed

    def reconstruct_and_display(self, faces, num_of_eigen):
        if self._face_plot is None:
            self._face_plot = FacePlot(len(faces))

        for face in faces:
            reconstructed = self._reconstruct(face, num_of_eigen)
            self._face_plot.add_plot(get_face_for_plot(reconstructed))

        display()
