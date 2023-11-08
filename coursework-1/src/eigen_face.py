import matplotlib.pyplot as plt
import numpy as np

from common.constants import context
from common.decorators import measure_time
from common.operations import reshape_face_for_plot, pca
from face_data import FaceData

reconstruction_choice_count = 3
reconstruction_eigen_counts = [3, 10, 50, 100, 200]


class EigenFace:
    def __init__(self, face_data: FaceData):
        self._face_data = face_data

    @measure_time(tag='Q1 - Eigenfaces - a')
    def _compute_eig_d_with_measure_time(self):
        return pca(self._face_data.centered_feature_train, self._face_data.data_count)

    @measure_time(tag='Q1 - Eigenfaces - b')
    def _compute_eig_n_with_measure_time(self):
        return pca(self._face_data.centered_feature_train.T, self._face_data.data_count)

    def test_dimensionality(self):
        plt.title('Mean face')
        plt.imshow(reshape_face_for_plot(self._face_data.mean_face))

        eigen_value_d, eigen_vector_d = self._compute_eig_d_with_measure_time()
        valid_indicies = np.where(eigen_value_d > context['eigen_value_tolerance'])
        valid_eigen_value_d = eigen_value_d[valid_indicies]
        print(f'There are {valid_eigen_value_d.shape[0]} non-zero eigenvalues from the PCA with (1/N)A(A^T).')

        eigen_value_n, eigen_vector_n = self._compute_eig_n_with_measure_time()
        valid_indicies = np.where(eigen_value_n > context['eigen_value_tolerance'])
        valid_eigen_value_n = eigen_value_n[valid_indicies]
        print(f'There are {valid_eigen_value_n.shape[0]} non-zero eigenvalues from the PCA with (1/N)(A^T)A.')

        print('Therefore, using (1/N)(A^T)A is more efficient with respect to time and space.')
        print(
            f'# of eigenvalues using (1/N)(A^T)A: {valid_eigen_value_n.shape[0]}, # of eigenvalues which are also in the set of eigenvalues using (1/N)A(A^T): {len([x in set(valid_eigen_value_d.T) for x in valid_eigen_value_n.T])}'
        )
        print('Hence, the set of eigenvalues from (1/N)(A^T)A is the subset of the set of eigenvalues from (1/N)A(A^T)')
