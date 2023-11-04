import numpy as np
import scipy
from sklearn.model_selection import train_test_split

from common import utils
from common.constants import context, config
from common.operations import pca, low_dimension_pca, sort_eigen


class FaceData:
    def _load_face_data(self):
        face_data = scipy.io.loadmat(utils.get_face_data_dir())

        assert context['face_row'] * context['face_column'] == face_data['X'].shape[0]
        assert face_data['X'].shape[1] == face_data['l'].shape[1]

        data_count = face_data['X'].shape[1]

        return face_data, data_count

    def _split_faces_into_train_and_test(self):
        return [
            x.T for x in
            train_test_split(
                self.face_data['X'].T,
                self.face_data['l'].T,
                test_size=0.2,
                random_state=config['train_test_split_seed']
            )
        ]

    def _convert_low_dimension_eigen_vector_to_high_dimension_eigen_vector(self, eigen_vector):
        return (self.feature_train @ eigen_vector.reshape(-1, 1)).reshape(-1)

    def _preprocess_eigen(self):
        eigen_value, eigen_vector = low_dimension_pca(self.feature_train, self.data_count)
        sorted_eigen_value, sorted_eigen_vector = sort_eigen(eigen_value, eigen_vector)
        high_dimension_eigen_vector = np.array(
            [self._convert_low_dimension_eigen_vector_to_high_dimension_eigen_vector(e) for e in sorted_eigen_vector])
        return sorted_eigen_value, high_dimension_eigen_vector

    def __init__(self):
        self.face_data, self.data_count = self._load_face_data()
        self.feature_train, self.feature_test, self.label_train, self.label_test = self._split_faces_into_train_and_test()
        self.mean_face = np.mean(self.feature_train, axis=1)
        self.eigen_value, self.eigen_vector = self._preprocess_eigen()

    def compute_eig_d(self):
        return pca(self.feature_train, self.data_count)

    def compute_eig_n(self):
        return pca(self.feature_train.T, self.data_count)
