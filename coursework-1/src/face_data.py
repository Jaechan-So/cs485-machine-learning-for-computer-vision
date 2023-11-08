import numpy as np
import scipy
from sklearn.model_selection import train_test_split

from common import utils
from common.constants import context, config
from common.operations import get_pca_eigen


class FaceData:
    def _load_face_data(self):
        face_data = scipy.io.loadmat(utils.get_face_data_dir())

        assert context['face_row'] * context['face_column'] == face_data['X'].shape[0]
        assert face_data['X'].shape[1] == face_data['l'].shape[1]

        return face_data

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

    def __init__(self):
        self.face_data = self._load_face_data()
        self.feature_train, self.feature_test, self.label_train, self.label_test = self._split_faces_into_train_and_test()
        self.data_count = self.feature_train.T.shape[0]
        self.mean_face = np.mean(self.feature_train, axis=1)
        self.centered_feature_train = (self.feature_train.T - self.mean_face).T
        self.centered_feature_test = (self.feature_test.T - self.mean_face).T
        self.eigen_values, self.eigen_vectors = get_pca_eigen(self.feature_train, self.data_count)
