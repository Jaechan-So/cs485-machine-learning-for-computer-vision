import numpy as np
import scipy
from sklearn.model_selection import train_test_split

from common import utils
from common.constants import context, config


def get_face_for_plot(face):
    return face.reshape(context['face_row'], context['face_column']).T


class FaceData:
    def __init__(self):
        self.face_data = scipy.io.loadmat(utils.get_face_data_dir())

        assert context['face_row'] * context['face_column'] == self.face_data['X'].shape[0]
        assert self.face_data['X'].shape[1] == self.face_data['l'].shape[1]

        self.data_count = self.face_data['X'].shape[1]

        self.feature_train, self.feature_test, self.label_train, self.label_test = [
            x.T for x in
            train_test_split(
                self.face_data['X'].T,
                self.face_data['l'].T,
                test_size=0.2,
                random_state=config['train_test_split_seed']
            )
        ]
