import numpy as np

from common.constants import context
from common.decorators import measure_time
from common.decorators import separate_logs
from common.operations import low_dimension_pca, sort_eigen, evaluate_face_recognition_result
from face_data import FaceData


class EigenSpaceClassFaceRecognition:

    @measure_time(tag='Q1 - Application of Eigenfaces - b: Computing Eigenspaces per Class')
    def _compute_eigen_spaces(self):
        class_to_feature_train = dict()
        for (index, class_num) in enumerate(self._face_data.label_train.T.reshape(-1)):
            if class_num not in class_to_feature_train:
                class_to_feature_train[class_num] = []
            class_to_feature_train[class_num].append(self._face_data.feature_train.T[index])

        class_to_eigen_space = dict()
        for (class_num, f) in class_to_feature_train.items():
            features = np.array(f)
            eigen_space = dict()

            data_count = features.shape[0]
            eigen_values, low_dimension_eigen_vectors = low_dimension_pca(features, data_count)
            sorted_eigen_values, sorted_low_dimension_eigen_vectors = sort_eigen(eigen_values,
                                                                                 low_dimension_eigen_vectors)
            high_dimension_eigen_vectors = (features.T @ sorted_low_dimension_eigen_vectors.T).T

            eigen_space['data_count'] = data_count
            eigen_space['eigen_values'] = sorted_eigen_values
            eigen_space['eigen_vectors'] = high_dimension_eigen_vectors
            eigen_space['mean_face'] = np.mean(features, axis=0)

            class_to_eigen_space[class_num] = eigen_space

        return class_to_eigen_space

    def __init__(self, face_data: FaceData):
        self._face_data = face_data
        self._class_to_eigen_space = self._compute_eigen_spaces()

    def _compute_projected_face_on_eigen_space(self, face, eigen_space):
        reconstruction = np.copy(eigen_space['mean_face'])
        projections = (face - eigen_space['mean_face']) @ eigen_space['eigen_vectors'].T

        for index, eigen_vector in enumerate(eigen_space['eigen_vectors']):
            reconstruction += (projections[index] * eigen_vector)

        return reconstruction

    def _find_min_error_class(self, face, norm):
        projected_faces = [(class_num, self._compute_projected_face_on_eigen_space(face, eigen_space)) for
                           (class_num, eigen_space) in
                           self._class_to_eigen_space.items()]
        reconstructed_errors = np.array([
            norm(face, projected_face)
            for (_, projected_face) in projected_faces
        ])
        min_error_index = np.argmin(reconstructed_errors)
        min_error_class_num, _ = projected_faces[min_error_index]
        return min_error_class_num

    def _compute_nearest_neighbors_predictions(self, norm):
        return np.array([self._find_min_error_class(face, norm) for face in
                         self._face_data.feature_test.T])

    @separate_logs
    def _compute_eigen_space_classes_and_evaluate_result(self, norm_name, norm):
        print(
            f'Compute classes with the lowest reconstruction error and evaluate results with {norm_name}')
        predictions = self._compute_nearest_neighbors_predictions(norm)
        labels = self._face_data.label_test.reshape(-1)

        total_count = self._face_data.feature_test.T.shape[0]
        evaluate_face_recognition_result(total_count, labels, predictions)

    def test_eigen_space_class_face_recognition(self):
        for (norm_name, norm) in context['norms'].items():
            self._compute_eigen_space_classes_and_evaluate_result(norm_name, norm)
