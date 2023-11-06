import numpy as np

from common.constants import context
from common.decorators import measure_time
from common.decorators import separate_logs
from common.operations import evaluate_face_recognition_result, get_pca_eigen, get_reconstructions, \
    plot_example_success_and_failure_case_of_face_recognition
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
            eigen_values, eigen_vectors = get_pca_eigen(features.T, data_count)

            eigen_space['data_count'] = data_count
            eigen_space['eigen_values'] = eigen_values
            eigen_space['eigen_vectors'] = eigen_vectors
            eigen_space['mean_face'] = np.mean(features, axis=0)

            class_to_eigen_space[class_num] = eigen_space

        return class_to_eigen_space

    def __init__(self, face_data: FaceData):
        self._face_data = face_data
        self._class_to_eigen_space = self._compute_eigen_spaces()

    def _find_min_error_class(self, face, norm):
        reconstructed_faces = [
            (class_num, get_reconstructions(face, eigen_space['mean_face'], eigen_space['eigen_vectors'])) for
            (class_num, eigen_space) in
            self._class_to_eigen_space.items()]
        reconstructed_errors = np.array([
            norm(face - reconstructed_face)
            for (_, reconstructed_face) in reconstructed_faces
        ])
        min_error_index = np.argmin(reconstructed_errors)
        min_error_class_num, _ = reconstructed_faces[min_error_index]
        return min_error_class_num

    def _compute_predictions(self, norm):
        return np.array([self._find_min_error_class(face, norm) for face in
                         self._face_data.feature_test.T])

    @separate_logs
    def _compute_eigen_space_classes_and_evaluate_result(self, norm_name, norm):
        print(
            f'Compute classes with the lowest reconstruction error and evaluate results with {norm_name}')
        predictions = self._compute_predictions(norm)
        labels = self._face_data.label_test.reshape(-1)

        total_count = self._face_data.feature_test.T.shape[0]
        evaluate_face_recognition_result(total_count, predictions, labels)

        plot_example_success_and_failure_case_of_face_recognition(predictions, labels, self._face_data.feature_test.T,
                                                                  f'Face recognition with Alternative Method')

    def test_eigen_space_class_face_recognition(self):
        for (norm_name, norm) in context['norms'].items():
            self._compute_eigen_space_classes_and_evaluate_result(norm_name, norm)
