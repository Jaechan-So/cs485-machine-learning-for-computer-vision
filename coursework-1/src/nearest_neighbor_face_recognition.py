import numpy as np

from common.constants import context
from common.decorators import measure_time, separate_logs
from common.operations import evaluate_face_recognition_result
from face_data import FaceData


class NearestNeighborFaceRecognition:
    @measure_time(tag='Q1 - Application of Eigenfaces - b: Computing Projections')
    def _compute_projections(self):
        return np.array(
            [
                ((face_train - self._face_data.mean_face).reshape(1, -1) @ self._face_data.eigen_vector.T).reshape(-1)
                for face_train in self._face_data.feature_train.T
            ]
        )

    def __init__(self, face_data: FaceData):
        self._face_data = face_data
        self._projections = self._compute_projections()

    def _compute_projection_of_face(self, eigen_vectors, face):
        return (face - self._face_data.mean_face).reshape(1, -1) @ eigen_vectors

    def _find_nearest_neighbor(self, eigen_vectors, norm, face):
        projection_of_face = self._compute_projection_of_face(eigen_vectors, face)
        projection_of_features = self._projections[:, :eigen_vectors.shape[1]]
        norm_differences = np.array(
            [norm(projection_of_face, projection_of_feature) for projection_of_feature in projection_of_features])
        nearest_neighbor_index = np.argmin(norm_differences)
        return self._face_data.label_train.T.reshape(-1)[nearest_neighbor_index]

    @measure_time(tag='Q1 - Application of Eigenfaces - b: Computing Nearest Neighbors')
    def _compute_nearest_neighbors_predictions(self, eigen_vectors, norm):
        return np.array([self._find_nearest_neighbor(eigen_vectors, norm, face) for face in
                         self._face_data.feature_test.T])

    @separate_logs
    def _compute_nearest_neighbors_and_evaluate_result(self, num_of_eigen, norm_name, norm):
        print(f'Compute nearest neighbors and evaluate results with # of eigenvectors: {num_of_eigen}, {norm_name}')
        eigen_vectors = self._face_data.eigen_vector[:num_of_eigen].T
        predictions = self._compute_nearest_neighbors_predictions(eigen_vectors, norm)
        labels = self._face_data.label_test.reshape(-1)

        total_count = self._face_data.feature_test.T.shape[0]
        evaluate_face_recognition_result(total_count, predictions, labels)

    def test_nearest_neighbor_recognition(self):
        for (num_of_eigen, norm_name, norm) in context['test_parameters']:
            self._compute_nearest_neighbors_and_evaluate_result(num_of_eigen, norm_name, norm)
