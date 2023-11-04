import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from common.constants import context
from common.decorators import measure_time
from face_data import FaceData


class NearestNeighborFaceRecognition:
    @measure_time(tag='Q1 - Application of Eigenfaces - b: Computing Projections')
    def _compute_projections(self):
        return np.array(
            [
                [
                    (face_train - self._face_data.mean_face).T @ eigen_vector
                    for eigen_vector in self._face_data.eigen_vector
                ]
                for face_train in self._face_data.feature_train.T
            ]
        )

    def __init__(self, face_data: FaceData):
        self._face_data = face_data
        self._projections = self._compute_projections()

    def _compute_projection_of_face(self, eigen_vectors, face):
        return (face - self._face_data.mean_face).T @ eigen_vectors

    def _find_nearest_neighbor(self, eigen_vectors, norm, face):
        nearest_neighbor_index = np.argmin(
            norm(self._compute_projection_of_face(eigen_vectors, face) - self._projections))
        return self._face_data.label_train[nearest_neighbor_index]

    def _evaluate_result(self, predictions, labels, num_of_eigen, norm_name):
        total_count = self._face_data.feature_test.T.shape[0]
        accuracy = np.count_nonzero(predictions == labels) / total_count
        print(f'With {num_of_eigen} eigenvalues and {norm_name}, accuracy = {(accuracy * 100):.2f}%')

        cm = confusion_matrix(labels, predictions)

        plt.figure()
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    @measure_time(tag='Q1 - Application of Eigenfaces - b: Computing Nearest Neighbors')
    def _compute_nearest_neighbors(self, eigen_vectors, norm):
        return np.array([self._find_nearest_neighbor(eigen_vectors, norm, face) for face in
                         self._face_data.feature_test.T])

    def _compute_nearest_neighbors_and_evaluate_result(self, num_of_eigen, norm_name, norm):
        eigen_vectors = self._face_data.eigen_vector[:num_of_eigen].T
        predictions = self._compute_nearest_neighbors(eigen_vectors, norm)
        labels = self._face_data.label_test

        self._evaluate_result(predictions, labels, num_of_eigen, norm_name)

    def test_nearest_neighbor_recognition(self):
        for (num_of_eigen, norm_name, norm) in context.test_parameters:
            self._compute_nearest_neighbors_and_evaluate_result(num_of_eigen, norm_name, norm)
