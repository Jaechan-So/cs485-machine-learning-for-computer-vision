from common.constants import context
from common.decorators import measure_time, separate_logs
from common.operations import evaluate_face_recognition_result, get_eigen_projections, get_nearest_neighbor
from face_data import FaceData


class NearestNeighborFaceRecognition:

    def __init__(self, face_data: FaceData):
        self._face_data = face_data

    @measure_time(tag='Q1 - Application of Eigenfaces - b: Computing Nearest Neighbors')
    def _compute_nearest_neighbors_predictions(self, num_of_eigen, norm):
        pca_projections_train = get_eigen_projections(self._face_data.feature_train.T, self._face_data.mean_face,
                                                      num_of_eigen, self._face_data.eigen_vectors)
        pca_projections_test = get_eigen_projections(self._face_data.feature_test.T, self._face_data.mean_face,
                                                     num_of_eigen, self._face_data.eigen_vectors)
        return get_nearest_neighbor(pca_projections_train, pca_projections_test,
                                    self._face_data.label_train.reshape(-1), norm)

    @separate_logs
    def _compute_nearest_neighbors_and_evaluate_result(self, num_of_eigen, norm_name, norm):
        print(f'Compute nearest neighbors and evaluate results with # of eigenvectors: {num_of_eigen}, {norm_name}')
        predictions = self._compute_nearest_neighbors_predictions(num_of_eigen, norm)
        labels = self._face_data.label_test.reshape(-1)

        total_count = self._face_data.feature_test.T.shape[0]
        evaluate_face_recognition_result(total_count, predictions, labels)

    def test_nearest_neighbor_recognition(self):
        for (num_of_eigen, norm_name, norm) in context['pca_test_parameters']:
            self._compute_nearest_neighbors_and_evaluate_result(num_of_eigen, norm_name, norm)
