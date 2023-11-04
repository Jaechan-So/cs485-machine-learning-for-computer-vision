import numpy as np

from common.constants import context
from common.decorators import separate_logs
from common.operations import get_eigen_projections, get_lda_eigen, get_nearest_neighbor, \
    evaluate_face_recognition_result
from face_data import FaceData


class PCALDA:
    def __init__(self, face_data: FaceData):
        self._face_data = face_data

    @separate_logs
    def _compute_pca_lda_nearest_neighbor_recognition(self, m_pca, m_lda, norm_name, norm):
        print(f'Compute PCA-LDA with M_pca: {m_pca}, M_lda: {m_lda}, {norm_name}')

        pca_projections_train = get_eigen_projections(self._face_data.feature_train.T, self._face_data.mean_face,
                                                      m_pca,
                                                      self._face_data.eigen_vectors)
        pca_projections_mean_train = np.mean(pca_projections_train, axis=0)
        lda_eigen_values_train, lda_eigen_vectors_train = get_lda_eigen(pca_projections_train,
                                                                        self._face_data.label_train.reshape(-1))
        lda_projections_train = get_eigen_projections(pca_projections_train, pca_projections_mean_train, m_lda,
                                                      lda_eigen_vectors_train)

        pca_projections_test = get_eigen_projections(self._face_data.feature_test.T, self._face_data.mean_face,
                                                     m_pca, self._face_data.eigen_vectors)
        lda_projections_test = get_eigen_projections(pca_projections_test, pca_projections_mean_train, m_lda,
                                                     lda_eigen_vectors_train)

        predictions = get_nearest_neighbor(lda_projections_train, lda_projections_test,
                                           self._face_data.label_train.reshape(-1), norm)
        labels = self._face_data.label_test.reshape(-1)

        evaluate_face_recognition_result(self._face_data.label_test.shape[1], predictions, labels)

    def test_pca_lda(self):
        for m_pca, m_lda, norm_name, norm in context['pca_lda_test_parameters']:
            self._compute_pca_lda_nearest_neighbor_recognition(m_pca, m_lda, norm_name, norm)
