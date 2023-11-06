import numpy as np

from common.operations import get_eigen_projections, get_lda_eigen


class PCALDAModel:
    def __init__(self, features_train, labels_train, mean_face, eigen_vectors, m_pca, m_lda):
        self._mean_face = mean_face
        self._eigen_vectors = eigen_vectors
        self._m_pca = m_pca
        self._m_lda = m_lda

        pca_projections_train = get_eigen_projections(features_train, mean_face, eigen_vectors[:m_pca])
        self._pca_projections_mean_train = np.mean(pca_projections_train, axis=0)
        lda_eigen_values_train, self._lda_eigen_vectors_train = get_lda_eigen(features_train, labels_train,
                                                                              m_pca)
        self._lda_projections_train = get_eigen_projections(pca_projections_train, self._pca_projections_mean_train,
                                                            self._lda_eigen_vectors_train[:m_lda])

    def feed(self, features_test):
        pca_projections_test = get_eigen_projections(features_test, self._mean_face,
                                                     self._eigen_vectors[:self._m_pca])
        lda_projections_test = get_eigen_projections(pca_projections_test, self._pca_projections_mean_train,
                                                     self._lda_eigen_vectors_train[:self._m_lda])
        return self._lda_projections_train, lda_projections_test