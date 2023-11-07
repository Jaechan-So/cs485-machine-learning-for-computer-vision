import numpy as np

from common.operations import get_eigen_projections, get_lda_eigen


class PCALDAModel:
    def __init__(self, pca_features_train, pca_eigen_vectors, lda_features_train, lda_labels_train, mean_face, m_lda):
        self._mean_face = mean_face
        self._pca_eigen_vectors = pca_eigen_vectors

        pca_projections_train = get_eigen_projections(pca_features_train, mean_face, pca_eigen_vectors)
        self._pca_projections_mean_train = np.mean(pca_projections_train, axis=0)
        lda_eigen_values_train, self._lda_eigen_vectors_train = get_lda_eigen(lda_features_train, lda_labels_train,
                                                                              pca_eigen_vectors.T)
        self._lda_eigen_vectors_train = self._lda_eigen_vectors_train[:m_lda]
        self._lda_projections_train = get_eigen_projections(pca_projections_train, self._pca_projections_mean_train,
                                                            self._lda_eigen_vectors_train)

    def feed(self, features_test):
        pca_projections_test = get_eigen_projections(features_test, self._mean_face,
                                                     self._pca_eigen_vectors)
        lda_projections_test = get_eigen_projections(pca_projections_test, self._pca_projections_mean_train,
                                                     self._lda_eigen_vectors_train)
        return self._lda_projections_train, lda_projections_test
