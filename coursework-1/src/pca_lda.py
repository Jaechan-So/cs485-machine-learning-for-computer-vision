import numpy as np

from common.constants import context
from common.decorators import separate_logs
from common.operations import get_nearest_neighbor, \
    evaluate_face_recognition_result, plot_example_success_and_failure_case_of_face_recognition
from face_data import FaceData
from pca_lda_model import PCALDAModel


def get_most_frequent(x):
    return np.bincount(x).argmax()


class PCALDA:
    def __init__(self, face_data: FaceData):
        self._face_data = face_data

    @separate_logs
    def _compute_pca_lda_nearest_neighbor_recognition(self, feature_train, feature_test, label_train, label_test,
                                                      mean_face, eigen_vectors, m_pca, m_lda, norm_name, norm):
        print(f'Compute PCA-LDA with M_pca: {m_pca}, M_lda: {m_lda}, {norm_name}')

        model = PCALDAModel(feature_train, eigen_vectors[:m_pca], feature_train, label_train, mean_face, m_lda)
        lda_projections_train, lda_projections_test = model.feed(feature_test)

        predictions = get_nearest_neighbor(lda_projections_train, lda_projections_test, label_train, norm)
        labels = label_test

        evaluate_face_recognition_result(label_test.shape[0], predictions, labels)

        plot_example_success_and_failure_case_of_face_recognition(predictions, labels, self._face_data.feature_test.T,
                                                                  f'Face recognition with PCA-LDA-NN classification with M_pca = {m_pca}, M_lda = {m_lda}')

    def test_pca_lda(self):
        for m_pca, m_lda, norm_name, norm in context['pca_lda_test_parameters']:
            self._compute_pca_lda_nearest_neighbor_recognition(
                self._face_data.feature_train.T,
                self._face_data.feature_test.T,
                self._face_data.label_train.reshape(-1),
                self._face_data.label_test.reshape(-1),
                self._face_data.mean_face,
                self._face_data.eigen_vectors,
                m_pca,
                m_lda,
                norm_name,
                norm,
            )

    def _random_sample_class(self, class_count):
        features_train = self._face_data.feature_train.T
        labels_train = self._face_data.label_train.reshape(-1)

        class_nums = np.unique(labels_train)
        class_choices = np.random.choice(class_nums, size=class_count, replace=False)
        indices = [index for index in range(features_train.shape[0]) if labels_train[index] in class_choices]

        return features_train[indices], labels_train[indices]

    def _get_models_from_pca_lda_with_bagging(self, num_of_models, class_count, m_pca, m_lda):
        models = []

        for _ in range(num_of_models):
            sampled_features_train, sampled_labels_train = self._random_sample_class(class_count)

            model = PCALDAModel(self._face_data.feature_train.T, self._face_data.eigen_vectors[:m_pca],
                                sampled_features_train,
                                sampled_labels_train, self._face_data.mean_face,
                                m_lda)
            models.append(model)

        return models

    def _get_models_from_pca_lda_with_feature_space_random_sampling(self, num_of_models, m_0, m_1, m_lda):
        models = []

        fixed_eigen_vectors = self._face_data.eigen_vectors[:m_0]
        eigen_count = self._face_data.eigen_vectors.shape[0]
        for _ in range(num_of_models):
            indices = np.random.choice(range(m_0, eigen_count), size=m_1, replace=False)
            indices.sort()
            sampled_eigen_vectors = self._face_data.eigen_vectors[indices]
            total_eigen_vectors = np.concatenate((fixed_eigen_vectors, sampled_eigen_vectors), axis=0)

            model = PCALDAModel(self._face_data.feature_train.T, total_eigen_vectors,
                                self._face_data.feature_train.T, self._face_data.label_train.reshape(-1),
                                self._face_data.mean_face, m_lda)
            models.append(model)

        return models

    @separate_logs
    def _evaluate_ensembled_model(self, models):
        features_test = self._face_data.feature_test.T
        labels_train = self._face_data.label_train.reshape(-1)
        labels_test = self._face_data.label_test.reshape(-1)
        norm = context['norms']['L1 Norm']

        local_predictions_per_models = []
        error_individual = 0

        for model in models:
            lda_projections_train, lda_projections_test = model.feed(features_test)
            local_predictions = get_nearest_neighbor(lda_projections_train, lda_projections_test, labels_train,
                                                     norm)
            error_individual += np.count_nonzero(local_predictions != labels_test)
            local_predictions_per_models.append(local_predictions)
        local_predictions_per_models = np.array(local_predictions_per_models).T
        error_individual /= len(models)

        predictions = []
        for local_predictions in local_predictions_per_models:
            predictions.append(get_most_frequent(local_predictions))
        predictions = np.array(predictions)

        error_committee = np.count_nonzero(predictions != labels_test)

        total_count = labels_test.shape[0]
        print(f'Error of the committee machine: {(error_committee / total_count) * 100:.2f}')
        print(f'Average error of indivdiual models: {(error_individual / total_count) * 100:.2f}')

        evaluate_face_recognition_result(total_count, predictions, labels_test)

    def test_pca_lda_random_sampling_manipulate_training_data(self):
        print('Test PCA-LDA with random sampling of training data')

        num_of_models = 20
        class_counts = [10, 25, 50]
        m_lda = 30
        m_0 = 30
        m_1 = 30
        m_pca = m_0 + m_1

        feature_space_models = self._get_models_from_pca_lda_with_feature_space_random_sampling(num_of_models, m_0, m_1,
                                                                                                m_lda)
        for class_count in class_counts:
            training_data_models = self._get_models_from_pca_lda_with_bagging(num_of_models, class_count, m_pca,
                                                                              m_lda)
            total_models = feature_space_models + training_data_models
            print(f'Bagging training data, for {class_count} data')
            self._evaluate_ensembled_model(total_models)

    def test_pca_lda_random_sampling_manipulate_feature_space(self):
        print('Test PCA-LDA with random sampling of feature space')

        num_of_models = 20
        class_count = 20
        m_lda = 30
        m_pca = 60
        m_0_candidates = [3, 5, 10, 25, 50]
        m_1_candidates = [m_pca - m for m in m_0_candidates]

        bagging_models = self._get_models_from_pca_lda_with_bagging(num_of_models, class_count, m_pca, m_lda)

        for m_0, m_1 in zip(m_0_candidates, m_1_candidates):
            feature_space_models = self._get_models_from_pca_lda_with_feature_space_random_sampling(num_of_models, m_0,
                                                                                                    m_1, m_lda)
            total_models = bagging_models + feature_space_models
            print(f'Random sampling for feature space, with m_0 = {m_0}, m_1 = {m_1}')
            self._evaluate_ensembled_model(total_models)

    def test_pca_lda_random_sampling_manipulate_num_of_base_model(self):
        for m_pca, m_lda, norm_name, norm in context['pca_lda_test_parameters']:
            self._compute_pca_lda_nearest_neighbor_recognition(m_pca, m_lda, norm_name, norm)
