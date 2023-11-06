import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.metrics import confusion_matrix

from common.constants import context
from common.utils import pipe


def remove_zero_eigens(eigen_values, eigen_vectors):
    valid_indices = np.where(eigen_values > context['eigen_value_tolerance'])
    return eigen_values[valid_indices], eigen_vectors[:, valid_indices].reshape(-1, valid_indices[0].shape[0])


def pca(matrix, data_count):
    covariance_matrix_train = (matrix @ matrix.T) / data_count
    return np.linalg.eig(covariance_matrix_train)


def low_dimension_pca(matrix, data_count):
    low_dimension_matrix = matrix if matrix.shape[0] < matrix.shape[1] else matrix.T
    return pca(low_dimension_matrix, data_count)


def sort_eigen(eigen_value, eigen_vector):
    eigen = zip(eigen_value, eigen_vector)
    sorted_eigen = sorted(eigen, key=lambda e: -e[0])

    sorted_eigen_value = np.array([e[0] for e in sorted_eigen])
    sorted_eigen_vector = np.array([e[1] for e in sorted_eigen])

    return sorted_eigen_value, sorted_eigen_vector


def normalize_eigen_vectors(eigen_vectors):
    return np.array([eigen_vector / context['norms']['L2 Norm'](eigen_vector) for eigen_vector in eigen_vectors])


def preprocess_eigen(eigen_values, eigen_vectors):
    return pipe(
        remove_zero_eigens,
        lambda eigen_values_inner, eigen_vectors_inner: (eigen_values_inner, eigen_vectors_inner.T),
        sort_eigen,
        lambda eigen_values_inner, eigen_vectors_inner: (
            eigen_values_inner, normalize_eigen_vectors(eigen_vectors_inner)
        ),
    )(eigen_values, eigen_vectors)


def get_pca_eigen(features, data_count):
    eigen_values, eigen_vectors = low_dimension_pca(features, data_count)
    high_dimension_eigen_vectors = features @ eigen_vectors
    return preprocess_eigen(eigen_values, high_dimension_eigen_vectors)


def evaluate_face_recognition_result(total_count, predictions, labels):
    accuracy = np.count_nonzero(predictions == labels) / total_count
    print(f'Accuracy: {(accuracy * 100):.2f}%')

    cm = confusion_matrix(labels, predictions)

    plt.figure()
    seaborn.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')


def group_by_class(features, labels):
    class_group = dict()

    for index, feature in enumerate(features):
        class_num = labels[index]

        if class_num not in class_group:
            class_group[class_num] = dict()
            class_group[class_num]['features'] = []

        class_group[class_num]['features'].append(feature)

    for class_num, data in class_group.items():
        class_group[class_num]['features'] = np.array(class_group[class_num]['features'])
        class_group[class_num]['class_mean'] = np.mean(class_group[class_num]['features'], axis=0)
        class_group[class_num]['total_count'] = class_group[class_num]['features'].shape[0]

    return class_group


def get_within_class_scatter_matrix(class_group, dimension):
    scatter_matrix = np.zeros((dimension, dimension))

    for _, data in class_group.items():
        for feature in data['features']:
            diff = (feature - data['class_mean']).reshape(-1, 1)
            scatter_matrix += (diff @ diff.T)

    return scatter_matrix


def get_between_class_scatter_matrix(class_group, dimension, total_mean):
    scatter_matrix = np.zeros((dimension, dimension))

    for _, data in class_group.items():
        diff = (data['class_mean'] - total_mean).reshape(-1, 1)
        scatter_matrix += (data['total_count'] * (diff @ diff.T))

    return scatter_matrix


def lda(features, labels, pca_eigen_vectors):
    total_count, dimension = features.shape
    total_mean = np.mean(features, axis=0)
    class_group = group_by_class(features, labels)

    within_class_scatter_matrix = get_within_class_scatter_matrix(class_group, dimension)
    between_class_scatter_matrix = get_between_class_scatter_matrix(class_group, dimension, total_mean)

    print(f'Rank of within-class scatter matrix: {np.linalg.matrix_rank(within_class_scatter_matrix)}')
    print(f'Rank of between-class scatter matrix: {np.linalg.matrix_rank(between_class_scatter_matrix)}')

    interpolated_within_class_scatter_matrix = pca_eigen_vectors.T @ within_class_scatter_matrix @ pca_eigen_vectors
    interpolated_between_class_scatter_matrix = pca_eigen_vectors.T @ between_class_scatter_matrix @ pca_eigen_vectors

    return np.linalg.eig(
        np.linalg.inv(interpolated_within_class_scatter_matrix) @ interpolated_between_class_scatter_matrix)


def get_lda_eigen(features, labels, pca_eigen_vectors):
    eigen_values, eigen_vectors = lda(features, labels, pca_eigen_vectors)
    return preprocess_eigen(eigen_values, eigen_vectors)


def get_eigen_projections(features, mean_face, eigen_vectors):
    return (features - mean_face) @ eigen_vectors.T


def get_reconstructions(features, mean_face, eigen_vectors):
    projections = get_eigen_projections(features, mean_face, eigen_vectors)
    return mean_face + (eigen_vectors.T @ projections.T).T


def get_nearest_neighbor(projections_train, projections_test, labels_train, norm):
    return [
        labels_train[np.argmin(np.array([norm(diff) for diff in projection_test - projections_train]))]
        for projection_test in projections_test
    ]


def reshape_face_for_plot(face):
    return face.reshape(context['face_row'], context['face_column']).T


def plot_example_success_and_failure_case_of_face_recognition(predictions, labels, feature_test, title):
    correct_indices = np.where(predictions == labels)[0]
    correct_plot_index = np.random.choice(correct_indices, size=1)[0]

    wrong_indices = np.where(predictions != labels)[0]
    wrong_plot_index = np.random.choice(wrong_indices, size=1)[0]

    plt.figure()
    plt.title(title)
    plt.axis('off')
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(reshape_face_for_plot(feature_test[correct_plot_index]), cmap='gist_gray')
    ax1.axis('off')
    ax1.set_title('Success case')
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(reshape_face_for_plot(feature_test[wrong_plot_index]), cmap='gist_gray')
    ax2.axis('off')
    ax2.set_title('Failure case')
