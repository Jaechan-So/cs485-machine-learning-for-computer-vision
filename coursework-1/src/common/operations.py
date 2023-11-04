import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.metrics import confusion_matrix


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


def get_pca_eigen(features, data_count):
    eigen_values, eigen_vectors = low_dimension_pca(features, data_count)
    eigen_vectors = eigen_vectors.T
    sorted_eigen_values, sorted_eigen_vectors = sort_eigen(eigen_values, eigen_vectors)
    high_dimension_eigen_vectors = (features @ sorted_eigen_vectors.T).T
    return sorted_eigen_values, high_dimension_eigen_vectors


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


def lda(features, labels):
    dimension = features.shape[1]
    total_mean = np.mean(features, axis=0)
    class_group = group_by_class(features, labels)

    within_class_scatter_matrix = get_within_class_scatter_matrix(class_group, dimension)
    between_class_scatter_matrix = get_between_class_scatter_matrix(class_group, dimension, total_mean)

    return np.linalg.eig(np.linalg.inv(within_class_scatter_matrix) @ between_class_scatter_matrix)


def get_lda_eigen(features, labels):
    eigen_values, eigen_vectors = lda(features, labels)
    eigen_vectors = eigen_vectors.T
    sorted_eigen_values, sorted_eigen_vectors = sort_eigen(eigen_values, eigen_vectors)
    return sorted_eigen_values, sorted_eigen_vectors


def get_eigen_projections(features, mean, num_of_eigen, eigen_vectors):
    return (features - mean) @ eigen_vectors.T[:, :num_of_eigen]


def get_nearest_neighbor(projections_train, projections_test, labels_train, norm):
    return [
        labels_train[np.argmin(np.array([norm(diff) for diff in projection_test - projections_train]))]
        for projection_test in projections_test
    ]
