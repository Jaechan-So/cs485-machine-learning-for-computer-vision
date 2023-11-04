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


def evaluate_face_recognition_result(total_count, predictions, labels):
    accuracy = np.count_nonzero(predictions == labels) / total_count
    print(f'Accuracy: {(accuracy * 100):.2f}%')

    cm = confusion_matrix(labels, predictions)

    plt.figure()
    seaborn.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
