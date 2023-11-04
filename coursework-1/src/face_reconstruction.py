import matplotlib.pyplot as plt
import numpy as np

from common.constants import context
from face_data import FaceData

choice_count = 3
eigen_counts = [3, 10, 50, 100, 200]


class FaceReconstruction:
    def __init__(self, face_data: FaceData):
        self._face_data = face_data

    def _get_sample_faces(self, faces):
        choice_indicies = np.random.choice(faces.shape[0], size=choice_count, replace=False)
        return faces[choice_indicies, :]

    def _reconstruct_face(self, face, num_of_eigen):
        reconstructed_face = np.copy(self._face_data.mean_face)

        for i in range(num_of_eigen):
            target_eigen_vector = self._face_data.eigen_vectors[i]
            a = (face - self._face_data.mean_face) @ target_eigen_vector
            reconstructed_face += a * target_eigen_vector

        return reconstructed_face

    def _reconstruct_faces(self, faces, num_of_eigen):
        return np.array([self._reconstruct_face(face, num_of_eigen) for face in faces])

    def _reshape_face_for_plot(self, face):
        return face.reshape(context['face_row'], context['face_column']).T

    def _reconstruct_and_display_faces(self, faces, title):
        sample_faces = self._get_sample_faces(faces)
        reconstructed_faces = np.array(
            [self._reconstruct_faces(sample_faces, num_of_eigen) for num_of_eigen in eigen_counts]
        )
        all_faces = np.concatenate((sample_faces.reshape(1, choice_count, -1), reconstructed_faces), axis=0)

        plt.figure()
        fig, ax = plt.subplots(len(eigen_counts) + 1, choice_count)
        fig.suptitle(title, fontsize=16)

        row_titles = ['Original'] + [f'Reconstructed with {count} best eigenvectors' for count in eigen_counts]

        for r, ax_row in enumerate(ax):
            for c, ax_col in enumerate(ax_row):
                ax_col.imshow(self._reshape_face_for_plot(all_faces[r][c]), cmap='gist_gray')
                ax_col.axis('off')

        for r, row_title in enumerate(row_titles):
            side = 1 / len(row_titles)
            fig.text(0.02, 1 - (r + 0.5) * side, row_title, fontsize=12)

        plt.tight_layout(rect=(0, 0, 0.9, 0.95))

    def test_reconstruction(self):
        faces_train = self._face_data.feature_train.T
        faces_test = self._face_data.feature_test.T

        self._reconstruct_and_display_faces(faces_train, 'Train face image samples')
        self._reconstruct_and_display_faces(faces_test, 'Test face image samples')
