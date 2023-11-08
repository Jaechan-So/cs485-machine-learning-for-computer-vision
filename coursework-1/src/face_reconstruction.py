import matplotlib.pyplot as plt
import numpy as np

from common.constants import norms
from common.operations import get_reconstructions, reshape_face_for_plot
from face_data import FaceData

choice_count = 3
eigen_counts = [3, 10, 50, 100, 200]


class FaceReconstruction:
    def __init__(self, face_data: FaceData):
        self._face_data = face_data

    def _get_sample_faces(self, faces):
        choice_indicies = np.random.choice(faces.shape[0], size=choice_count, replace=False)
        return faces[choice_indicies, :]

    def _reconstruct_faces(self, faces, num_of_eigen):
        return get_reconstructions(faces, self._face_data.mean_face, self._face_data.eigen_vectors[:num_of_eigen])

    def _reconstruct_and_display_faces(self, faces, title):
        sample_faces = self._get_sample_faces(faces)
        reconstructed_faces = np.array(
            [self._reconstruct_faces(sample_faces, num_of_eigen) for num_of_eigen in eigen_counts]
        )
        all_faces = np.concatenate((sample_faces.reshape(1, choice_count, -1), reconstructed_faces), axis=0)

        plt.figure()
        fig, ax = plt.subplots(len(eigen_counts) + 1, choice_count)
        fig.suptitle(title, fontsize=16)

        row_titles = ['Original'] + [f'M = {count}' for count in eigen_counts]

        for r, ax_row in enumerate(ax):
            for c, ax_col in enumerate(ax_row):
                ax_col.imshow(reshape_face_for_plot(all_faces[r][c]), cmap='gist_gray')
                ax_col.axis('off')

        for r, row_title in enumerate(row_titles):
            side = 1 / len(row_titles)
            fig.text(0.02, (1 - (r + 0.5) * side) * 0.85, row_title, fontsize=12)

        plt.tight_layout(rect=(0, 0, 0.9, 0.95))

        face_differences = sample_faces - reconstructed_faces
        for eigen_count, diff in zip(eigen_counts, face_differences):
            theoretical_reconstruction_error = np.sum(self._face_data.eigen_values[eigen_count:])

            face_count = diff.shape[0]
            experimental_reconstruction_error = np.sum(np.array([norms['L2 Norm'](d) ** 2 for d in diff])) / face_count
            print(
                f'{title} with M = {eigen_count}: theoretical - {theoretical_reconstruction_error}, experimental: {experimental_reconstruction_error}')

    def test_reconstruction(self):
        faces_train = self._face_data.feature_train.T
        faces_test = self._face_data.feature_test.T

        self._reconstruct_and_display_faces(faces_train, 'Train face image reconstruction')
        self._reconstruct_and_display_faces(faces_test, 'Test face image reconstruction')
