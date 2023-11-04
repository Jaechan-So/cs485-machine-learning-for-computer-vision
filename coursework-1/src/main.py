import matplotlib.pyplot as plt

from eigen_face import EigenFace
from eigen_space_class_face_recognition import EigenSpaceClassFaceRecognition
from face_data import FaceData
from face_reconstruction import FaceReconstruction
from nearest_neighbor_face_recognition import NearestNeighborFaceRecognition


def main():
    face_data = FaceData()

    # Q1 - Eigenfaces
    eigen_face = EigenFace(face_data=face_data)
    # eigen_face.test_dimensionality()

    # Q1 - Application of Eigenfaces
    face_reconstruction = FaceReconstruction(face_data=face_data)
    # face_reconstruction.test_reconstruction()

    nearest_neighbor_face_recognition = NearestNeighborFaceRecognition(face_data=face_data)
    # nearest_neighbor_face_recognition.test_nearest_neighbor_recognition()

    eigen_space_class_face_recognition = EigenSpaceClassFaceRecognition(face_data=face_data)
    eigen_space_class_face_recognition.test_eigen_space_class_face_recognition()

    plt.show()


if __name__ == '__main__':
    main()
