from eigen_face import EigenFace
from face_data import FaceData
from face_reconstruction import FaceReconstruction


def main():
    face_data = FaceData()

    # Q1 - Eigenfaces
    eigen_face = EigenFace(face_data=face_data)
    # eigen_face.test_dimensionality()

    # Q1 - Application of Eigenfaces
    face_reconstruction = FaceReconstruction(face_data=face_data)
    face_reconstruction.test_reconstruction()


if __name__ == '__main__':
    main()
