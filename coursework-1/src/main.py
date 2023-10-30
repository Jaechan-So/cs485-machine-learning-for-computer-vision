from eigen_face import EigenFace
from face_data import FaceData


def main():
    face_data = FaceData()

    # Q1 - Eigenfaces
    eigen_face = EigenFace(face_data)
    eigen_face.test_dimensionality()

    # Q1 - Application of Eigenfaces
    eigen_face.preprocess_eigen()
    eigen_face.test_reconstruction()


if __name__ == '__main__':
    main()
