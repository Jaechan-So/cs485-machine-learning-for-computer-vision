from eigen_face import EigenFace
from face_data import FaceData


def main():
    face_data = FaceData()

    # Q1
    eigen_face = EigenFace(face_data)
    eigen_face.test_dimensionality()


if __name__ == '__main__':
    main()
