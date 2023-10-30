from pathlib import Path


def get_root_dir():
    return Path.cwd() / 'coursework-1'


def get_assets_dir():
    return get_root_dir() / 'assets'


def get_face_data_dir():
    return get_assets_dir() / 'face.mat'
