from collections.abc import Iterable
from pathlib import Path


def get_root_dir():
    return Path.cwd() / 'coursework-1'


def get_assets_dir():
    return get_root_dir() / 'assets'


def get_face_data_dir():
    return get_assets_dir() / 'face.mat'


def pipe(*funcs):
    def wrapper(*args, **kwargs):
        result = funcs[0](*args, **kwargs)

        for func in funcs[1:]:
            if isinstance(result, Iterable):
                result = func(*result)
            else:
                result = func(result)

        return result

    return wrapper
