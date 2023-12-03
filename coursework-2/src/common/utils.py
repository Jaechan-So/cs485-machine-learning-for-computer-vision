from pathlib import Path


def get_root_dir():
    return Path.cwd() / "coursework-2"


def get_assets_dir():
    return get_root_dir() / "assets"


def get_caltech_101_dataset_dir():
    return get_assets_dir() / "Caltech_101" / "101_ObjectCategories"
