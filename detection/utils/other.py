from pathlib import Path


def changeFileNames():
    path = "detection/dataset/machine/images"
    my_file = Path(path)
    q = list(sorted(my_file.iterdir()))

    path = "detection/dataset/machine/labels"
    my_file = Path(path)
    w = list(sorted(my_file.iterdir()))

    for item in zip(q, w):
        new_name = item[0].stem
        new_path = item[1].parent
        my_file = Path(f"{item[1]}")
        my_file.rename(f"{new_path}/{new_name}.txt")