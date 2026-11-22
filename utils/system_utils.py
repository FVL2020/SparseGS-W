import os


def mkdir_p(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
