import os
import shutil
from collections import defaultdict
import csv
import numpy as np
import operator


BASE_TRAIN_PATH = "/home/abhipanda/CNNpainter/data/PROCESSED_DATA/TRAIN"
BASE_VALIDATION_PATH = "/home/abhipanda/CNNpainter/data/PROCESSED_DATA/VALIDATION"


def trim_data(f):
    reader = csv.DictReader(f, delimiter=",")
    artists = []
    for line in reader:
        artists.append(line['artist'])
    freqs = defaultdict(int)
    for artist in artists:
        freqs[artist] += 1

    sorted_freqs = sorted(freqs.items(), key=operator.itemgetter(1))
    final_list = list(reversed(sorted_freqs))
    delete_artists = []
    for u,v in final_list[300:]:
        delete_artists.append(u)

    for artist in delete_artists:
        del_path1 = os.path.join(BASE_TRAIN_PATH, artist)
        del_path2 = os.path.join(BASE_VALIDATION_PATH, artist)
        try:
            shutil.rmtree(del_path1)
            shutil.rmtree(del_path2)
        except:
            print("1 less thing to worry about !!")


if __name__ == "__main__":
    with open ("train_info.csv") as f:
        trim_data(f)
