import csv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyzedataset(f):
    reader = csv.DictReader(f, delimiter=",")
    artists = []
    for line in reader:
        artists.append(line['artist'])

    freq_map = Counter(artists)
    freqs = sorted(list(freq_map.values()), reverse=True)
    print(np.mean(freqs))
    print(np.std(freqs))

    x = [i for i in range(len(freqs)) ]

    # fig, (ax1) = plt.subplots(1,1)
    # ax1.fill_between(x, 0, freqs, color="orange")
    # plt.margins(0)
    # plt.show()


if __name__ == "__main__":
    with open ("train_info.csv") as f:
        analyzedataset(f)
