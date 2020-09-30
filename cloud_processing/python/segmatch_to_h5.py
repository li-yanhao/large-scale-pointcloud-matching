# Basically this script transforms the data of segmatch to custom .h5 file containing segments
# input: segments_database.csv
# output: segments.h5

import csv
import numpy as np
from tqdm import tqdm


def from_segmatch_to_npy(csv_filename):
    database = []
    segment = []
    initialized = False
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        first_row = next(reader)
        prev_id, x, y, z = first_row
        segment.append(np.array([x, y, z]))
        for row in tqdm(reader):
            id, x, y, z = row
            if prev_id == id:
                segment.append(np.array([x,y,z]))
            else:
                database.append(np.array(segment))
                segment = []
            prev_id = id
    print(len(database))

if __name__ == '__main__':
    csv_filename = '/media/admini/My_data/0629/segmatch_dir/segments_database.csv'
    from_segmatch_to_npy(csv_filename)