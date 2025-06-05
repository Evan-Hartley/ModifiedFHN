import csv
import numpy as np
import os

def LoadAndProcessCsv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        row = next(reader)  # Assuming all data is in one row

    # Skip the first two cells (width and height)
    data = row[2:]

    # Convert the first channel (fval[indx]) to floats
    # Each color has 4 values (R, G, B, A), so we pick every 4th value starting from 0 for the R channel
    rchannel_data = np.array([float(data[i]) for i in range(0, len(data), 4)])
    
     # Each color has 4 values (R, G, B, A), so we pick every 4th value starting from 1 for the G channel
    gchannel_data = np.array([float(data[i]) for i in range(1, len(data), 4)])

    # Reshape into a 512x512 array
    u = rchannel_data[:512*512].reshape(512, 512)
    v = gchannel_data[:512*512].reshape(512, 512)

    return [u, v]