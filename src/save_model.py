#!/usr/local/bin/python3

# This file is responsible for converting txt data
# into binary for faster loading

# imports
from tqdm import tqdm
import sys
import numpy as np

# settings
np.set_printoptions(threshold=np.nan)

# adjustable parameters

#========================================================
def save(array, output):
	with open(output, "w+") as f:
		for i in range(len(array)):
			f.write(str(i + 1) + ":\n")
			for item in array[i]:
				f.write(str(item) + "\n")
