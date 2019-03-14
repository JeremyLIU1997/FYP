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
input = "../Model/item.txt"

#========================================================
def load(input):
	user_count = 0
	movie_count = 0
	with open(input,"r") as f:
		for line in f:
			if line[-2] == ":":
				movie_count += 1
				continue
			user_count += 1
			
		R = np.zeros((user_count, movie_count)).astype(float)
		print("#users: " + str(user_count))
		print("#movies: " + str(movie_count))
		f.seek(0)
		current_movie = 0
		for line in f:
			if line[-2] == ":":
				current_movie = int(line.split(":")[0])
				continue
			split = line.split(",")
			R[int(split[0])][current_movie - 1] = int(split[1])
	return R.astype(float)

def load_as_sparse():
	user_count = 0
	movie_count = 0
	with open(input,"r") as f:
		for line in f:
			if line[-2] == ":":
				movie_count += 1
				continue
			user_count += 1
		

		R_sparse = [[],[],[],[]] # matrix dimension, x_index, y_index, value
		R_sparse[0][0] = user_count; R_sparse[0][1] = movie_count # assign array dimensions
		print("#users: " + str(user_count))
		print("#movies: " + str(movie_count))
		f.seek(0)
		current_movie = 0
		for line in f:
			if line[-2] == ":":
				current_movie = int(line.split(":")[0])
				continue
			split = line.split(",")
			R[int(split[0])][current_movie - 1] = int(split[1])
	return R.astype(float)	


if __name__ == '__main__':
	load(input)