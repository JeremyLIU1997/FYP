#!/usr/local/bin/python3

# This file is responsible for converting txt data
# into binary for faster loading

# imports
from tqdm import tqdm
import sys
import numpy as np

# settings
np.set_printoptions(threshold=np.nan)


#========================================================
def load(input):
	user_count = 0
	movie_count = 0
	exist_dictionary = {}
	with open(input,"r") as f:
		for line in f:
			if line[-2] == ":":
				movie_count += 1
				continue
			split = line.split(",")
			id = int(split[0])
			if id not in exist_dictionary.keys():
				exist_dictionary[id] = True
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

# this method loads the matrix in sparse reprensentation
def load_as_sparse(input):
	user_count = 0
	movie_count = 0
	exist_dictionary = {}
	with open(input,"r") as f:
		for line in f:
			if line[-2] == ":":
				movie_count += 1
				continue
			split = line.split(",")
			id = int(split[0])
			if not (id in exist_dictionary.keys()):
				exist_dictionary[id] = True
				user_count += 1

		print("#users: " + str(user_count))
		print("#movies: " + str(movie_count))
		print("Constructing Sparse...")

		R_sparse = [[0,0],[],[]] # [matrix dimension, index (2D), value (2D)]
		R_sparse[0][0] = user_count; R_sparse[0][1] = movie_count # assign array dimensions		
		# Note the index here is by column. i.e., index[0] is first column of movie
		index = [[] for i in range(movie_count)] # initialize 2D array
		values = [[] for i in range(movie_count)] # initialize 2D array
		f.seek(0)

		current_index = 0
		for line in f:
			if line[-2] == ":":
				current_index = int(line.split(":")[0]) - 1
				continue
			split = line.split(",")
			index[current_index].append(int(split[0]))
			values[current_index].append(float(split[1]))

		R_sparse[1] = index
		R_sparse[2] = values
		print("======================================")
		print("Index: " + str(index[-1]))
		print("Values: " + str(values[-1]))

	return R_sparse


if __name__ == '__main__':
	# load(input)
	input = "../Data/netflix_data/my_data_30_sorted.txt"
	load_as_sparse(input)