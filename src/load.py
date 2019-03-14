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
		f.seek(0)

		R_sparse = [[0,0],[],[]] # [matrix dimension, index (2D), value (2D)]
		R_sparse[0][0] = user_count; R_sparse[0][1] = movie_count # assign array dimensions
		
		index = [[0 for j in range(1)] for i in range(user_count)] # initialize 2D array
		print(np.array(index).shape)
		exit(0)
		f.seek(0)

		for i in range(movie_count):
			line = f.readline()
			output.write(line)
			while True:
				prev_cursor = f.tell()
				line = f.readline()
				if line[-2] == ":":
					ratings.sort()
					for e in ratings:
						output.write(str(e[0]) + "," + e[1] + "\n")
					ratings = []
					break
				split = line.split(",")
				ratings.append([int(id_map_dict[int(split[0])]),split[1]])
			f.seek(prev_cursor) # revert to previous line




		current_movie = 0
		for line in f:
			if line[-2] == ":":
				current_movie = int(line.split(":")[0])
				continue
			split = line.split(",")
			R[int(split[0])][current_movie - 1] = int(split[1])
	return R.astype(float)


if __name__ == '__main__':
	# load(input)
	input = "../Data/netflix_data/my_data_30_sorted.txt"
	load_as_sparse(input)