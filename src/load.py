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
		print("Constructing Sparse_Column...")

		# construct sparse_column ###################################################
		R_sparse_column = [[0,0],[],[]] # [matrix dimension, index (2D), value (2D)]
		R_sparse_column[0][0] = user_count
		R_sparse_column[0][1] = movie_count # assign array dimensions		
		# Note the index here is by column. i.e., index[0] is first column of R
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

		R_sparse_column[1] = index
		R_sparse_column[2] = values
		print("Success!\n")

		print("Constructing Sparse_Row...")
		# construct sparse_row ###################################################
		index = None; values = None
		R_sparse_row = [[0,0],[],[]] # [matrix dimension, index (2D), value (2D)]
		R_sparse_row[0][0] = user_count
		R_sparse_row[0][1] = movie_count # assign array dimensions		
		# Note the index here is by row. i.e., index[0] is first row of R
		index = [[] for i in range(user_count)] # initialize 2D array
		values = [[] for i in range(user_count)] # initialize 2D array

		length = []
		for i in range(movie_count): # assign the lengths of the movie columns
			length.append(len(R_sparse_column[1][i]))
		progress = [0 for i in range(movie_count)] # track progress of each column

		remaining_columns = movie_count # track number of columns that has not finished scanning
		remaining_bool = [True for i in range(movie_count)]
		while True:
			min_ = None
			min_index = None
			if remaining_columns == 0:
				break
			for i in range(movie_count):
				if (progress[i] != length[i]) and (min_ == None or R_sparse_column[1][i][progress[i]] < min_):
					min_index = i
					min_ = R_sparse_column[1][i][progress[i]]
					progress[i] += 1
					# check if the column has finished
					if progress[i] == length[i]:
						if remaining_bool[i] == True:
							remaining_bool[i] = False
							remaining_columns -= 1
			index[min_].append(min_index)
			values[min_].append(R_sparse_column[2][min_index][progress[min_index] - 1])

		print("Success!\n")

	return R_sparse_column, R_sparse_row


if __name__ == '__main__':
	# load(input)
	input = "../Data/netflix_data/my_data_30_sorted.txt"
	load_as_sparse(input)