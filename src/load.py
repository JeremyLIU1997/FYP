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
			progress[min_index] += 1
			# check if the column has finished
			if progress[min_index] == length[min_index]:
				if remaining_bool[min_index] == True:
					remaining_bool[min_index] = False
					remaining_columns -= 1
			index[min_].append(min_index)
			values[min_].append(R_sparse_column[2][min_index][progress[min_index] - 1])
		R_sparse_row[1] = index
		R_sparse_row[2] = values
		print("Success!\n")

	return R_sparse_column, R_sparse_row

def extract_from_sparse(ar_sparse, indices, option):
	if option == "col": # columns will be extracted (from column sparse matrix)
		dense_array = np.zeros((ar_sparse[0][0],len(indices))).astype(float)
		extracted_indices = np.array(ar_sparse[1])[indices]
		extracted_values = np.array(ar_sparse[2])[indices]
		for j in range(len(extracted_indices)):
			for i in range(len(extracted_indices[j])):
				dense_array[extracted_indices[j][i]][j] = extracted_values[j][i]
	elif option == "row": # rows will be extracted (from row sparse matrix)
		dense_array = np.zeros((len(indices),ar_sparse[0][1])).astype(float)
		extracted_indices = np.array(ar_sparse[1])[indices]
		extracted_values = np.array(ar_sparse[2])[indices]
		for i in range(len(extracted_indices)):
			for j in range(len(extracted_values[i])):
				dense_array[i][extracted_indices[i][j]] = extracted_values[i][j]
	else:
		print("Invalid option error. Exit.")
		exit(1)
	return dense_array




if __name__ == '__main__':
	# load(input)
	input = "../Data/netflix_data/my_data_30_sorted.txt"
	# print("By column nonzero: " + str(np.count_nonzero(R,axis=0)))
	# print("By row nonzero: " + str(np.count_nonzero(R,axis=1)))
	R_sparse_column, R_sparse_row = load_as_sparse(input)
	"""
	for i in range(1):
		print("=========Dense: " + str(R[:,i]))
		print("Sparse: " + str(by_column[1][i]) + "\n--> " + str(by_column[2][i]))
		print("length: " + str(len(by_column[1][i])))
	"""

