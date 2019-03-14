#!/usr/local/bin/python3

# This file is responsible for loading netflix_data
# Since the user IDs are gapped, originally ranging
# from 0 all the way to 260,0000, this script
# maps the range to a smaller range tightly packed
# without any gaps. The unique users contained in 
# combined_data_1.txt is around 470,000

# Due to memory limitation, this scripts load only
# the ratings for the first 200 movies, which takes
# up apprx. 2.2GB in non-sparse representation.

# imports
from tqdm import tqdm
import sys

if len(sys.argv) == 1: # if no argument
	print("Takes 1 argument. Exit.")
	exit(1)
# adjust parameters here to generate different
# sizes of datasets
movie_index = 0
users = []
number_of_movies_chosen = int(sys.argv[1])
output_file = "../Data/netflix_data/my_data_" + str(number_of_movies_chosen) + "_sorted.txt"

################################################################
with open("../Data/netflix_data/combined_data_1.txt",'r') as f:
	for line in f:
		if line[-2] == ":":
			if int(line.split(":")[0]) == number_of_movies_chosen + 1:
				break
			continue
		users.append(int(line.split(",")[0]))
print("Getting unique...")
users = list(set(users))
print("Unique users: " + str(len(users)))
print("Sorting...")
users.sort()
id_map_dict = {}
for i in range(len(users)):
	id_map_dict[users[i]] = i


# generate ordered
with open("../Data/netflix_data/combined_data_1.txt",'r') as f:
	with open(output_file, "w+") as output:
		ratings = []
		for i in range(number_of_movies_chosen):
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
			
