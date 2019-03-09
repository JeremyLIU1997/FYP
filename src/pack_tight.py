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

# adjust parameters here to generate different
# sizes of datasets
movie_index = 0
users = []
number_of_movies_chosen = 80
output_file = "../Data/netflix_data/my_data_" + str(number_of_movies_chosen) + ".txt"

################################################################
with open("../Data/netflix_data/combined_data_1.txt",'r') as f:
	for line in tqdm(f):
		if line[-2] == ":":
			if movie_index == number_of_movies_chosen:
				break;
			movie_index += 1
			continue
		users.append(int(line.split(",")[0]))
print("Getting unique...")
users = list(set(users))
print("Total users: " + str(len(users)))
print("Sorting...")
users.sort()
id_map_dict = {}
for i in range(len(users)):
	id_map_dict[users[i]] = i

with open("../Data/netflix_data/combined_data_1.txt",'r') as f:
	with open(output_file, "w+") as output:
			for line in tqdm(f):
				if line[-2] == ":":
					if int(line.split(":")[0]) == number_of_movies_chosen:
						break;
					output.write(line)
					continue
				split = line.split(",")
				print("Key: " + str(split[0]))
				output.write(str(id_map_dict[int(split[0])]) + "," + split[1] + "\n")

