#!/usr/local/bin/python3

# This file is responsible for loading my_data.txt
# which is the pre-processed file from the original
# combined_data_1.txt
# my_data.txt contains the first 200 movies

# imports
from tqdm import tqdm
import sys


with open("../Data/netflix_data/my_data.txt","r") as f:
