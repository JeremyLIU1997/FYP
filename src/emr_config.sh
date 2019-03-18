#!/bin/bash

export PS1="\[\e[1;33m\][\u@\h \W]\$\[\e[0m\] "
export PYSPARK_PYTHON=/usr/bin/python3
export PYSPARK_DRIVER_PYTHON=/usr/bin/python3
export GREP_OPTIONS='--color=always'
export GREP_COLOR='1;35;40'

# alias
alias lls="ls -alhF"
alias so="open -a \"Sublime Text\""
alias o="open"
alias swu="softwareupdate"
alias hdls="hdfs dfs -ls"
alias hdrm="hdfs dfs -rm"

mkdir ./src
mkdir -r ./Data/netflix_data/
aws s3 cp s3://al-spark/als_spark_emr.py /home/hadoop/src/
aws s3 cp s3://al-spark/als_spark.py /home/hadoop/src/
aws s3 cp s3://al-spark/pack_tight.py /home/hadoop/src/
aws s3 cp s3://al-spark/load.py /home/hadoop/src/
aws s3 cp s3://al-spark/my_data_3_sorted.txt /home/hadoop/Data/netflix_data/
aws s3 cp s3://al-spark/my_data_10_sorted.txt /home/hadoop/Data/netflix_data/
aws s3 cp s3://al-spark/my_data_30_sorted.txt /home/hadoop/Data/netflix_data/
aws s3 cp s3://al-spark/my_data_80_sorted.txt /home/hadoop/Data/netflix_data/


