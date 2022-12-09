# 474-Project2
Project 2: Implement a distributed algorithm using MPI

Group members:

Johnson Tong jt28@csu.fullerton.edu

Derrick Lee derricklee@csu.fullerton.edu

Miguel Macias miguel6021857@csu.fullerton.edu

# How to run

Parallelized Execution
  
    sh r.sh

Sequential Execution

    sh sequentialExecution.sh

Custom Execution

mpic++ -std=c++11 knn.cc -o knn

mpirun -n 5 ./knn 3000 7000 5 30 testdata.txt

argv[1] = number of queries, argv[2] = number of training instances

argv[3] = number of columns in training instance, argv[4] = how many neighbors

argv[5] = name of textfile for training data
