mpic++ knn.cc -o knn
mpirun -n 3 ./knn 8 10 5 3

# argv[1] = number of queries, argv[2] = number of training instances
# argv[3] = number of columns in training instance, argv[4] = how many neighbors