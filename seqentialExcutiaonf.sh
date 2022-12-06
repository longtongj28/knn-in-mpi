mpic++ -std=c++11 knn.cc -o knn
mpirun -n 1 ./knn 3000 7000 5 30 testdata.txt

# argv[1] = number of queries, argv[2] = number of training instances
# argv[3] = number of columns in training instance, argv[4] = how many neighbors
#argv[5] = name of textfile for training data