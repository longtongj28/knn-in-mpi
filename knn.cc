#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <bits/stdc++.h>
#include <utility>
#include <chrono>
#include <fstream>
#include <string>
using namespace std;

// User provides number of training data points and number of query points
class SortDistances {
    public:
        bool operator()(std::pair<int, double> const & left, std::pair<int, double> const & right) {
            return left.second < right.second;
        }
};
class MostCommonClass {
    public:
        bool operator()(std::pair<int, int> const & left, std::pair<int, int> const & right) {
            return left.second < right.second;
        }
};
class File{
  public:
  int trainingData[20][5];
  int query[20][5];
};

int KNN(int training_points[][5],  int query[], int training_row_size, int training_col_size, int k, int rank) {
    // Calculate and store all of the distances between query and data points
    vector<double> neighbor_dists;
    for (int i = 0; i < training_row_size; i++) {
        // Only calculate distances from the non-target features
        int total = 0;
        for (int j = 0; j < training_col_size-1; j++) {
            total += pow(training_points[i][j] - query[j], 2);
        }
        neighbor_dists.push_back(sqrt(total));                                        
    }

    // Map each distance to it's respective index in the training set before sorting.
    vector< pair<int, double> > index_to_distances;
    for (int i = 0; i < neighbor_dists.size(); i++) {
        pair<int, double> index_to_dist(i, neighbor_dists[i]);
        index_to_distances.push_back(index_to_dist);
    }

    // Sorting each distance_to_index in ascending order with respect to distance
    sort(index_to_distances.begin(), index_to_distances.end(), SortDistances());
  
    // Count the number of each class in the k-nearest-neighbors
    map<int, int> all_classes_to_count;
    for (int i = 0; i < k; ++i) {
        const int& datapoint_index = index_to_distances[i].first;
        const int& classification = training_points[datapoint_index][training_col_size-1];
        all_classes_to_count[classification]++;
    }

    // Get the largest element based on how many occurances of that class there are
    std::map<int, int>::iterator largest_class = max_element(all_classes_to_count.begin(), all_classes_to_count.end(), MostCommonClass());
    // cout << "largest class " << largest_class->first << " has " << largest_class->second << endl; 
    return largest_class->first;
}
File parser(string fileName){
  fstream file;
  string filename = fileName;
  int buffer;
  int flag;
  File parsedData;
  file.open(filename, ios::in);

  if(file.is_open()){
    cout << "File: " << filename << " has been opened." << endl;
    while(file >> flag){
      for (int x = 0; x < 10; x++){
        for (int y = 0; y < 5; y++){
          if(file >> buffer){
            parsedData.trainingData[x][y] = buffer;
          }
        }
      }
      for (int x = 0; x < 8; x++){
        for (int y = 0; y < 5; y++){
          if(file >> buffer){
           parsedData.query[x][y] = buffer;
          }
        }
      }
    }
    file.close();

  }else{
   cout << "File cannot be opened" << endl;  
  }
  return parsedData;
}
// argv[1] = number of queries, argv[2] = number of training instances
// argv[3] = number of columns in training instance, argv[4] = how many neighbors
//argv[5] = filename of test data 
int main(int argc, char *argv[])
{
     int rank, size;
     MPI_Init( &argc, &argv );
     MPI_Comm_rank( MPI_COMM_WORLD, &rank );
     MPI_Comm_size( MPI_COMM_WORLD, &size );

    // There are more processes than query instances
    if (size > atoi(argv[1])) {
        cout << "The number of processes must be equal or less than the number of queries" << endl;
        return -1;
    }

    // There are more neighbors than training instances
    if (atoi(argv[4]) > atoi(argv[3])) {
         cout << "The number of neighbors must be equal or less than the number of training instances" << endl;
        return -1;
    }
     //Starting clock 
     auto process_start = chrono::high_resolution_clock::now();
     
     File file = parser(argv[5]);
    
     //int training[20][5] = {file.trainingData};
    
     //int queries[20][5] = {file.query};

    int predicted_total[20] = {};

    int num_queries = atoi(argv[1]);

    int start;
    int end;
    
    // Split size for all processes except for last process
    int ceiling = ceil(double(num_queries)/size);

    // 4 queries and 3 processes
    // 0 1 2 3

    // 1 1 2
    // 0 1 23
    
    // int ceiling = floor(double(num_queries)/size);
    // 8 / 3 = 2
    // 8 % 3 = 2 if remain > 1 split with rest of the process otherwise add to end
    // 4 / 3 = 1, 4 % 3 = 1 add to end or first
    // 5 / 3 = 1 5 % 3 = 2 add to rest of processes

    // vector where each index represents the process
    // first for loop add the initial even amount
    // second for loop to add the remainder

    // 4 queries 3 process
    vector<int> processes_distribution;
    int flooring = num_queries/size;
    for (int i = 0; i < size; i++) {
        processes_distribution.push_back(flooring);
    }
    int remainder = atoi(argv[1]) - flooring*size;
    int i = 0;
    while (remainder > 0) {
        processes_distribution[i] += 1;
        i++;
        remainder--;
    }

    vector<int> total_sum;
    for (int i = 0; i < size; ++i) {
        if (i == 0) {
            total_sum.push_back(processes_distribution[i]);
        }
        else {
            total_sum.push_back(total_sum[i-1] + processes_distribution[i]);
        }
    }

    start = total_sum[rank]- processes_distribution[rank];
    end = start + processes_distribution[rank] - 1;

    int predicted_split[20] = {};
    for (int i = start; i <= end; i++) {
        predicted_split[i - start] = KNN(file.trainingData, file.query[i], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), rank);
        cout << "predicted index is " << i-start << " with " <<  predicted_split[i-start] << endl;
    }

    if (rank == 0) {
        
        // start and end
        int start_from_other_process;
        int end_from_other_process;
        int received_split[20] = {};

        // Rank 0's predicted split into predicted_total
        for (int i = start; i <= end; i++) {
            predicted_total[i] = predicted_split[i];
        }
        // All other ranks' predicted split into predicted_total
        for (int r = 1; r < size; ++r) {
            //MPI_Recv(&buffer, count, datatype, source, tag, communicator, &status)
            MPI_Recv(&start_from_other_process, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&end_from_other_process, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int num_queries = end_from_other_process - start_from_other_process + 1;
            MPI_Recv(&received_split, num_queries, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            cout << "To process 0 from rank " << r << " " << num_queries << endl;
            // final result should be in predicted_total
            for (int i = 0; i < num_queries; ++i) {
                cout << received_split[i] << endl;
            }
            // Put the received stuff into predicted_total array
            int i = 0;
            for (int j = start_from_other_process; j <= end_from_other_process; j++) {
                predicted_total[j] = received_split[i];
                // cout << received_split[i] << endl;
                i++;
            }

        }
        
        cout << "totals " << endl;
        for(int i=0; i < atoi(argv[1]); i++) {
            cout<< predicted_total[i] << " ";
        }
        cout << endl;

        
    }
    else {
        //MPI_Send(&buffer, count, datatype, destination, tag, communicator)
        MPI_Send(&start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&predicted_split, end-start+1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
     /*
    False Positive FP/FP+TN
    False Negative FN/FN+TP
    True Positive TP/TP+FN
    True Negative TN/TN+FP
    */
    if (rank == 0) {
        double tp_count = 0;
        double fn_count = 0;
        double tn_count = 0;
        double fp_count = 0;
        int classification = atoi(argv[3]);
        for (int i = 0; i < num_queries; i++) {
            if (file.query[i][classification-1] == false) {
                if (file.query[i][classification-1] == predicted_total[i]) {
                    tn_count += 1;
                } else {
                    fp_count += 1;
                }
            } else if (file.query[i][classification-1] == 1) {
                if (file.query[i][classification-1] == predicted_total[i]) {
                    tp_count +=1; 
                } else {
                    fn_count += 1;
                }
            }
        }
        //Stop process and get duration 
        auto process_stop = chrono::high_resolution_clock::now();
        auto process_duration = chrono::duration_cast<chrono::microseconds>(process_stop - process_start);

        cout << endl;
        cout << fp_count << " False Postives " << (fp_count/(fp_count+tn_count)) * 100 <<"%" << endl;
        cout << fn_count <<" False Negatives " << (fn_count/(fp_count+tn_count)) * 100 <<"%"  << endl;
        cout << tp_count <<" True Postives " << (tp_count/(tp_count+fn_count)) * 100 <<"%"  << endl;
        cout << tn_count <<" True Negatives " << (tn_count/(fp_count+tn_count)) * 100 <<"%"  << endl;
        cout <<" Process took: " << process_duration.count() << " Microseconds. " << endl;
    }
    return 0;
}
