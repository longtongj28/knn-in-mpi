# 474-Project2
Project 2: Implement a distributed algorithm using MPI

Group members:

Johnson Tong jt28@csu.fullerton.edu

Derrick Lee derricklee@csu.fullerton.edu

Num processes need to be equal or less than number of test points

   /*
    knn(dataset, query, k) {
        // MPI_scatter from rank 0 process
        if rank == 0:
            get num_processes from argv
            split_size = dataset.size()/num_processes
            send dataset splits and query to processes
        else:
            Each process should 
                distances = []
                for each datapoint in split 
                    distances.push(distance(query, datapoint))
                send distances to rank 0
        
        r = recombine(dataset_splits) // MPI_gather
        sort(r) //sorted in ascending ordre

        classes = map<class, int>
        for class in r[i] = r[:k]
            classes[class]++;
        sort(classes.begin(), classes.end(), cmp_fn)
        // cmp_fn to compare the counts of the elements in classes

        return classes.begin().element

    }
    */