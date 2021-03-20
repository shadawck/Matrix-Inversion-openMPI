//
// Created by shadawck on 3/10/21.
//

#include "MpiUtils.hpp"
#include <iostream>
#include <openmpi/mpi.h>

using namespace std;

void MpiUtils::initMpi(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
}

void MpiUtils::totalSize(int size){
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    cout << "Proc Number : " << size << endl;
}

void MpiUtils::rank(int rank) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout << "Rank : " << rank << endl;
}

void MpiUtils::processorName() {
    char processor_n[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_n, &name_len);
    cout << "Processor name: " << processor_n << endl;
}

void MpiUtils::cleanup() {
    MPI_Finalize();
}

