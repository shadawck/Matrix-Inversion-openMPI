//
// Created by shadawck on 3/10/21.
//

#include "MpiUtils.h"
#include <iostream>
#include <openmpi/mpi.h>

using namespace std;

int MpiUtils::initMpi() {
    MPI_Init(nullptr, nullptr);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    cout << "Proc Number : " << world_size << endl;
    return world_size;
}

int MpiUtils::rank() {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    cout << "Rank : " << world_rank << endl;
    return world_rank;
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