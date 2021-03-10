//
// Created by shadawck on 3/10/21.
//

#ifndef TP3_MPIUTILS_H
#define TP3_MPIUTILS_H


class MpiUtils {
public:

    static int initMpi();

    static int rank();

    static void processorName();

    static void cleanup();
};


#endif //TP3_MPIUTILS_H
