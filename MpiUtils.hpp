//
// Created by shadawck on 3/10/21.
//

#ifndef TP3_MPIUTILS_H
#define TP3_MPIUTILS_H


class MpiUtils {
public:

    static void initMpi(int argc, char **);

    static void rank(int);

    static void processorName();

    static void cleanup();

    static void totalSize(int size);
};

#endif //TP3_MPIUTILS_H
