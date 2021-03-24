#include "Matrix.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <openmpi/mpi.h>
#include <unistd.h>
#include <sstream>

using namespace std;

void findPivot(size_t row, MatrixConcatCols &augmentedMatrix, size_t &pivot);

void splitAugmentedMatrix(Matrix &mat, MatrixConcatCols &augmentedMatrix);

double *convertValArrayToDouble(valarray<double> array);

void checkSingularity(const MatrixConcatCols &augmentedMatrix, size_t p, size_t k);

void rebuildMatrix(Matrix &mat, size_t matrixDimension, const MatrixConcatCols &augmentedMatrix, int size,
                   const double *recvArray);
Matrix multiplyMatrix(const Matrix &iMat1, const Matrix &iMat2);

void printResult(int matrixDimension, double totalPar, Matrix &lRes);

struct {
    double value;
    size_t index;
} send, recv;


const unsigned int ROOT_PROCESS = 0;

/**
 * Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
 * @param mat
 */
void invertSequential(Matrix &mat) {
    assert(mat.rows() == mat.cols());
    MatrixConcatCols augmentedMatrix(mat, MatrixIdentity(mat.rows()));

    // traiter chaque rangée
    for (size_t row = 0; row < mat.rows(); ++row) {

        size_t pivot;
        findPivot(row, augmentedMatrix, pivot);
        checkSingularity(augmentedMatrix, pivot, row);

        // échanger la index courante avec celle du pivot
        if (pivot != row) augmentedMatrix.swapRows(pivot, row);

        double pivotValue = augmentedMatrix(row, row);

        for (size_t col = 0; col < augmentedMatrix.cols(); ++col) {
            // On divise les éléments de la rangée index par la valeur du pivot.
            // Ainsi, augmentedMatrix(index,index) deviendra égal à 1.
            augmentedMatrix(row, col) /= pivotValue;
        }

        for (size_t i = 0; i < augmentedMatrix.rows(); ++i) {         // Pour chaque rangée...
            if (i != row) { // ...différente de index
                double llValue = augmentedMatrix(i, row);
                // On soustrait la rangée index multipliée par l'élément index de la rangée courante
                augmentedMatrix.getRowSlice(i) -= augmentedMatrix.getRowCopy(row) * llValue;
            }
        }
    }
    // On copie la partie droite de la matrice AI ainsi transformée dans la matrice courante (this).
    splitAugmentedMatrix(mat, augmentedMatrix);
}

std::string convertirArrayEnString(double *tablo, int size) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < size; i++)
        oss << tablo[i] << ", ";
    oss << "]";
    return oss.str();
}


/**
 * Invert matrix with Gauss Jordan method
 * @param mat Original matrix
 */
void invertParallel(Matrix &mat) {
    assert(mat.rows() == mat.cols());
    size_t matrixDimension = mat.rows() * mat.rows();

    MatrixConcatCols augmentedMatrix(mat, MatrixIdentity(mat.rows()));
    size_t rowLength = augmentedMatrix.rows();
    size_t colLength = augmentedMatrix.cols();

    int rank = MPI::COMM_WORLD.Get_rank();
    int size = MPI::COMM_WORLD.Get_size();

    for (size_t k = 0; k < mat.rows(); ++k) {
        size_t locPivot = k;
        double lMax = numeric_limits<double>::lowest();
        double lp;
        for (size_t i = k; i < rowLength; i++) {
            lp = fabs(augmentedMatrix(i, k));
            if ((i % (size) == rank) && (lp > lMax)) {
                lMax = lp;
                locPivot = i;
            }
        }

        send.value = lMax;
        send.index = locPivot;

        MPI_Allreduce(&send, &recv, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        double *rowMaxArray = convertValArrayToDouble(augmentedMatrix.getRowCopy(recv.index));
        double *rowKArray = convertValArrayToDouble(augmentedMatrix.getRowCopy(k));
        MPI_Bcast(rowMaxArray, colLength, MPI_DOUBLE, (int) recv.index % size, MPI_COMM_WORLD);
        MPI_Bcast(rowKArray, colLength, MPI_DOUBLE, (int) k % size, MPI_COMM_WORLD);

        for (size_t i = 0; i < colLength; i++) {
            augmentedMatrix(recv.index, i) = rowMaxArray[i]; //
            augmentedMatrix(k, i) = rowKArray[i];
        }

        delete[] rowMaxArray;
        delete[] rowKArray;

        checkSingularity(augmentedMatrix, locPivot, k);

        if (recv.index != k) {
            augmentedMatrix.swapRows(recv.index, k);
        }

        double pivot = augmentedMatrix(k, k);
        for (size_t col = 0; col < colLength; ++col) {
            augmentedMatrix(k, col) /= pivot;
        }

        for (size_t r = 0; r < rowLength; ++r) {
            if (r != k && r % size == rank) {
                double scale = augmentedMatrix(r, k);
                augmentedMatrix.getRowSlice(r) -= augmentedMatrix.getRowCopy(k) * scale;
            }
        }
    }

    splitAugmentedMatrix(mat, augmentedMatrix);

    double* processRow;
    for (int i = 0; i < mat.rows(); i++) {
        /* Pointless to send data to root if you're already root :) */
        if (rank != 0 && rank == i % size) {
            processRow = convertValArrayToDouble(mat.getRowCopy((mat.cols() * i + i) % mat.cols()));
            MPI_Send(processRow, rowLength, MPI_DOUBLE, ROOT_PROCESS, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "rank" << rank << endl;
    MPI_Status mpiStatus;
    if (rank == 0) {
        for (int i = 0; i < mat.rows(); i++) {
            /* Pointless to try to receive data if your mat is already updated */
            if(i%size != 0) {
                MPI_Recv(&mat.getDataArray()[i * mat.rows()], rowLength, MPI_DOUBLE, i % size, MPI_ANY_TAG, MPI_COMM_WORLD, &mpiStatus);
            }
        }
    }
//
//    MPI_Barrier(MPI_COMM_WORLD);
//
//
//    if(rank == 0)
//        cout <<  mat.str() << endl;
//
//    auto *recvArray = (double *) malloc(size * mat.rows() * mat.rows() * sizeof(double));
//    double *sendArray = convertValArrayToDouble(mat.getDataArray());
//
//    // gather matrix from each process
//    MPI_Gather(sendArray, matrixDimension, MPI_DOUBLE,
//               recvArray, matrixDimension, MPI_DOUBLE,
//               ROOT_PROCESS,
//               MPI_COMM_WORLD);
//
//
//    // rebuild inverse matrix
//    if (rank == 0) {
//        rebuildMatrix(mat, matrixDimension, augmentedMatrix, size, recvArray);
//        cout << "Matrix" << endl;
//        cout << mat.str() << endl;
//    }
//
//    delete[] sendArray;
//    delete[] recvArray;
}


int main(int argc, char **argv) {
    srand((unsigned) time(nullptr));

    int matrixDimension = 5;
    if (argc == 2) {
        matrixDimension = atoi(argv[1]);
    }

    int rank, size;
    MatrixRandom randomMatrix(matrixDimension, matrixDimension);
    const Matrix &copyRandomMatrix(randomMatrix);

    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /**
    * sequential execution
    */
    if (rank == 0) {
        cout << "--- SEQUENTIAL EXECUTION ---" << endl;
        Matrix seqMatrix(randomMatrix);

        double cronSeqStart = MPI_Wtime();
        invertSequential(seqMatrix);
        double cronSeqEnd = MPI_Wtime();
        double totalSeq = cronSeqEnd - cronSeqStart;

        Matrix lResSeq = multiplyMatrix(seqMatrix, copyRandomMatrix);
        printResult(matrixDimension, totalSeq, lResSeq);
    }

    /**
     * parallel execution
     */
    Matrix parMatrix = Matrix(randomMatrix);
    double cronParStart = MPI_Wtime();
    invertParallel(parMatrix);
    double cronParEnd = MPI_Wtime();
    double totalPar = cronParEnd - cronParStart;

    if (rank == 0) {
        cout << endl << " --- PARALLEL EXECUTION --- " << endl;
        Matrix lResPar = multiplyMatrix(parMatrix, copyRandomMatrix);
        printResult(matrixDimension, totalPar, lResPar);
    }

    MPI_Finalize();
    return 0;
}

void printResult(int matrixDimension, double totalPar, Matrix &lRes) {
    cout << "Matrix dimension : " << matrixDimension << endl;
    cout << "Erreur: " << lRes.getDataArray().sum() - matrixDimension << endl;
    cout << "Total execution time : " << totalPar << endl;
}

void splitAugmentedMatrix(Matrix &mat, MatrixConcatCols &augmentedMatrix) {
    for (unsigned int i = 0; i < mat.rows(); ++i) {
        mat.getRowSlice(i) = augmentedMatrix.getDataArray()[slice(i * augmentedMatrix.cols() + mat.cols(), mat.cols(),
                                                                  1)];
    }
}

void findPivot(size_t row, MatrixConcatCols &augmentedMatrix, size_t &pivot) {
    pivot = row;
    double lMax = fabs(augmentedMatrix(row, row));
    for (size_t i = row; i < augmentedMatrix.rows(); ++i) {
        if (fabs(augmentedMatrix(i, row)) > lMax) {
            lMax = fabs(augmentedMatrix(i, row));
            pivot = i;
        }
    }
}

double *convertValArrayToDouble(valarray<double> array) {
    auto *newArray = new double[array.size()];
    copy(begin(array), end(array), newArray);
    return newArray;
}

void checkSingularity(const MatrixConcatCols &augmentedMatrix, size_t p, size_t k) {
    if (augmentedMatrix(p, k) == 0) {
        throw runtime_error("Matrix not invertible");
    }
}

void rebuildMatrix(Matrix &mat, size_t matrixDimension, const MatrixConcatCols &augmentedMatrix, int size,
                   const double *recvArray) {
    size_t process, row, column;
    for (size_t i = 0; i < size * matrixDimension; i++) {
        process = i / (matrixDimension);
        row = (i % (matrixDimension) / augmentedMatrix.rows());
        column = (i % (mat.rows()));
        if (row % size == process) {
            mat.getDataArray()[(row * augmentedMatrix.rows()) + column] = recvArray[i];
        }
    }
}

Matrix multiplyMatrix(const Matrix &iMat1, const Matrix &iMat2) {
    assert(iMat1.cols() == iMat2.rows());
    Matrix lRes(iMat1.rows(), iMat2.cols());
    for (size_t i = 0; i < lRes.rows(); ++i) {
        for (size_t j = 0; j < lRes.cols(); ++j) {
            lRes(i, j) = (iMat1.getRowCopy(i) * iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}