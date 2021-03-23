#include "Matrix.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <openmpi/mpi.h>
#include <unistd.h>

using namespace std;

void findPivot(size_t row, MatrixConcatCols &augmentedMatrix, size_t &pivot);

void splitAugmentedMatrix(Matrix &mat, MatrixConcatCols &augmentedMatrix);

double *convertValArrayToDouble(valarray<double> array);

void checkSingularity(const MatrixConcatCols &augmentedMatrix, size_t p, int k);

struct {
    double value;
    int index;
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

/** NAIVE IMPLEMENTATION
 * Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
 * @param mat
 */
void invertParallel(Matrix &mat) {
    assert(mat.rows() == mat.cols());
    size_t matrixDimension = mat.rows() * mat.rows();

    MatrixConcatCols augmentedMatrix(mat, MatrixIdentity(mat.rows()));
    size_t colLength = augmentedMatrix.cols();

    int rank = MPI::COMM_WORLD.Get_rank();
    int size = MPI::COMM_WORLD.Get_size();

    for (int k = 0; k < mat.rows(); ++k) {
        size_t locPivot = k;
        double lMax = numeric_limits<double>::lowest();

        for (size_t i = k; i < augmentedMatrix.rows(); i++) {
            double lp = fabs(augmentedMatrix(i, k));
            if ((i % (size) == (unsigned) rank) && (lp > lMax)) {
                lMax = lp;
                locPivot = i;
            }
        }
        send.value = lMax;
        send.index = locPivot;

        MPI_Allreduce(&send, &recv, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        double *rowMaxArray = convertValArrayToDouble(augmentedMatrix.getRowCopy(recv.index));
        double *rowKArray = convertValArrayToDouble(augmentedMatrix.getRowCopy(k));

        MPI_Bcast(rowMaxArray, colLength, MPI_DOUBLE, recv.index % size, MPI_COMM_WORLD);
        MPI_Bcast(rowKArray, colLength, MPI_DOUBLE, k % size, MPI_COMM_WORLD);

        for (size_t i = 0; i < colLength; i++) {
            augmentedMatrix(recv.index, i) = rowMaxArray[i];
            augmentedMatrix(k, i) = rowKArray[i];
        }

        delete[] rowMaxArray;
        delete[] rowKArray;

        checkSingularity(augmentedMatrix, locPivot, k);

        if (recv.index != k) augmentedMatrix.swapRows(recv.index, k);

        double pivot = augmentedMatrix(k, k);
        for (size_t j = 0; j < augmentedMatrix.cols(); ++j) {
            augmentedMatrix(k, j) /= pivot;
        }

        for (size_t i = 0; i < augmentedMatrix.rows(); ++i) {
            if (i != k && i % size == (unsigned) rank) {
                double scale = augmentedMatrix(i, k);
                augmentedMatrix.getRowSlice(i) -= augmentedMatrix.getRowCopy(k) * scale;
            }
        }
    }

    // On copie la partie droite de la matrice AI ainsi transformée dans la matrice courante (this).
    splitAugmentedMatrix(mat, augmentedMatrix);

    auto *recvArray = (double *) malloc(size * mat.rows() * mat.rows() * sizeof(double));
    MPI_Barrier(MPI_COMM_WORLD);

    double *sendBuf = convertValArrayToDouble(mat.getDataArray());

    // gather matrix from each proccess
    MPI_Gather(sendBuf, matrixDimension, MPI_DOUBLE, recvArray, matrixDimension, MPI_DOUBLE, ROOT_PROCESS,
               MPI_COMM_WORLD);

    // rebuild inverse matrix
    if (rank == 0) {
        size_t process, row, column;
        for (size_t i = 0; i < size * augmentedMatrix.rows() * augmentedMatrix.rows(); i++) {
            process = i / (augmentedMatrix.rows() * augmentedMatrix.rows());
            row = (i % (augmentedMatrix.rows() * augmentedMatrix.rows()) / augmentedMatrix.rows());
            column = (i % (mat.rows()));
            if (row % size == process) {
                mat.getDataArray()[(row * augmentedMatrix.rows()) + column] = recvArray[i];
            }
        }
    }

    delete[] sendBuf;
}

void checkSingularity(const MatrixConcatCols &augmentedMatrix, size_t p, int k) {
    if (augmentedMatrix(p, k) == 0) {
        throw runtime_error("Matrix not invertible");
    }
}

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix &iMat1, const Matrix &iMat2) {
    assert(iMat1.cols() == iMat2.rows());
    Matrix lRes(iMat1.rows(), iMat2.cols());
    for (size_t i = 0; i < lRes.rows(); ++i) { /// index
        for (size_t j = 0; j < lRes.cols(); ++j) { /// column
            lRes(i, j) = (iMat1.getRowCopy(i) * iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}

int main(int argc, char **argv) {
    srand((unsigned) time(nullptr));

    unsigned int matrixDimension = 5;
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
    // If Size = 1 : Execute sequential algo
    if (rank == 0) {
        double chronoSeqStart, chronoSeqEnd;
        cout << "SEQUENTIAL EXECUTION" << endl;
        Matrix seqMatrix(randomMatrix);

        chronoSeqStart = MPI_Wtime();
        invertSequential(seqMatrix);
        chronoSeqEnd = MPI_Wtime();

//        cout << "Matrice inverse:\n" << seqMatrix.str() << endl;
//        Matrix lRes = multiplyMatrix(seqMatrix, copyRandomMatrix);
//        cout << "Produit des deux matrices:\n" << lRes.str() << endl;
//
//        cout << "Erreur: " << lRes.getDataArray().sum() - matrixDimension << endl;
        cout << "Total sequential execution time : " << chronoSeqEnd - chronoSeqStart << endl;
    }

    /**
     * parallel execution
     */

    Matrix parMatrix = Matrix(randomMatrix);
    double chronoParStart = MPI_Wtime();
    invertParallel(parMatrix);
    double chronoParEnd = MPI_Wtime();

    if (rank == 0) {
        cout << "ORIGINAL MATRIX" << endl;
        cout << copyRandomMatrix.str() << endl;

        cout << "PARALLEL EXECUTION" << endl;
        cout << "Matrice inverse:\n" << parMatrix.str() << endl;
        Matrix lRes = multiplyMatrix(parMatrix, copyRandomMatrix);
        cout << "Produit des deux matrices:\n" << lRes.str() << endl;

        cout << "Erreur: " << lRes.getDataArray().sum() - matrixDimension << endl;
        cout << "Total parallel execution time : " << chronoParEnd - chronoParStart << endl;
    }

    MPI_Finalize();

    return 0;
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