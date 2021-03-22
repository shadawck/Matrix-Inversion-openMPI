#include "Matrix.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <openmpi/mpi.h>
#include <unistd.h>

void findPivot(size_t row, MatrixConcatCols &augmentedMatrix, size_t &pivot);

void checkSingularity(const MatrixConcatCols &augmentedMatrix, size_t row, size_t pivot);

void splitAugmentedMatrix(Matrix &mat, MatrixConcatCols &augmentedMatrix);

using namespace std;
int ROOT_PROCESS = 0;

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
        checkSingularity(augmentedMatrix, row, pivot);

        // échanger la ligne courante avec celle du pivot
        if (pivot != row) augmentedMatrix.swapRows(pivot, row);

        double pivotValue = augmentedMatrix(row, row);

        for (size_t col = 0; col < augmentedMatrix.cols(); ++col) {
            // On divise les éléments de la rangée row par la valeur du pivot.
            // Ainsi, augmentedMatrix(row,row) deviendra égal à 1.
            augmentedMatrix(row, col) /= pivotValue;
        }

        // Pour chaque rangée...
        for (size_t i = 0; i < augmentedMatrix.rows(); ++i) {
            if (i != row) { // ...différente de row
                double llValue = augmentedMatrix(i, row);
                // On soustrait la rangée row multipliée par l'élément row de la rangée courante
                augmentedMatrix.getRowSlice(i) -= augmentedMatrix.getRowCopy(row) * llValue;
            }
        }
    }
    // On copie la partie droite de la matrice AI ainsi transformée dans la matrice courante (this).
    splitAugmentedMatrix(mat, augmentedMatrix);
}

void splitAugmentedMatrix(Matrix &mat, MatrixConcatCols &augmentedMatrix) {
    for (unsigned int i = 0; i < mat.rows(); ++i) {
        mat.getRowSlice(i) = augmentedMatrix.getDataArray()[slice(i * augmentedMatrix.cols() + mat.cols(), mat.cols(),
                                                                  1)];
    }
}

void checkSingularity(const MatrixConcatCols &augmentedMatrix, size_t row, size_t pivot) {
    if (augmentedMatrix(pivot, row) == 0) throw runtime_error("Matrix not invertible");
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

struct {
    double localPivotValue;
    int localPivotIndex;
} send, recv;

/** NAIVE IMPLEMENTATION
 * Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
 * @param mat
 */
void invertParallel(Matrix &mat) {
    int rank, size;
    size = MPI::COMM_WORLD.Get_size();
    rank = MPI::COMM_WORLD.Get_rank();

    assert(mat.rows() == mat.cols());
    MatrixConcatCols augmentedMatrix(mat, MatrixIdentity(mat.rows()));

    for (size_t k = 0; k < augmentedMatrix.rows(); ++k) {

        double lMax = 0;
        // trouver le l'index du plus grand pivot (local) de la colonne k en valeur absolue pour le process i%size
        int localPivotIndex = k;
        for (size_t i = k; i < augmentedMatrix.rows(); ++i) {
            if ((i % size) == rank) {
                if (fabs(augmentedMatrix(i, k)) > lMax) {
                    lMax = fabs(augmentedMatrix(i, k));
                    localPivotIndex = i;
                }
            }
        }
        send.localPivotIndex = localPivotIndex;
        send.localPivotValue = lMax;

        // Find max in struct on each process and send result to every process
        MPI_Allreduce(&send, &recv, size, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD); //recv.localPivotValue is never used - change with MPI_recv and MPI_send

        int maxPivotIndex = recv.localPivotIndex;

        int root = maxPivotIndex % size;
        MPI_Bcast(&augmentedMatrix(maxPivotIndex, 0), augmentedMatrix.cols(), MPI_DOUBLE, root, MPI_COMM_WORLD);

        checkSingularity(augmentedMatrix, maxPivotIndex, k);

        // on swap la ligne du pivot avec la ligne k
        if (maxPivotIndex != k) augmentedMatrix.swapRows(maxPivotIndex, k);


        // on normalise la ligne k afin que l'element (k,k) soit egale a 1
        double lValue = augmentedMatrix(k, k);
        for (size_t j = 0; j < augmentedMatrix.cols(); ++j) {
            augmentedMatrix(k, j) /= lValue;
        }

        // Pour chaque rangée...
        for (size_t i = 0; i < augmentedMatrix.rows(); ++i) {
            if ((i % size) == rank) {
                if (i != k) { // ...différente de k
                    // On soustrait la rangée k
                    // multipliée par l'élément k de la rangée courante
                    double lValue = augmentedMatrix(i, k);
                    augmentedMatrix.getRowSlice(i) -= augmentedMatrix.getRowCopy(k) * lValue;
                }
            }
        }

        for (int i = 0; i < augmentedMatrix.rows(); ++i) {
            MPI_Bcast(&augmentedMatrix(i, 0), augmentedMatrix.cols(), MPI_DOUBLE, i % size, MPI_COMM_WORLD);
        }

        for (unsigned int i = 0; i < mat.rows(); ++i) {
            mat.getRowSlice(i) = augmentedMatrix.getDataArray()[slice(
                    i * augmentedMatrix.cols() + mat.cols(), mat.cols(), 1)];
        }
    }
}

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix &iMat1, const Matrix &iMat2) {
    assert(iMat1.cols() == iMat2.rows());
    Matrix lRes(iMat1.rows(), iMat2.cols());
    for (size_t i = 0; i < lRes.rows(); ++i) { /// row
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

        cout << "Matrice inverse:\n" << seqMatrix.str() << endl;
        Matrix lRes = multiplyMatrix(seqMatrix, copyRandomMatrix);
        cout << "Produit des deux matrices:\n" << lRes.str() << endl;

        cout << "Erreur: " << lRes.getDataArray().sum() - matrixDimension << endl;
        cout << "Total sequential execution time : " << chronoSeqEnd - chronoSeqStart << endl;
    }

    /**
     * parallel execution
     */

    Matrix parMatrix = Matrix(randomMatrix);
    MPI_Bcast(&parMatrix(0, 0), matrixDimension * matrixDimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
