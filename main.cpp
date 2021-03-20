#include "Matrix.hpp"
#include "MpiUtils.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <openmpi/mpi.h>
#include <unistd.h>

void findPivot(size_t row, MatrixConcatCols &augmentedMatrix, size_t &pivot);

void checkSingularity(const MatrixConcatCols &augmentedMatrix, size_t row, size_t pivot);


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
    for (unsigned int i = 0; i < mat.rows(); ++i) {
        mat.getRowSlice(i) = augmentedMatrix.getDataArray()[slice(i * augmentedMatrix.cols() + mat.cols(), mat.cols(),
                                                                  1)];
    }
}

void checkSingularity(const MatrixConcatCols &augmentedMatrix, size_t row, size_t pivot) {
    if (augmentedMatrix(pivot, row) == 0) throw runtime_error("Matrix not invertible");
}

double *convertValArrayToDouble(valarray<double> array);

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

/** NAIVE IMPLEMENTATION
 * Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
 * @param mat
 */
void invertParallel(Matrix &mat) { // Number of row staw matrixDimension but for augmentedMatrix col number change
    int rank, nbProcess;

    assert(mat.rows() == mat.cols());
    MatrixConcatCols augmentedMatrix(mat, MatrixIdentity(mat.rows()));

    int rowSize = augmentedMatrix.rows();
    int colSize = augmentedMatrix.cols();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbProcess);

    int rowByProcess = rowSize / nbProcess;
//    cout << "Row By process: " << rowByProcess << endl;

    if (rank == 0) {
        cout << "PRINT AUGMENTED MATRIX" << endl;
        cout << augmentedMatrix.str() << endl;
    }

    double *sendarray = convertValArrayToDouble(augmentedMatrix.getDataArray());
    double recvarray[rowByProcess * colSize];

    MPI_Scatter(sendarray, rowByProcess * colSize, MPI_DOUBLE,
                recvarray, rowByProcess * colSize, MPI_DOUBLE, ROOT_PROCESS,
                MPI_COMM_WORLD);

    cout << "Array for rank : " << rank << endl;
    int size = sizeof(recvarray)/sizeof(recvarray[0]);
    Matrix::printArray(recvarray, size);

}

double *convertValArrayToDouble(valarray<double> array) {// This is how you can get a dynamic array from a valarray.
    auto *newArray = new double[array.size()];
    copy(begin(array), end(array), newArray);
    return newArray;
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

    MatrixRandom lA(matrixDimension, matrixDimension);
//    cout << "Matrice random:\n" << lA.str() << endl;

    Matrix lB(lA);
//    invertSequential(lB);
//    cout << "Matrice inverse:\n" << lB.str() << endl;
//
//    Matrix lRes = multiplyMatrix(lA, lB);
//    cout << "Produit des deux matrices:\n" << lRes.str() << endl;
//
//    cout << "Erreur: " << lRes.getDataArray().sum() - matrixDimension << endl;


    cout << "Start MPI implementation" << endl;

    MPI_Init(nullptr, nullptr);
    invertParallel(lB);
    MPI_Finalize();

    return 0;
}
