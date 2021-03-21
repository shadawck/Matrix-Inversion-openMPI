#include "Matrix.hpp"

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

valarray<double> convertDoubleToValArray(double *array, int arrayLength);

/** NAIVE IMPLEMENTATION
 * Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
 * @param mat
 */
void invertParallel(Matrix &mat) { // Number of row staw matrixDimension but for mat col number change
    int rank, size;
    assert(mat.rows() == mat.cols());
    MatrixConcatCols *augmentedMat;

    int dim = mat.rows();
    int rowSize = mat.rows();
    int colSize = mat.rows();

    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size > dim) {
        cout << "Number of proc need to be greater than dim of matrix" << endl;
        MPI_Finalize();
        exit(0);
    }

    MatrixIdentity eye(dim);

    double *matrixArray;
    double *eyeArray;
    if (rank == 0) {
        augmentedMat = new MatrixConcatCols(mat, MatrixIdentity(mat.rows()));
        cout << mat.str() << endl;
        matrixArray = convertValArrayToDouble(mat.getDataArray());
        eyeArray = convertValArrayToDouble(eye.getDataArray());
    }

    int numRows = rowSize / size;

    double subArray[numRows * colSize];
    double subEyeArray[numRows * colSize];
    /// START SCATTER
    if (size == 1) {
        memcpy(subArray, matrixArray, rowSize * colSize * sizeof(double));
        memcpy(subEyeArray, eyeArray, rowSize * colSize * sizeof(double));
    } else {
        for (int row = 0; row < numRows; row++) {
//            if (rank == 0) cout << "Iteration " << row << " | Scatter start at : " << matrixArray[row * colSize * size] << endl;
            MPI_Scatter(&matrixArray[row * colSize * size], colSize, MPI_DOUBLE,
                        &subArray[row * colSize], colSize, MPI_DOUBLE,
                        ROOT_PROCESS,
                        MPI_COMM_WORLD);
        }
    }
    /// DEBUG SCATTER
    int subArrayLength = numRows * colSize;
    Matrix::printArray(subArray, subArrayLength, rank, "SUB_MATRIX");

    /// All processor pivoting
    double pivot, scale;
    int local_row, which_rank;
    double row[colSize];
    double eyeRow[colSize];

    for (int i = 0; i < rowSize; i++) {
        // Which row in the sub-matrix are we accessing?
        local_row = i / size;
        // Which rank does this row belong to?
        which_rank = i % size;

        // Eliminate if the pivot belongs to this rank
        if (rank == which_rank) {
            pivot = subArray[local_row * rowSize + i];

            // Divide the rest of the row by the pivot
            for (int j = i + 1; j < rowSize; j++) {
                subArray[local_row * rowSize + j] /= pivot;
                subEyeArray[local_row * rowSize + j] /= pivot;
            }

            // Use assignment for the trivial self-division
            subArray[local_row * rowSize + i] = 1;
            subEyeArray[local_row * rowSize + i] = 1;

            // Copy the row into our send buffer
            memcpy(row, &subArray[local_row * rowSize], rowSize * sizeof(double));
            memcpy(eyeRow, &subEyeArray[local_row * rowSize], rowSize * sizeof(double));

            // Broadcast this row to all the ranks
            MPI_Bcast(row, rowSize, MPI_DOUBLE, which_rank, MPI_COMM_WORLD);
            MPI_Bcast(eyeRow, rowSize, MPI_DOUBLE, which_rank, MPI_COMM_WORLD);

            // Eliminate for the other rows mapped to this rank
            for (int j = local_row + 1; j < numRows; j++) {
                scale = subArray[j * rowSize + i];

                // Subtract to eliminate pivot from later rows
                for (int k = i + 1; k < rowSize; k++) {
                    subArray[j * rowSize + k] -= scale * row[k];
                    subEyeArray[j * rowSize + k] -= scale * eyeRow[k];
                }

                // Use assignment for the trivial elimination
                subArray[j * rowSize + i] = 0;
                subEyeArray[j * rowSize + i] = 0;
            }
        } else {
            // Receive a row to use for elimination
            MPI_Bcast(row, rowSize, MPI_DOUBLE, which_rank, MPI_COMM_WORLD);
            MPI_Bcast(eyeRow, rowSize, MPI_DOUBLE, which_rank, MPI_COMM_WORLD);

            // Eliminate for all the rows mapped to this rank
            for (int j = local_row; j < numRows; j++) {
                if ((which_rank < rank) || (j > local_row)) {
                    scale = subArray[j * rowSize + i];

                    //Subtract to eliminate pivot from later rows
                    for (int k = i + 1; k < rowSize; k++) {
                        subArray[j * rowSize + k] -= scale * row[k];
                        subEyeArray[j * rowSize + k] -= scale * eyeRow[k];
                    }

                    // Use assignment for the trivial elimination
                    subArray[j * rowSize + i] = 0;
                    subEyeArray[j * rowSize + i] = 0;
                }
            }
        }
    }

    // Barrier to track when calculations are done
    MPI_Barrier(MPI_COMM_WORLD);

    /// START GATHER
    if (size == 1) {
        memcpy(matrixArray, subArray, rowSize * colSize * sizeof(double));
        memcpy(eyeArray, subEyeArray, rowSize * colSize * sizeof(double));
    } else {
        for (int r = 0; r < numRows; r++) {
            MPI_Gather(&subArray[r * colSize], colSize, MPI_DOUBLE,
                       &matrixArray[r * size * colSize], colSize, MPI_DOUBLE,
                       ROOT_PROCESS,
                       MPI_COMM_WORLD);
        }
    }

    /// DEBUG GATHER
    int matrixArrayLength = mat.rows() * mat.cols();
    if (rank == 0) Matrix::printArray(matrixArray, matrixArrayLength, rank, "MATRIX");

    MPI_Finalize();

    if (rank == 0) Matrix::printArrayAsMatrix(matrixArray, rowSize, colSize, "MATRIX");
    if (rank == 0) Matrix::printArrayAsMatrix(eyeArray, rowSize, colSize, "INVERSED RESULT");

}

// This is how you can get a dynamic array from a valarray.
double *convertValArrayToDouble(valarray<double> array) {
    auto *newArray = new double[array.size()];
    copy(begin(array), end(array), newArray);
    return newArray;
}

valarray<double> convertDoubleToValArray(double *array, int arrayLength) {
    return valarray<double>(array, arrayLength);
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

    int matrixDimension = 5;
    if (argc == 2) {
        matrixDimension = atoi(argv[1]);
    }

    MatrixRandom lA(matrixDimension, matrixDimension);
//    MatrixExample matrixExample(matrixDimension, 3);

//    cout << "Matrice random:\n" << lA.str() << endl;

    Matrix lB(lA);
//    invertSequential(lB);
//    cout << "Matrice inverse:\n" << lB.str() << endl;
//
//    Matrix lRes = multiplyMatrix(lA, lB);
//    cout << "Produit des deux matrices:\n" << lRes.str() << endl;
//
//    cout << "Erreur: " << lRes.getDataArray().sum() - matrixDimension << endl;


    invertParallel(lB);

    return 0;
}
