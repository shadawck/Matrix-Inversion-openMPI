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
 * @param augmentedMatrix
 */
void invertParallel(
        Matrix &augmentedMatrix) { // Number of row staw matrixDimension but for augmentedMatrix col number change
    int rank, size;
    assert(augmentedMatrix.rows() == augmentedMatrix.cols());
    int rowSize = augmentedMatrix.rows();
    int colSize = augmentedMatrix.rows();

    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

//    MatrixConcatCols *augmentedMatrix;

    double *originalArray;
    if (rank == 0) {
//        augmentedMatrix = new MatrixConcatCols(augmentedMatrix, MatrixIdentity(augmentedMatrix.rows()));
        cout << "PRINT AUGMENTED MATRIX" << endl;
        cout << augmentedMatrix.str() << endl;
        originalArray = convertValArrayToDouble(augmentedMatrix.getDataArray());
    }

    int numRows = rowSize / size;

//    cout << "Array for rank : " << rank << endl;
//    int subArrayLength = sizeof(subArray) / sizeof(subArray[0]);
//    Matrix::printArray(subArray, subArrayLength);

    double subArray[numRows * colSize];
    if (size == 1) {
        memcpy(subArray, originalArray, rowSize * colSize * sizeof(double));
    } else {
        for (int row = 0; row < numRows; row++) {
            MPI_Scatter(&originalArray[row * colSize * size], colSize, MPI_DOUBLE,
                        &subArray[row * colSize], colSize, MPI_DOUBLE, ROOT_PROCESS,
                        MPI_COMM_WORLD);
        }
    }
    /** DEBUG */
    //cout << "Array for rank : " << rank << endl;
    int subArrayLength = sizeof(subArray) / sizeof(subArray[0]);
    //Matrix::printArray(subArray, subArrayLength);

    double rowToSend[colSize];

    // Get start time
    double chronoStart;
    if (rank == 0) {
        chronoStart = MPI_Wtime();
    }

    int localRow, whichRank;
    double pivot, scale;
    for (int row = 0; row < rowSize; row++) {
        Matrix::printArray(subArray, subArrayLength);
        // row of the subMatrix
        localRow = row / size;
        // rank of this localRow
        whichRank = row % size;

        // if pivot in rank of localRow then eliminate
        if (rank == whichRank) {
            pivot = subArray[localRow * colSize + row];

            // iterate through each element of row (without the pivot)
            for (int j = row + 1; j < colSize; j++) {
                double tmp = subArray[localRow * colSize + j];
                subArray[localRow * colSize + j] /= pivot;
                cout << "Before : " << tmp << "| After : " << subArray[localRow * colSize + j] << "| Pivot was :" << pivot << endl;
            }

            // directly assigne 1 to divided pivot
            subArray[localRow * colSize + row] = 1;
//            cout << endl << "Divided Row of rank : " << rank << endl;
//            Matrix::printArray(subArray, subArrayLength);

            // Copy the row into our send buffer
            memcpy(rowToSend, &subArray[localRow * colSize], colSize * sizeof(double));

            //cout << "Sended Row" << endl;
            //Matrix::printArray(rowToSend, subArrayLength);

            // Broadcast this row to all the ranks
            MPI_Bcast(rowToSend, colSize, MPI_DOUBLE, whichRank, MPI_COMM_WORLD);

            // Eliminate for the other rows mapped to this rank
            for (int j = localRow + 1; j < numRows; j++) {
                scale = subArray[j * colSize + row];

                // Subtract to eliminate pivot from later rows
                for (int k = row + 1; k < rowSize; k++) {
                    double tmp = subArray[j * colSize + k];
                    subArray[j * colSize + k] -= scale * rowToSend[k];
                    cout << "Before : " << tmp << "| After : " << subArray[j * colSize + k] << "| Scale was :" << scale << "| rTS value was : " << rowToSend[k] << endl;
                }

                subArray[j * colSize + row] = 0;

//                cout << endl << "UPDATED SUBARRAY of rank " << rank << endl;
//                Matrix::printArray(subArray, subArrayLength);
//                cout << endl;

            }
        } /*else {
            // Receive a row to use for elimination
            MPI_Bcast(rowToSend, colSize, MPI_DOUBLE, whichRank, MPI_COMM_WORLD);

            for (int j = localRow; j < numRows; j++) {
                if (whichRank < rank || j > localRow) {
                    scale = subArray[j * colSize + row];
                    //cout << "scale" << scale << endl;

                    //Subtract to eliminate pivot from later rows
                    for (int k = row + 1; k < rowSize; k++) {
                        subArray[j * colSize + k] -= scale * rowToSend[k];
                    }

                    // Use assignment for the trivial elimination
                    subArray[j * colSize + row] = 0;
                }
            }
        }*/
    }

    // Stop the time before the gather phase
    double chronoEnd;
    double total;
    if (rank == 0) {
        chronoEnd = MPI_Wtime();
        total = chronoEnd - chronoStart;
    }

    /*
 * Collect all Sub-Matrices
 * All sub-matrices are gathered using the gather function
 */


    if (size == 1) {
        memcpy(originalArray, subArray, rowSize * colSize * sizeof(double));
    } else {
        // Gather "size" rows at a time
        for (int row = 0; row < numRows; row++) {
            MPI_Gather(&subArray[row * colSize], colSize, MPI_DOUBLE,
                       &originalArray[row * size * colSize], colSize, MPI_DOUBLE, ROOT_PROCESS,
                       MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();

    if (rank == 0) {
        auto val = convertDoubleToValArray(originalArray, sizeof(originalArray) / sizeof(originalArray[0]));
        //cout << "FINAL ARRAY" << endl;
        //Matrix::printValArray(val);

        //cout << total << " Seconds" << endl;
    }


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

    cout << "EXAMPLE" << endl;
    MatrixExample matrixExample(matrixDimension, matrixDimension);

//    cout << "Matrice random:\n" << lA.str() << endl;

    Matrix lB(lA);
//    invertSequential(lB);
//    cout << "Matrice inverse:\n" << lB.str() << endl;
//
//    Matrix lRes = multiplyMatrix(lA, lB);
//    cout << "Produit des deux matrices:\n" << lRes.str() << endl;
//
//    cout << "Erreur: " << lRes.getDataArray().sum() - matrixDimension << endl;

    invertParallel(lA);

    return 0;
}
