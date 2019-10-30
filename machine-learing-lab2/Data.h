
//
// Created by FETS on 2019/9/15.
//

#include <utility>
#include "ProductStrategy.h"
#include "MultivariateNormalRandomVariable.h"

#ifndef CLION_DATA_H
#define CLION_DATA_H


class Data {
public:
    /**
     * move constructor.
     * @param xMatrix X_matrix
     * @param yMatrix Y_matrix
     */
    Data(dataMatrix xMatrix, dataMatrix yMatrix) : X_matrix(std::move(xMatrix)),
                                                   Y_matrix(std::move(yMatrix)) {}

    /**
     * create an X vector, float numbers in it will be evenly distributed in [-1,1]
     * and int or else will be evenly distributed in its values.
     * @param dataNum the number of data will create.
     * @return the X_matrix,
     */
    static dataMatrix createX_matrix(int dataNum);

    /**
     * creat an X_matrix for poly experiment
     * each data will get
     * @param dataNum
     * @param order
     * @return
     */
    static dataMatrix createX_matrixForPoly(int dataNum, int order);

    static dataMatrix createX_matrixForLogistic(dataMatrix covariance, int dataNum);

    static Data getDataFromBankNote(const std::string &filename);

    /**
     *
   const   * @param &dMatrix
     * @param mean
     * @param deviation
     */
    static void addNoise(dataMatrix &dMatrix, element mean, element deviation);

    /**
    * generate Random value with Gauss.
    * @param mean the mean of the distribution.
    * @param deviation the deviation of the distribution.
    * @return double value.
    */
    static element generateRandom(element mean, element deviation);


    dataMatrix getX_matrix();


    dataMatrix getY_matrix();

    /*
     * @param xMatrix the input Matrix, can create by static function.
     * @param context the context which used to product Y_matrix.
     */
    Data(dataMatrix xMatrix, const ProductContext &context) : X_matrix(std::move(xMatrix)), context(context) {
        this->productY_matrix();
    }

    /**
     * set the Y_matrix, which used to update Y_matrix.
     * @param yMatrix Y_matrix will update.
     */
    void setY_Matrix(const dataMatrix &yMatrix);


private:
    dataMatrix X_matrix;// input matrix
    dataMatrix Y_matrix;// label matrix
    ProductContext context;//function context, use to create Data.

    /**
     * call this product Y_matrix, auto call in constructor.
     */
    void productY_matrix();

};


#endif //CLION_DATA_H
