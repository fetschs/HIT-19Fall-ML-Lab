
//
// Created by FETS on 2019/9/15.
//

#include <utility>
#include "ProductStrategy.h"

#ifndef CLION_DATA_H
#define CLION_DATA_H


class Data {
public:
    Data(dataMatrix xMatrix, dataMatrix yMatrix) : X_matrix(std::move(xMatrix)),
                                                                 Y_matrix(std::move(yMatrix)) {}

    static dataMatrix createX_matrix(int dataNum);

    static dataMatrix createX_matrixForPoly(int dataNum, int order);

    static void addNoise(dataMatrix &dMatrix, element mean, element deviation);

    /**
    * generate Random value with Gauss.
    * @param mean the mean of the distribution.
    * @return double value.
    */
    static element generateRandom(element mean, element deviation);

    dataMatrix getX_matrix();

    dataMatrix getY_matrix();

    /**
     * constructor.
     * @param xMatrix the input Matrix, can create by static function.
     * @param context the context which used to product Y_matrix.
     */
    Data(dataMatrix xMatrix, const ProductContext &context) : X_matrix(std::move(xMatrix)), context(context) {
        this->productY_matrix();
    }


    void setY_Matrix(const dataMatrix &yMatrix);


private:
    dataMatrix X_matrix;
    dataMatrix Y_matrix;
    ProductContext context;

    /**
     * call this product Y_matrix, auto call in constructor.
     */
    void productY_matrix();
};


#endif //CLION_DATA_H
