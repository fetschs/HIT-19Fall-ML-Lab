//
// Created by FETS on 2019/9/17.
//
#include "Data.h"

#ifndef CLION_LEASTSQUARE_H
#define CLION_LEASTSQUARE_H


class LeastSquare {
public:
    static dataMatrix getW_matrixByAnalysis(Data trainData, int norm, element lambda);

    static dataMatrix getW_matrixByConjugateGradient(Data trainData, int norm, element lambda);

    static dataMatrix fitY_matrix(const dataMatrix &X_matrix, const dataMatrix &W_matrix);

    static element cal_Loss(Data trainData, const dataMatrix &W_matrix, int norm, element lambda);

    static dataMatrix cal_derivative(Data trainData, const dataMatrix &W_matrix, int norm, element lambda);

    static dataMatrix initW_matrix(int size);

    static dataMatrix
    getW_matrixByBatchGradientDescent(Data trainData, int norm, element lambda, int epochs,
                                      element learning_rate, Data *testData = nullptr);

};


#endif //CLION_LEASTSQUARE_H
