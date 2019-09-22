//
// Created by FETS on 2019/9/17.
//
#include "Data.h"

#ifndef CLION_LEASTSQUARE_H
#define CLION_LEASTSQUARE_H


class LeastSquare {
public:
    /**
     * get solution by math analysis.
     * @param trainData input Data.
     * @param norm regularizer	,0 will not use ,2 will use L2.
     * @param lambda regularizer parameter.
     * @return solution.
     */
    static dataMatrix getW_matrixByAnalysis(Data trainData, int norm, element lambda);
    /**
     * get solution by CG method.
     * @param trainData input Data.
     * @param norm norm regularizer	,0 will not use ,2 will use L2.
     * @param lambda regularizer parameter.
     * @return solution.
     */
    static dataMatrix getW_matrixByConjugateGradient(Data trainData, int norm, element lambda);
    /**
     * use solution calculate Y_matrix.
     * @param X_matrix input X_matrix.
     * @param W_matrix solution.
     * @return Y_matrix calculated by W_matrix and X_matrix.
     */
    static dataMatrix fitY_matrix(const dataMatrix &X_matrix, const dataMatrix &W_matrix);
    /**
     * cal loss use solution and data.
     * @param inputData input Data.
     * @param W_matrix solution.
     * @param norm regularizer	,0 will not use ,2 will use L2.
     * @param lambda regularizer parameter.
     * @return loss value.
     */
    static element cal_Loss(Data inputData, const dataMatrix &W_matrix, int norm, element lambda);
    /**
     * cal derivative by solution and input Data.
     * @param inputData input Data/
     * @param W_matrix solution.
     * @param norm regularizer	,0 will not use ,2 will use L2.
     * @param lambda regularizer parameter.
     * @return  derivative value.
     */
    static dataMatrix cal_derivative(Data inputData, const dataMatrix &W_matrix, int norm, element lambda);
    /**
     * init solution by random.
     * @param size solution's size.
     * @return intial solution.
     */
    static dataMatrix initW_matrix(int size);
    /**
     * get solution by BGD.
     * @param trainData input Data
     * @param norm regularizer	,0 will not use ,2 will use L2.
     * @param lambda regularizer parameter.
     * @param epochs # of epoch.
     * @param learningRate step size.
     * @param testData default will be nullptr , call function with it will get test loss.
     * @return solution.
     */
    static dataMatrix
    getW_matrixByBatchGradientDescent(Data trainData, int norm, element lambda, int epochs,
                                      element learningRate, Data *testData = nullptr);

};


#endif //CLION_LEASTSQUARE_H
