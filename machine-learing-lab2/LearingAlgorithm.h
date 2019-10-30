//
// Created by FETS on 2019/10/20.
//
#include "Data.h"

#ifndef MACHIEN_LEARING_LAB2_MACHINELEARNINGALGORITHM_H
#define MACHIEN_LEARING_LAB2_MACHINELEARNINGALGORITHM_H


class LearningAlgorithm {
public:
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
    virtual element cal_Loss(Data inputData, const dataMatrix &W_matrix, int norm, element lambda) = 0;

    LearningAlgorithm() = default;

    virtual ~LearningAlgorithm() = default;

    /**
     * cal derivative by solution and input Data.
     * @param inputData input Data/
     * @param W_matrix solution.
     * @param norm regularizer	,0 will not use ,2 will use L2.
     * @param lambda regularizer parameter.
     * @return  derivative value.
     */
    virtual dataMatrix cal_derivative(Data inputData, const dataMatrix &W_matrix, int norm, element lambda) = 0;

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
    dataMatrix
    getW_matrixByBatchGradientDescent(Data trainData, int norm, element lambda, int epochs,
                                      element learningRate, Data *testData);
};


#endif //MACHIEN_LEARING_LAB2_MACHINELEARNINGALGORITHM_H
