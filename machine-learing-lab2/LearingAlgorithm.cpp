//
// Created by FETS on 2019/10/21.
//

#include "LearingAlgorithm.h"

dataMatrix LearningAlgorithm::fitY_matrix(const dataMatrix &X_matrix, const dataMatrix &W_matrix) {
    return X_matrix * W_matrix;
}


dataMatrix LearningAlgorithm::getW_matrixByBatchGradientDescent(Data trainData, int norm, element lambda, int epochs,
                                                               element learningRate, Data *testData) {
    const dataMatrix X_matrix = trainData.getX_matrix();
    const dataMatrix Y_matrix = trainData.getY_matrix();
    const element threshold = 1e-11;
    int order = X_matrix.cols();
    dataMatrix W_matrix = this->initW_matrix(order);
    dataMatrix myY_matrix = this->fitY_matrix(X_matrix, W_matrix);
    element loss = this->cal_Loss(trainData, W_matrix, norm, lambda);
    std::cout << "epoch : " << 0 << " loss : " << loss<<std::endl ;
    if (testData) {
        element testLoss = this->cal_Loss((*testData), W_matrix, norm, lambda);
        std::cout << " test loss : " << testLoss / (*testData).getX_matrix().rows()<<std::endl;
    }
    int epoch = 0;
    while (epoch < epochs) {
        W_matrix = W_matrix - learningRate * this->cal_derivative(trainData, W_matrix, norm, lambda);
        myY_matrix = this->fitY_matrix(X_matrix, W_matrix);
        epoch++;
        element newLoss = this->cal_Loss(trainData, W_matrix, norm, lambda);
        //compare newLoss with oldLoss, if the delta smaller than an small const
        //we think it have been achieved best solution, break the loop.
        if (std::fabs(loss - newLoss) < threshold) {
            break;
        }
        loss = newLoss;
        if (epoch % (epochs / 10) == 0) {
            std::cout << "epoch : " << epoch << " loss : " << loss ;
            if (testData) {
                element testLoss = this->cal_Loss((*testData), W_matrix, norm, lambda);
                std::cout << " test loss : " << testLoss / (*testData).getX_matrix().rows();
            }
            std::cout << std::endl;
        }
    }
    return W_matrix;
}

dataMatrix LearningAlgorithm::initW_matrix(int size) {
    dataMatrix retMatrix = dataMatrix::Random(size, 1);
    return retMatrix;
}
