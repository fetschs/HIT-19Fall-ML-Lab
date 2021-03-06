//
// Created by FETS on 2019/9/17.
//

#include "LeastSquare.h"

dataMatrix LeastSquare::getW_matrixByAnalysis(Data trainData, int norm, element lambda) {
    const dataMatrix X_matrix = trainData.getX_matrix();
    const dataMatrix Y_matrix = trainData.getY_matrix();
    if (norm == 0) {
        return (X_matrix.transpose() * X_matrix).inverse() *
               X_matrix.transpose() * Y_matrix;
    }
    if (norm == 2) {
        return (X_matrix.transpose() * X_matrix +
                element(2.0) * lambda * dataMatrix::Identity(X_matrix.cols(), X_matrix.cols())).inverse() *
               (X_matrix.transpose() * Y_matrix);
    }
}

dataMatrix LeastSquare::fitY_matrix(const dataMatrix &X_matrix, const dataMatrix &W_matrix) {
    return X_matrix * W_matrix;
}

dataMatrix LeastSquare::cal_derivative(Data inputData, const dataMatrix &W_matrix, int norm, element lambda) {
    const dataMatrix X_matrix = inputData.getX_matrix();
    const dataMatrix Y_matrix = inputData.getY_matrix();
    dataMatrix ans_matrix = X_matrix.transpose() * X_matrix * W_matrix - X_matrix.transpose() * Y_matrix;
    if (norm == 2) {
        ans_matrix = ans_matrix + element(2.0) * lambda * W_matrix;
    }
    return ans_matrix;
}

element LeastSquare::cal_Loss(Data inputData, const dataMatrix &W_matrix, int norm, element lambda) {
    const dataMatrix X_matrix = inputData.getX_matrix();
    const dataMatrix Y_matrix = inputData.getY_matrix();
    dataMatrix ans_matrix = 0.5 * (X_matrix * W_matrix - Y_matrix).transpose() * (X_matrix * W_matrix - Y_matrix);
    if (norm == 2) {
        ans_matrix += lambda * W_matrix.transpose() * W_matrix;
    }
    return ans_matrix(0, 0);
}

dataMatrix LeastSquare::getW_matrixByBatchGradientDescent(Data trainData, int norm, element lambda, int epochs,
                                                          element learningRate, Data *testData) {
    const dataMatrix X_matrix = trainData.getX_matrix();
    const dataMatrix Y_matrix = trainData.getY_matrix();
    const element threshold = 1e-11;
    int order = X_matrix.cols();
    dataMatrix W_matrix = LeastSquare::initW_matrix(order);
    dataMatrix myY_matrix = LeastSquare::fitY_matrix(X_matrix, W_matrix);
    element loss = LeastSquare::cal_Loss(trainData, W_matrix, norm, lambda);
    int epoch = 0;
    while (epoch < epochs) {
        W_matrix = W_matrix - learningRate * LeastSquare::cal_derivative(trainData, W_matrix, norm, lambda);
        myY_matrix = LeastSquare::fitY_matrix(X_matrix, W_matrix);
        epoch++;
        element newLoss = LeastSquare::cal_Loss(trainData, W_matrix, norm, lambda);
        //compare newLoss with oldLoss, if the delta smaller than an small const
        //we think it have been achieved best solution, break the loop.
        if(std::fabs(loss-newLoss)<threshold){
            break;
        }
        loss = newLoss;
        if (epoch % (epochs / 10) == 0) {
            std::cout << "epoch : " << epoch << " loss : " << loss / X_matrix.rows();
            if (testData) {
                element testLoss = LeastSquare::cal_Loss((*testData), W_matrix, norm, lambda);
                std::cout << " test loss : " << testLoss / (*testData).getX_matrix().rows();
            }
            std::cout << std::endl;
        }
    }
    return W_matrix;
}

dataMatrix LeastSquare::initW_matrix(int size) {
    dataMatrix retMatrix = dataMatrix::Random(size, 1);
    return retMatrix;
}

dataMatrix LeastSquare::getW_matrixByConjugateGradient(Data trainData, int norm, element lambda) {
    const dataMatrix X_matrix = trainData.getX_matrix();
    const dataMatrix Y_matrix = trainData.getY_matrix();
    dataMatrix A = X_matrix.transpose() * X_matrix;
    if (norm == 2) {
        A = lambda * dataMatrix::Identity(A.rows(), A.cols()) + A;
    }
    dataMatrix b = X_matrix.transpose() * Y_matrix;
    int order = X_matrix.cols();
    dataMatrix W_matrix = LeastSquare::initW_matrix(order);
    dataMatrix d = b - A * W_matrix;
    dataMatrix r = d;
    //iterations only order times.
    //but beacuse of float precision, more time will improve its loss.
    for (int iteration = 1; iteration <= order; iteration++) {
        dataMatrix r_square = r.transpose() * r;
        dataMatrix alpha = (r_square).array() / (d.transpose() * A * d).array();//  it's a scalar
        W_matrix = W_matrix + d * alpha;
        r = r - A * d * alpha;
        dataMatrix beta = (r.transpose() * r).array() / (r_square).array();// it's a scalar
        d = r + d * beta;
    }
    return W_matrix;
}

