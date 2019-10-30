//
// Created by FETS on 2019/10/20.
//

#include "LogisticRegression.h"


LogisticRegression::~LogisticRegression() = default;


dataMatrix LogisticRegression::cal_derivative(Data inputData, const dataMatrix &W_matrix, int norm, element lambda) {
    const dataMatrix X_matrix = inputData.getX_matrix();
    const dataMatrix Y_matrix = inputData.getY_matrix();
    assert(X_matrix.rows() == Y_matrix.rows());
    int dataNum = X_matrix.rows();
    dataMatrix derivative = dataMatrix::Zero(W_matrix.rows(),W_matrix.cols());
    for (int i = 0; i < dataNum; i++) {
        dataMatrix length = ((Y_matrix.row(i).array() - (X_matrix.row(i) * W_matrix).array().exp() /
                                                        (1 + (X_matrix.row(i) * W_matrix).array().exp())).array());
        derivative += X_matrix.row(i).transpose() * length;
    }
    if (norm == 2) {
        derivative += 2.0 * lambda * W_matrix;
    }
    derivative = -derivative;
    return derivative;
}

element LogisticRegression::cal_Loss(Data inputData, const dataMatrix &W_matrix, int norm, element lambda) {
    const dataMatrix X_matrix = inputData.getX_matrix();
    const dataMatrix Y_matrix = inputData.getY_matrix();
    assert(X_matrix.rows() == Y_matrix.rows());
    dataMatrix loss = dataMatrix::Zero(1,1);
    int dataNum = X_matrix.rows();
    for (int i = 0; i < dataNum; i++) {
        dataMatrix tempLoss = ((Y_matrix.row(i) * X_matrix.row(i) * W_matrix).array() -
                               (1 + (X_matrix.row(i) * W_matrix).array().exp()).log());
        loss = loss + tempLoss;
    }
    if (norm == 2) {
        loss += lambda * W_matrix.transpose() * W_matrix;
    }
    loss = loss / X_matrix.rows();
    loss = -loss;
    return loss(0, 0);
}

