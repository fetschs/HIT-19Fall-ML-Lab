//
// Created by FETS on 2019/9/17.
//

#include "LogisticRegression.h"

int main() {
    dataMatrix covariance = dataMatrix::Zero(2, 2);
    covariance << 1, 0.1,
            0.1, 1;

    const int epochs = 1000;
    const element norm = 0;
    const element learningRate = 0.001;
    const element lambda = 0.0005;
    const element dataNum = 100;
    const element testTimes = 5.0;
    const std::string bankNoteFilename = "./data_banknote_authentication.txt";
//
//    Data trainData = Data(Data::createX_matrixForLogistic(covariance, dataNum),
//                          ProductContext(new LinerStrategy));
//    Data *testDataPtr = new Data(Data::createX_matrixForLogistic(covariance, dataNum * testTimes),
//                                 ProductContext(new LinerStrategy));
//    std::ofstream X_matrixOut("X_matrix.txt");
//    X_matrixOut << testDataPtr->getX_matrix();
//    X_matrixOut.close();
//    std::ofstream Y_matrixOut("Y_matrix.txt");
//    Y_matrixOut << testDataPtr->getY_matrix();
//    Y_matrixOut.close();
    Data trainData = Data::getDataFromBankNote(bankNoteFilename);
    Data *testDataPtr = nullptr;
    LogisticRegression logisticRegression = LogisticRegression();
    dataMatrix W_matrixByBGD = logisticRegression.getW_matrixByBatchGradientDescent(trainData, norm, lambda, epochs,
                                                                                    learningRate,testDataPtr);
    dataMatrix X_matrix = trainData.getX_matrix();
    dataMatrix Y_matrix = trainData.getY_matrix();
//    std::ofstream solutionOut("W_matrixByBGD.txt");
//    solutionOut << W_matrixByBGD;
//    solutionOut.close();

    return 0;
}
