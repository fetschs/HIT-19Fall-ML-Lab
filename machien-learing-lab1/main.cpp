//
// Created by FETS on 2019/9/17.
//

#include "LeastSquare.h"

int main() {
    const int order = 6;
    const int dataNum = 20;
    const int testDataNum = 1000;
    const int norm = 2;
    const int epochs = 1000000;
    const element mean = 0.1;
    const element deviation = 0.1;
    const element learningRate = 0.000001;
    const element lambda = 0.0005;
    //const hyperparameters.

    Data trainData = Data(Data::createX_matrixForPoly(dataNum, order), ProductContext(new SinStrategy));
    dataMatrix Y_matrix = trainData.getY_matrix().col(1);
    Data::addNoise(Y_matrix, mean, deviation);
    trainData.setY_Matrix(Y_matrix);
    // generate train dataset.

    dataMatrix W_matrixByAnalysis = LeastSquare::getW_matrixByAnalysis(trainData, norm, lambda);
    std::cout << "Analysis loss: " << LeastSquare::cal_Loss(trainData, W_matrixByAnalysis, norm, lambda) / dataNum
              << std::endl;

    dataMatrix W_matrixByConjugateGradient = LeastSquare::getW_matrixByConjugateGradient(trainData, norm, lambda);
    std::cout << "ConjugateGradient loss: "
              << LeastSquare::cal_Loss(trainData, W_matrixByConjugateGradient, norm, lambda) / dataNum << std::endl;
    //cal Analysis and CG train loss.

    Data testData = Data(Data::createX_matrixForPoly( testDataNum, order),
                         ProductContext(new SinStrategy));
    Y_matrix = testData.getY_matrix().col(1);
    testData.setY_Matrix(Y_matrix);
    Data *testDataPtr = &testData;
    //generate test dataset.

    dataMatrix W_matrixByBGD = LeastSquare::getW_matrixByBatchGradientDescent(trainData, norm, lambda, epochs,
                                                                              learningRate, testDataPtr);
    //BGD train and test loss output in the function.

    std::ofstream fout1("W_matrixByAnalysis.txt");
    fout1 << W_matrixByAnalysis;
    fout1.close();
    std::ofstream fout2("W_matrixByConjugateGradient.txt");
    fout2 << W_matrixByConjugateGradient;
    fout2.close();

    std::ofstream fout3("W_matrixByBGD.txt");
    fout3 << W_matrixByBGD;
    fout3.close();
    //save solution, use python script to draw figures.
    return 0;

}
