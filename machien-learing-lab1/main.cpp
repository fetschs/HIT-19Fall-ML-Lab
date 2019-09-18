//
// Created by FETS on 2019/9/17.
//

#include "LeastSquare.h"
int main() {
    const int dataNum = 10;
    const int order = 6;
    const int norm = 0;
    const element lambda = 8;
    const int epochs = 5000000;
    const element learningRate = 0.000001;


    Data trainData = Data(Data::createX_matrixForPoly(dataNum, order), ProductContext(new SinStrategy));
    dataMatrix Y_matrix = trainData.getY_matrix().col(1);
    Data::addNoise(Y_matrix, 0, 1);
    trainData.setY_Matrix(Y_matrix);
    dataMatrix W_matrixByAnalysis = LeastSquare::getW_matrixByAnalysis(trainData, norm, lambda);
    std::cout << "Analysis loss: " << LeastSquare::cal_Loss(trainData, W_matrixByAnalysis, norm, lambda) << std::endl;


    dataMatrix W_matrixByConjugateGradient = LeastSquare::getW_matrixByConjugateGradient(trainData, norm, lambda);
    std::cout << "ConjugateGradient loss: "
              << LeastSquare::cal_Loss(trainData, W_matrixByConjugateGradient, norm, lambda) << std::endl;


    Data testData = Data(Data::createX_matrixForPoly(dataNum * 100, order),
                         ProductContext(new SinStrategy));
    Y_matrix = testData.getY_matrix().col(1) * 10;
    Data::addNoise(Y_matrix, 0, 1);
    testData.setY_Matrix(Y_matrix);
    Data* testDataPtr = &testData;


    dataMatrix W_matrixByBGD = LeastSquare::getW_matrixByBatchGradientDescent(trainData, norm, lambda, epochs,
                                                                              learningRate,testDataPtr);
    std::cout << "BGD loss: " << LeastSquare::cal_Loss(trainData, W_matrixByBGD, norm, lambda) / dataNum << std::endl;

    std::ofstream fout1("W_matrixByAnalysis.txt");
    fout1<<W_matrixByAnalysis;
    fout1.close();
    std::ofstream fout2("W_matrixByConjugateGradient.txt");
    fout2<<W_matrixByConjugateGradient;
    fout2.close();

    std::ofstream fout3("W_matrixByBGD.txt");
    fout3<<W_matrixByBGD;
    fout3.close();

    return 0;

}
