//
// Created by FETS on 2019/9/15.
//

#include "Data.h"

element Data::generateRandom(element mean, element deviation) {
    const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);
    std::normal_distribution<element> dis(mean, deviation);
    return dis(engine);
}


void Data::addNoise(dataMatrix &dMatrix, element mean, element deviation) {

    for (int i = 0, row = dMatrix.rows(); i < row; i++) {
        for (int j = 0, col = dMatrix.cols(); j < col; j++) {
            dMatrix(i, j) += Data::generateRandom(mean, deviation);
        }
    }
}

dataMatrix Data::createX_matrix(int dataNum) {
    dataMatrix colVectorX = dataMatrix::Random(dataNum, 1);
    return colVectorX;
}

dataMatrix Data::createX_matrixForPoly(int dataNum, int order) {
    dataMatrix colVectorX = Data::createX_matrix(dataNum);
    for (int i = 0; i < dataNum; i++) {
        colVectorX(i, 0) *= std::acos(-1);
    }
    dataMatrix retMatrix = dataMatrix::Ones(dataNum, order + 1);
    for (int i = 1; i <= order; i++) {
        retMatrix.col(i) = colVectorX.array() * retMatrix.col(i - 1).array();
    }
    return retMatrix;
}

dataMatrix Data::createX_matrixForLogistic(dataMatrix covariance, int dataNum) {
    MultivariateNormalRandomVariable sample{covariance};
    int dimension = covariance.rows();
    dataMatrix sampleMatrix = dataMatrix::Ones(dataNum, dimension);
    for (int i = 0; i < dataNum; i++)
        sampleMatrix.row(i) = sample().transpose();
    dataMatrix retMatrix = dataMatrix::Ones(dataNum, dimension + 1);
    for (int i = 1; i <= dimension; i++)
        retMatrix.col(i) = sampleMatrix.col(i - 1);
    return retMatrix;
}

void Data::productY_matrix() {
    this->Y_matrix = std::move(this->context.useStrategy(this->X_matrix));
}

dataMatrix Data::getX_matrix() {
    return this->X_matrix;
}

dataMatrix Data::getY_matrix() {
    return this->Y_matrix;
}

void Data::setY_Matrix(const dataMatrix &yMatrix) {
    Y_matrix = yMatrix;
}

Data Data::getDataFromBankNote(const std::string& filename) {
    std::vector<element> line;
    line.clear();
    std::vector<std::vector<element>> lines;
    lines.push_back(line);
    element temp = 0;
    int lineInd = 0;
    std::ifstream fileIn(filename);
    assert(fileIn.is_open());
    while (!fileIn.eof()) {
        fileIn>>temp;
        lines[lineInd].push_back(temp);
        char ch;
        fileIn.get(ch);
        if(ch=='\n'){
            lineInd++;
            lines.push_back(line);
        }
    }

    dataMatrix retMatrix = dataMatrix::Zero(lineInd, lines[0].size());
    for (int i = 0; i < lineInd; i++) {
        for (int j = 0; j < lines[i].size(); j++) {
            retMatrix(i, j) = lines[i][j];
        }
    }
    Data retData = Data(retMatrix.leftCols(lines[0].size() - 1), retMatrix.rightCols(1));
    return retData;
}
