//
// Created by FETS on 2019/9/15.
//

#include "ProductStrategy.h"


ProductStrategy::~ProductStrategy() = default;

ProductStrategy::ProductStrategy() = default;

SinStrategy::SinStrategy() = default;

SinStrategy::~SinStrategy() = default;


void ProductContext::setProductStrategy(ProductStrategy *pStrategy) {
    this->productStrategy = pStrategy;
}

ProductContext::~ProductContext() = default;


dataMatrix ProductContext::useStrategy(dataMatrix X_Matrix) {
    return this->productStrategy->productFunction(std::move(X_Matrix));
}

ProductContext::ProductContext(ProductStrategy *pStrategy) : productStrategy(pStrategy) {
}

ProductContext::ProductContext() = default;

dataMatrix SinStrategy::productFunction(dataMatrix X_matrix) {
    dataMatrix retMatrix = X_matrix.array().sin();
    return retMatrix;
}

LinerStrategy::LinerStrategy() = default;

LinerStrategy::~LinerStrategy() = default;

dataMatrix LinerStrategy::productFunction(dataMatrix X_matrix) {
    element mean = X_matrix.sum() / X_matrix.size();
    dataMatrix retMatrix = dataMatrix::Zero(X_matrix.rows(), 1);
    for (int i = 0; i < X_matrix.rows(); i++) {
        if (X_matrix.row(i).sum() > mean) {
            retMatrix(i, 0) = 1;
        }
    }
    return retMatrix;
}
