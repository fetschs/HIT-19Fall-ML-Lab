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

ProductContext::ProductContext() {

}

dataMatrix SinStrategy::productFunction(dataMatrix X_Matrix) {
    dataMatrix retMatrix = X_Matrix.array().sin();
    return retMatrix;
}
