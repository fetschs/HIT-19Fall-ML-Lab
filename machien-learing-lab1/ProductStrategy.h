//
// Created by FETS on 2019/9/15.
//
#include <utility>

#include "BasicSettings.h"

#ifndef CLION_PRODUCTSTRATEGY_H
#define CLION_PRODUCTSTRATEGY_H


class ProductStrategy {
public:
    /**
     * product YMatrix by specific function.
     * @param XMatrix the XMatrix as input of f(x).
     */
    virtual dataMatrix productFunction(dataMatrix X_Matrix) = 0;

    virtual ~ProductStrategy();

protected:
    ProductStrategy();
};

class SinStrategy : public ProductStrategy {
public:
    SinStrategy();

    ~SinStrategy() override;


    dataMatrix productFunction(dataMatrix X_Matrix) override;

};

class ProductContext {
public:
    ProductContext();
    explicit ProductContext(ProductStrategy *pStrategy);

    ~ProductContext();
    /*
     * use strategy to product Y_matrix.
     */
    dataMatrix useStrategy(dataMatrix X_Matrix);
    /**
     * update product strategy.
     * @param pStrategy strategy will update.
     */
    void setProductStrategy(ProductStrategy *pStrategy);

private:
    ProductStrategy *productStrategy;
};

#endif //CLION_PRODUCTSTRATEGY_H
