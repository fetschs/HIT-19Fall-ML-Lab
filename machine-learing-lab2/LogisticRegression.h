//
// Created by FETS on 2019/10/20.
//

#ifndef MACHIEN_LEARING_LAB2_LOGISTICREGRESSION_H
#define MACHIEN_LEARING_LAB2_LOGISTICREGRESSION_H


#include "LearingAlgorithm.h"

class LogisticRegression : public LearningAlgorithm {
public:

    dataMatrix cal_derivative(Data inputData, const dataMatrix &W_matrix, int norm, element lambda) override;

    element cal_Loss(Data inputData, const dataMatrix &W_matrix, int norm, element lambda) override;

    LogisticRegression() = default;

    ~LogisticRegression() override;
};


#endif //MACHIEN_LEARING_LAB2_LOGISTICREGRESSION_H
