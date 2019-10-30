//
// Created by FETS on 2019/10/21.
//

#ifndef MACHIEN_LEARING_LAB2_MULTIVARIATENORMALRANDOMVARIABLE_H
#define MACHIEN_LEARING_LAB2_MULTIVARIATENORMALRANDOMVARIABLE_H

#include <utility>

#include "BasicSettings.h"

class MultivariateNormalRandomVariable {
public:
    explicit MultivariateNormalRandomVariable(Eigen::MatrixXd const &covariance)
            : MultivariateNormalRandomVariable(Eigen::VectorXd::Zero(covariance.rows()), covariance) {}

    MultivariateNormalRandomVariable(Eigen::VectorXd mean, Eigen::MatrixXd const &covariance)
            : mean(std::move(mean)) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> EigenSolver(covariance);
        transform = EigenSolver.eigenvectors() * EigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const {
        static std::mt19937 gen{std::random_device{}()};
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{mean.size()}.unaryExpr([&](auto x) { return dist(gen); });
    }
};


#endif //MACHIEN_LEARING_LAB2_MULTIVARIATENORMALRANDOMVARIABLE_H
