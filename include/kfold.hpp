#pragma once
#include <vector>
#include <random>
#include "eigen-3.4.0/Eigen/Eigen"

using namespace Eigen;

class KFold {
private:
    int k;
    std::vector<int> indices;

public:
    KFold(int k_folds, int n_samples, unsigned seed = 42);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> split() const;

};

MatrixXd extract_submatrix(const MatrixXd& K, const std::vector<int>& row_idx, const std::vector<int>& col_idx);

VectorXd extract_vector(const VectorXd& v, const std::vector<int>& indices);