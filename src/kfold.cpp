#include "../include/kfold.hpp"
#include "eigen-3.4.0/Eigen/Eigen"
#include <algorithm>

using namespace Eigen;

KFold::KFold(int k_folds, int n_samples, unsigned seed) : k(k_folds){
    indices.resize(n_samples);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0...n-1

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
}

std::vector<
    std::pair<
        std::vector<int>, std::vector<int>
    >
> 
    KFold::split() const {
    std::vector<std::pair<std::vector<int>, std::vector<int>>> folds;
    int fold_size = indices.size() / k;

    for (int i = 0; i < k; ++i) {
        int start = i * fold_size;
        int end   = (i == k - 1) ? indices.size() : (i + 1) * fold_size;

        std::vector<int> val_idx(indices.begin() + start, indices.begin() + end);
        std::vector<int> train_idx;

        train_idx.reserve(indices.size() - val_idx.size());
        for (int j = 0; j < indices.size(); ++j) {
            if (j < start || j >= end)
                train_idx.push_back(indices[j]);
        }

        folds.emplace_back(train_idx, val_idx);
    }

    return folds;
}

inline VectorXi to_Eigen(const std::vector<int>& v){
    VectorXi out(v.size());
    for (int i = 0; i < v.size(); ++i)
        out(i) = v[i];
    return out;
}

inline MatrixXd extract_submatrix(const MatrixXd& K, const std::vector<int>& row_idx, const std::vector<int>& col_idx) {
    const VectorXi row_eigen = to_Eigen(row_idx);
    const VectorXi col_eigen = to_Eigen(col_idx);

    return K(row_eigen, col_eigen);
}

inline VectorXd extract_vector(const VectorXd& v, const std::vector<int>& indices) {
    return v(to_Eigen(indices));
}