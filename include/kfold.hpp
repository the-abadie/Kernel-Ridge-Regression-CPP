#pragma once

#include "eigen-3.4.0/Eigen/Eigen"
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <fstream>
#include <sstream>

using namespace Eigen;

class KFold {
    public:
        KFold(int k) : k_(k) {}

        std::vector<std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>>
        split(const Eigen::MatrixXd& data, const Eigen::VectorXd& targets) {
            int n = data.rows();
            if (targets.size() != n) {
                throw std::invalid_argument("Number of rows in data must match size of targets.");
            }
    
            std::vector<int> indices(n);
            std::iota(indices.begin(), indices.end(), 0);
    
            std::vector<std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd>> folds;
    
            int fold_size = n / k_;
            for (int i = 0; i < k_; ++i) {
                int start = i * fold_size;
                int end = (i == k_ - 1) ? n : start + fold_size;
    
                std::vector<int> test_indices(indices.begin() + start, indices.begin() + end);
                std::vector<int> train_indices;
                train_indices.reserve(n - test_indices.size());
    
                for (int j = 0; j < n; ++j) {
                    if (j < start || j >= end) {
                        train_indices.push_back(indices[j]);
                    }
                }
    
                Eigen::MatrixXd X_train(train_indices.size(), data.cols());
                Eigen::VectorXd y_train(train_indices.size());
                Eigen::MatrixXd X_test(test_indices.size(), data.cols());
                Eigen::VectorXd y_test(test_indices.size());
    
                for (size_t j = 0; j < train_indices.size(); ++j) {
                    X_train.row(j) = data.row(train_indices[j]);
                    y_train(j) = targets(train_indices[j]);
                }
    
                for (size_t j = 0; j < test_indices.size(); ++j) {
                    X_test.row(j) = data.row(test_indices[j]);
                    y_test(j) = targets(test_indices[j]);
                }
    
                folds.emplace_back(X_train, y_train, X_test, y_test);
            }
    
            return folds;
        }
    
    private:
        int k_;
    };
    