#include "eigen-3.4.0/Eigen/Eigen"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <tuple>

using namespace Eigen;

enum kernelType{
    GAUSSIAN,
    LAPLACIAN
};

enum lossMetric{
    MAE,
    RSME
};

class KRR{
private:
    double sigma;
    double lambda;
    kernelType kernel;

    VectorXd alphas;
    MatrixXd trainingData;

public:
    //Constructors
    KRR(const double SIGMA, const double LAMBDA, const kernelType KERNEL){
        sigma  = SIGMA;
        lambda = LAMBDA;
        kernel = KERNEL;
    };
    
    //Getters
    const VectorXd get_alphas(){return alphas;}
    const double   get_sigma (){return sigma;}
    const double   get_lambda(){return lambda;}

    //Setters
    void set_alphas(const VectorXd ALPHAS){alphas = ALPHAS;}
    void set_sigma (const double SIGMA)   {sigma  = SIGMA;}
    void set_lambda(const double LAMBDA)  {lambda = LAMBDA;}

    //Methods
    void fit(const MatrixXd TRAININGDATA, const VectorXd TRAININGTARGET, int verbose){
        auto start = std::chrono::high_resolution_clock::now();

        trainingData = TRAININGDATA;

        int nTrain = trainingData.rows();

        MatrixXd K = MatrixXd::Zero(nTrain, nTrain);

        // Construct Kernel Matrix
        double dst = 1./(-2.*sigma*sigma);

        if (verbose > 0){std::cout << "Training model...\n";};

        for(int i = 0; i < nTrain; i++){
            for(int j = 0; j < nTrain; j++){
                if      (i == j){K(i, j) = 1.     ; continue;}
                else if (i  > j){K(i, j) = K(j, i); continue;}

                K(i, j) = 
                    std::exp((trainingData(i, all) - trainingData(j, all)).squaredNorm() * dst);
            }
        }

        // Construct lambdaI
        MatrixXd lambdaI = MatrixXd::Identity(nTrain, nTrain)*lambda;
        
        //Get alphas
        alphas = ((K + lambdaI).inverse())*TRAININGTARGET;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (verbose > 0){std::cout << "Model trained in " << duration.count() << " ms.\n" << std::endl;};
    };

    VectorXd predict(const MatrixXd testing_data, const int verbose){
        const auto start = std::chrono::high_resolution_clock::now();

        const int nTest = testing_data.rows();
        const double dst = 1./(-2.*sigma*sigma);
        
        VectorXd predictions = VectorXd::Zero(nTest);

        for (int i = 0; i < nTest; i++){
            MatrixXd T_x = (-trainingData).rowwise() + testing_data(i, all);

            VectorXd D_x = T_x.rowwise().squaredNorm();
            D_x *= dst;

            VectorXd D_exp = D_x.array().exp();

            predictions(i) = (alphas.array() * D_exp.array()).sum();
        }

        const auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (verbose > 0){std::cout << "Evaluated " << nTest << " points in " << duration.count() << " ms." << std::endl;};
        return predictions;
    };

    std::pair<VectorXd, double> evaluate(MatrixXd testing_data, VectorXd testing_trgt, lossMetric loss, int verbose){
        auto start = std::chrono::high_resolution_clock::now();

        int nTest = testing_data.rows();
        double dst = (-2.*sigma*sigma);
         
        VectorXd predictions = VectorXd::Zero(nTest);

        for (int i = 0; i < nTest; i++){
            MatrixXd T_x = (-trainingData).rowwise() + testing_data(i, all);

            VectorXd D_x = T_x.rowwise().squaredNorm();
            D_x /= dst;

            VectorXd D_exp = D_x.array().exp();

            predictions(i) = (alphas.array() * D_exp.array()).sum();
        }

        double error = 0.;

        if (loss == MAE){
            error = (predictions - testing_trgt).cwiseAbs().mean();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (verbose > 0){std::cout << "Evaluated " << nTest << " points in " << duration.count() << " ms." << std::endl;};

        return {predictions, error};
    };
};

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
