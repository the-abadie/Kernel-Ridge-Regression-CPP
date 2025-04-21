#include "eigen-3.4.0/Eigen/Eigen"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

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
    void fit(const MatrixXd& TRAININGDATA, const VectorXd& TRAININGTARGET, const int verbose){
        auto start = std::chrono::high_resolution_clock::now();

        trainingData = TRAININGDATA;

        int nTrain = trainingData.rows();

        MatrixXd K = MatrixXd::Zero(nTrain, nTrain);

        // Construct Kernel Matrix

        double dst = 0.;

        if     (kernel == GAUSSIAN) {dst = 1./(-2.*sigma*sigma);}
        else if(kernel == LAPLACIAN){dst = 1./-sigma;}
        else   {std::cerr << "Kernel not specified, or not one of the allowed types."; exit(-1);}

        if (verbose > 0){std::cout << "Training model...\n";};

        for(int i = 0; i < nTrain; i++){
            for(int j = 0; j < nTrain; j++){
                if      (i == j){K(i, j) = 1.     ; continue;}
                else if (i  > j){K(i, j) = K(j, i); continue;}

                if (kernel == GAUSSIAN){
                    K(i, j) = 
                        std::exp((trainingData.row(i) - trainingData.row(j)).squaredNorm() * dst);
                }
                else if (kernel == LAPLACIAN){
                    K(i, j) = 
                        std::exp((trainingData.row(i) - trainingData.row(j)).lpNorm<1>() * dst);
                }
                
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

    VectorXd predict(const MatrixXd& testing_data, const int verbose){
        const auto start = std::chrono::high_resolution_clock::now();

        const int nTest = testing_data.rows();

        double dst = 0.;
        if     (kernel == GAUSSIAN) {dst = 1./(-2.*sigma*sigma);}
        else if(kernel == LAPLACIAN){dst = 1./-sigma;}
        else   {std::cerr << "Kernel not specified, or not one of the allowed types."; exit(-1);}
        
        VectorXd predictions = VectorXd::Zero(nTest);

        MatrixXd T_x(trainingData.rows(), trainingData.cols());
        VectorXd D_x(trainingData.rows());
        VectorXd D_exp(trainingData.rows());

        #pragma omp parallel for private(T_x, D_x, D_exp)
        for (int i = 0; i < nTest; i++){
            T_x = (-trainingData).rowwise() + testing_data.row(i);

            D_x = T_x.rowwise().squaredNorm();
            
            if      (kernel == GAUSSIAN) {D_x = T_x.rowwise().squaredNorm();}
            else if (kernel == LAPLACIAN){D_x = T_x.rowwise().lpNorm<1>();}
            D_x *= dst;

            D_exp = D_x.array().exp();

            predictions(i) = (alphas.array() * D_exp.array()).sum();
        }

        const auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (verbose > 0){std::cout << "Evaluated " << nTest << " points in " << duration.count() << " ms." << std::endl;};
        return predictions;
    };

    std::pair<VectorXd, double> evaluate(const MatrixXd& testing_data, const VectorXd& testing_trgt, const lossMetric loss, const int verbose){
        auto start = std::chrono::high_resolution_clock::now();

        int nTest = testing_data.rows();
        double dst = 0.;
        if     (kernel == GAUSSIAN) {dst = 1./(-2.*sigma*sigma);}
        else if(kernel == LAPLACIAN){dst = 1./-sigma;}
        else   {std::cerr << "Kernel not specified, or not one of the allowed types."; exit(-1);}
        
        VectorXd predictions = VectorXd::Zero(nTest);

        MatrixXd T_x(trainingData.rows(), trainingData.cols());
        VectorXd D_x(trainingData.rows());
        VectorXd D_exp(trainingData.rows());

        #pragma omp parallel for private(T_x, D_x, D_exp)
        for (int i = 0; i < nTest; i++){
            T_x = (-trainingData).rowwise() + testing_data.row(i);

            D_x = T_x.rowwise().squaredNorm();
            
            if      (kernel == GAUSSIAN) {D_x = T_x.rowwise().squaredNorm();}
            else if (kernel == LAPLACIAN){D_x = T_x.rowwise().lpNorm<1>();}
            D_x *= dst;

            D_exp = D_x.array().exp();

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
