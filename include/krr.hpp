#pragma once

#include "eigen-3.4.0/Eigen/Eigen"
#include <iostream>
#include <omp.h>

using namespace Eigen;

enum kernelType{
    GAUSSIAN,
    LAPLACIAN
};

enum lossMetric{
    MAE,
    RMSE
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
    const VectorXd get_alphas() const {return alphas;}
    const double   get_sigma () const {return sigma;}
    const double   get_lambda() const {return lambda;}

    //Setters
    void set_alphas(const VectorXd ALPHAS) {alphas = ALPHAS;}
    void set_sigma (const double SIGMA)    {sigma  = SIGMA;}
    void set_lambda(const double LAMBDA)   {lambda = LAMBDA;}

    //Methods
    void fit(const MatrixXd& TRAININGDATA, const VectorXd& TRAININGTARGET);

    VectorXd predict(const MatrixXd& TESTINGDATA) const;

    std::pair<VectorXd, double>
    evaluate(const MatrixXd& TESTINGDATA, const VectorXd& TESTINGTARGET, const lossMetric loss) const;

    //Friends
    friend MatrixXd 
    compute_kernel(const MatrixXd& A, const MatrixXd& B, const kernelType KERNEL, const double sigma);
};
