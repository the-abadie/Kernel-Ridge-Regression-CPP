#include "krr.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

VectorXd readPBE0(std::string filepath){
    std::ifstream file(filepath);

    if (!file.is_open()){std::cerr << "Failed to open PBE0 file.\n"; exit(1);};

    std::string line;
    std::vector<double> E;

    while(std::getline(file, line)){
        E.push_back(std::stod(line));
    }
    file.close();

    VectorXd E_vec = VectorXd::Zero(E.size());

    for (int i = 0; i < E.size(); i++){
        E_vec(i) = E[i];
    }

    return E_vec;
}

MatrixXd readCoulomb(std::string filepath){
    std::ifstream file(filepath);

    if (!file.is_open()){std::cerr << "Failed to open eigenvalue file.\n"; exit(1);};

    std::string line;
    std::vector<std::vector<double>> descriptors;

    while(std::getline(file, line)){
        std::istringstream iss(line);
        std::vector<double> row;
        double ev;

        while(iss >> ev){
            row.push_back(ev);
        }

        if(!row.empty()){
            descriptors.emplace_back(row);
        }
    }

    file.close();

    MatrixXd D = MatrixXd::Zero(descriptors.size(), descriptors[0].size());

    for (int i = 0; i < descriptors.size(); i++){
        for (int j = 0; j < descriptors[i].size(); j++){
            D(i, j) = descriptors[i][j];
        }
    }

    return D;
}

int main(){

    MatrixXd trainingData = readCoulomb("./data/trainingSplit/coulomb_train.txt");
    VectorXd trainingTrgt = readPBE0   ("./data/trainingSplit/PBE0_train.txt");

    MatrixXd testingData  = readCoulomb("./data/trainingSplit/coulomb_test.txt");
    VectorXd testingTrgt  = readPBE0   ("./data/trainingSplit/PBE0_test.txt");

    std::cout << "Training Size : " << trainingData.rows() << "\n"; 
    std::cout << "Testing  Size : "  << testingData.rows()  << "\n"; 

    std::vector<double> sigmas;
    std::vector<double> lambdas;
    kernelType kernel = GAUSSIAN;
    lossMetric loss   = MAE;

    double sigma  = 10;
    double lambda = 0.001; 

    KRR fit1(sigma, lambda, kernel);

    fit1.fit(trainingData, trainingTrgt, 1);

    auto pred_fit1 = fit1.predict(testingData, 1);

    std::cout << "Vec AVG Prediction " << pred_fit1.sum() << " kcal/mol" << std::endl;
    return 0;
}
