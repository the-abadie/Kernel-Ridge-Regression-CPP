#include "krr.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>

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

int main(int argc, char *argv[]){

    std::string in_path  = argv[1];
    std::string out_path = argv[2];
    
    MatrixXd trainingData = readCoulomb(in_path + "coulomb_train.txt");
    VectorXd trainingTrgt = readPBE0   (in_path + "PBE0_train.txt");

    MatrixXd testingData  = readCoulomb(in_path + "coulomb_test.txt");
    VectorXd testingTrgt  = readPBE0   (in_path + "PBE0_test.txt");

    std::cout << "Training Size: " << trainingData.rows() << "\n"; 
    std::cout << "Testing Size : "  << testingData.rows()  << "\n"; 

    VectorXd sigmas  = readPBE0(in_path + "sigmas.txt");
    VectorXd lambdas = readPBE0(in_path + "lambdas.txt");

    kernelType kernel = LAPLACIAN;
    lossMetric loss   = MAE;

    int k = 5;

    int nSigmas  = sigmas.rows();
    int nLambdas = lambdas.rows();
    
    MatrixXd MAEs = MatrixXd::Zero(sigmas.size(), lambdas.size());
    

    KFold kf(5);  // 5-fold CV
    auto folds = kf.split(trainingData, trainingTrgt);

    int CV_index = 1;
    double best_error = std::pow(10, 100);

    Vector2i best_param {0, 0};


    for(int i = 0; i < sigmas.size(); i++){
        for (int j = 0; j < lambdas.size(); j++){
            std::cout << "CV " << CV_index << " of " << sigmas.size()*lambdas.size() << "\n";
            CV_index++;

            VectorXd fold_errors = VectorXd::Zero(k);

            for (int n = 0; n < k; n++) {
                auto& [X_train, y_train, X_test, y_test] = folds[n];
            
                KRR folded_krr(sigmas[i], lambdas[j], kernel);

                folded_krr.fit(X_train, y_train, 0);

                auto eval = folded_krr.evaluate(X_test, y_test, loss, 0);

                fold_errors(n) = eval.second;
            };

            MAEs(i, j) = fold_errors.mean();

            if (MAEs(i, j) < best_error){
                best_error = MAEs(i, j);
                best_param << i, j;
            }
            
        };
    };

    double opt_sigma  = sigmas [best_param[0]];
    double opt_lambda = lambdas[best_param[1]];

    std::cout << "\nOptimal Sigma : " << opt_sigma;
    std::cout << "\nOptimal Lambda: " << opt_lambda << std::endl;

    std::ofstream file(strcat(argv[2], "MAE.txt"));
    if (file.is_open()){
        file << MAEs << std::endl;

        // Close the file
        file.close();
    }

    KRR finalModel(opt_sigma, opt_lambda, kernel);

    finalModel.fit(trainingData, trainingTrgt, 1);
    
    auto eval = finalModel.evaluate(testingData, testingTrgt, loss, 1);

    std::cout << "MAE: " << eval.second << "kcal/mol" << std::endl;

    auto PBE = testingTrgt.transpose();
    auto est = eval.first.transpose();

    std::ofstream file2(out_path + "Eref_Eest.txt");
    if (file2.is_open()){
        file2 << PBE << std::endl;
        file2 << est << std::endl;

        // Close the file
        file2.close();
    }
    return 0;
}
