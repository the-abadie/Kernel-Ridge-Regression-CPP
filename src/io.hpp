#include "eigen-3.4.0/Eigen/Eigen"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace Eigen;

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

void writeEnergies(const std::string filepath, const VectorXd& PBE0, const VectorXd& pred){
    std::ofstream file(filepath + "Eref_Eest.txt");

    RowVectorXd PBE0T = PBE0.transpose();
    RowVectorXd predT = pred.transpose();

    if (file.is_open()){
        file << PBE0T << std::endl;
        file << predT << std::endl;

        // Close the file
        file.close();
    }

}

void writeAlphas(const std::string filepath, const VectorXd& alphas){
    std::ofstream file(filepath + "alphas.txt");

    if (file.is_open()){
        file << alphas << std::endl;

        // Close the file
        file.close();
    }
}

void writeMAEs(const std::string filepath, const MatrixXd& MAEs){
    std::ofstream file(filepath + "MAEs.txt");

    if (file.is_open()){
        file << MAEs << std::endl;
    
        // Close the file
        file.close();
    }
}
