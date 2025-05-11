#include "../include/krr.hpp"
#include "../include/kfold.hpp"
#include "../include/io.hpp"

int main(int argc, char *argv[]){

    const int         k        = std::stoi(argv[1]);
    const int         nThreads = std::stoi(argv[2]);
    const std::string in_path  = argv[3];
    const std::string out_path = argv[4];

    omp_set_num_threads(nThreads);

    const MatrixXd trainingData = readDescriptors(in_path + "train_data.txt");
    const VectorXd trainingTrgt = readTargets    (in_path + "train_trgt.txt");

    const MatrixXd testingData  = readDescriptors(in_path + "test_data.txt");
    const VectorXd testingTrgt  = readTargets    (in_path + "test_trgt.txt");

    std::cout << "Training Size: " << trainingData.rows() << "\n"; 
    std::cout << "Testing Size : "  << testingData.rows()  << "\n"; 

    const VectorXd sigmas  = readTargets(in_path + "sigmas.txt");
    const VectorXd lambdas = readTargets(in_path + "lambdas.txt");

    const kernelType kernel = GAUSSIAN;
    const lossMetric loss   = MAE;

    const int nSigmas  = sigmas.rows();
    const int nLambdas = lambdas.rows();
    
    MatrixXd MAEs = MatrixXd::Zero(sigmas.size(), lambdas.size());

    KFold kf(k);  // 5-fold CV
    auto folds = kf.split(trainingData, trainingTrgt);

    Vector2i best_param {0, 0};

    auto start = std::chrono::high_resolution_clock::now();
    
    int CV_index = 0;

    for (int i = 0; i < sigmas.size(); i++){
        for (int j = 0; j < lambdas.size(); j++){
            std::cout << "CV " << CV_index << " of " << sigmas.size()*lambdas.size() << "\n";
            CV_index++;

            double fold_error = 0.0;
            #pragma omp parallel for
            for (int n = 0; n < k; n++) {
                auto &[X_train, y_train, X_test, y_test] = folds[n];
            
                KRR folded_krr(sigmas[i], lambdas[j], kernel);

                folded_krr.fit(X_train, y_train);

                auto eval = folded_krr.evaluate(X_test, y_test, loss);

                fold_error += eval.second;
                std::cout << "Fold" << n << "complete.\n"; 
            };

            MAEs(i, j) = fold_error / k;
        };
    };

    double lowestError = std::pow(10, 100);
    for (int i=0; i < MAEs.rows(); i++){
        for (int j=0; j < MAEs.cols(); j++){
            if(MAEs(i, j) < lowestError){
                lowestError = MAEs(i, j);
                best_param  = {i, j};
            }
        }
    }

    const double opt_sigma  = sigmas [best_param[0]];
    const double opt_lambda = lambdas[best_param[1]];

    std::cout << "\nOptimal Sigma : " << opt_sigma;
    std::cout << "\nOptimal Lambda: " << opt_lambda << std::endl;

    KRR finalModel(opt_sigma, opt_lambda, kernel);

    finalModel.fit(trainingData, trainingTrgt);
    
    auto eval = finalModel.evaluate(testingData, testingTrgt, loss);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << "Training completed after " << duration.count() << " seconds.\n" << std::endl;
    
    std::cout << "MAE: " << eval.second << " kcal/mol" << std::endl;

    writeTargets (out_path, testingTrgt, eval.first);
    writeAlphas  (out_path, finalModel.get_alphas());
    writeMAEs    (out_path, MAEs);
    writeHPs     (out_path, opt_sigma, opt_lambda, kernel);
    writeEout    (out_path, eval.second);
    return 0;
}
