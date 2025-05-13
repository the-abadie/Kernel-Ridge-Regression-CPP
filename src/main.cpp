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
    std::cout << "Testing Size : " << testingData.rows()  << std::endl; 

    const VectorXd sigmas  = readTargets(in_path + "sigmas.txt");
    const VectorXd lambdas = readTargets(in_path + "lambdas.txt");

    const kernelType kernel = GAUSSIAN;
    const lossMetric loss   = MAE;

    const int nSigmas  = sigmas.rows();
    const int nLambdas = lambdas.rows();
    
    MatrixXd MAEs = MatrixXd::Zero(sigmas.size(), lambdas.size());

    Vector2i best_param {0, 0};
    
    int CV_index = 0;

    KFold kf(k, trainingData.rows(), 69);
    auto folds = kf.split();

    for (int i = 0; i < sigmas.size(); ++i){
        //Compute full training kernel matrix per sigma
        std::cout << "\nComputing training kernel for sigma " << i << "...\n";
        const MatrixXd K_FULL = compute_kernel(trainingData, trainingData, kernel, sigmas(i));
        std::cout << "K_FULL computed. Searching lambda space...\n\n";

        // Pre-compute all fold-specific matrices and vectors (once per sigma)
        std::vector<MatrixXd> K_train_folds(k);
        std::vector<MatrixXd> K_val_folds(k);
        std::vector<VectorXd> y_train_folds(k);
        std::vector<VectorXd> y_val_folds(k);

        #pragma omp parallel for num_threads(k)
        for (int n = 0; n < k; ++n) {
            auto &[train_idx, val_idx] = folds[n];
            K_train_folds[n] = extract_submatrix(K_FULL, train_idx, train_idx);
            K_val_folds[n]   = extract_submatrix(K_FULL, val_idx, train_idx);
            y_train_folds[n] = extract_vector(trainingTrgt, train_idx);
            y_val_folds[n]   = extract_vector(trainingTrgt, val_idx);
}
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < lambdas.size(); ++j){
            std::cout << "lambda " << j << std::endl;
            double val_error = 0.0; 

            #pragma omp parallel for num_threads(k) reduction(+:val_error)
            for (int n = 0; n < k; ++n){
    
                auto &[train_idx, val_idx] = folds[n];
    
                MatrixXd& K_train = K_train_folds[n];
                MatrixXd& K_val = K_val_folds[n];
                VectorXd& y_train = y_train_folds[n];
                VectorXd& y_val = y_val_folds[n];

                KRR krr_fold(sigmas(i), lambdas(j), kernel);
            
                krr_fold.fit(K_train, y_train);
                val_error += krr_fold.evaluate_kernel(K_val, y_val, loss);    
            };
            MAEs(i, j)  = val_error;
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
    
    VectorXd predictions = finalModel.predict(testingData);
    double   final_eval  = finalModel.evaluate(testingData, testingTrgt, loss);
        
    std::cout << "MAE: " << final_eval << std::endl;

    writeTargets (out_path, testingTrgt, predictions);
    writeAlphas  (out_path, finalModel.get_alphas());
    writeMAEs    (out_path, MAEs);
    writeHPs     (out_path, opt_sigma, opt_lambda, kernel);
    writeEout    (out_path, final_eval);
    return 0;
}
