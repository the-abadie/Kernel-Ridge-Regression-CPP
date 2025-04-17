import numpy as np
import numpy.linalg as lin
import sklearn as skl
from sklearn.model_selection import KFold

def euclidian_norm(v1, v2):
    assert len(v1) == len(v2), "Vectors must be same size."
    tmp = (v1 - v2)**2

    return np.sqrt(np.sum(tmp))

def KRR_evaluate(x, X, alphas, sigma):
    sum = 0
    for i in range(len(alphas)):
        sum += alphas[i]*np.exp(-euclidian_norm(x, X[i])**2 / (2*sigma**2))
    return sum

def KRR_train(training_data, training_trgt, sigma, lamda):
    K = np.zeros((len(training_data), len(training_data)))

    for i in range(len(training_data)):
        for j in range(len(training_data)):
            K[i][j] = np.exp(-euclidian_norm(training_data[i], training_data[j])**2 / (2*sigma**2))
    
    lambdaI = np.identity(len(training_data))*lamda

    alphas  = lin.inv(K + lambdaI)@training_trgt

    mae = 0
    for i in range(len(training_data)):
        pred = KRR_evaluate(training_data[i], training_data, alphas, sigma) 
        mae += np.abs(pred - training_trgt[i])
    
    return alphas, mae


trainingData = np.array([[1, 2],
                         [3, 4],
                         [5, 6],
                         [7, 8]])

trainingTrgt = np.array([11, 12, 13, 14])

lamda = 0.001
sigma = 10

alphas, mae = KRR_train(trainingData, trainingTrgt, sigma=sigma, lamda=lamda)

testData = np.array([[3, 7],
                     [4, 8],
                     [5, 9],
                     [6, 10]])

predictions = []
for i in range(len(testData)):
    predictions.append(KRR_evaluate(testData[i], trainingData, sigma = sigma, alphas= alphas))

print("Alphas:", alphas)
print(np.array(predictions))
