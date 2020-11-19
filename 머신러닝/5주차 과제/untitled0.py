import numpy as np
import matplotlib.pyplot as plt
from data_utils import decision_boundary, generate_dataset
import sklearn
import sklearn.datasets
np.random.seed(1)


def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(x, 0)

class NeuralNetwork:
    def __init__(self,layerDims, nSample):
        self.size = len(layerDims)
        
        ############

        self.nSample = nSample
        self.parameters = self.weightInit(layerDims)
        self.grads = {}
        self.vel = {}
        self.s = {}
        self.cache = {}
        ##self.initialize_optimizer()

    def weightInit(self, layerDims):
        np.random.seed(1)
        
        W = [0 for _ in range(len(layerDims))]
        b = [0 for _ in range(len(layerDims))]
        
        for l in range(1, len(layerDims)):
            W[l] = np.random.randn(layerDims[l], layerDims[l - 1]) * 0.1
            b[l] = np.zeros((layerDims[l], 1))

        return {"W": W, "b": b}

    
    
    def forward(self, X):
        W, b = self.parameters.values()
        size = self.size
        
        Z = [0 for _ in range(size)]
        A = [0 for _ in range(size)]
        
        A[0] = X
        
        for i in range(1, size):
            Z[i] = np.dot(W[i], A[i - 1]) + b[i]
            A[i] = relu(Z[i])
        
        A[size - 1] = sigmoid(Z[size - 1])
        
        self.cache.update(X = X, Z = Z, A = A)

        return A[size - 1]

    def backward(self, lambd = 0.7):
        W, b = self.parameters.values()
        A = self.cache['A']
        
        size = self.size - 1
        
        dW = [0 for _ in range(self.size)]
        db = [0 for _ in range(self.size)]
        dZ = [0 for _ in range(self.size)]
        
        m = A[0].shape[1]
        
        dZ[size] = A[size] - self.cache['Y']
        dW[size] = 1./m * np.dot(dZ[size], A[size - 1].T) + lambd / m * W[size]
        db[size] = 1./m * np.sum(dZ[size], axis=1, keepdims = True)
        
        for i in range(size - 1, 1, -1):
            dZ[i] = np.multiply(np.dot(W[i + 1].T, dZ[i + 1]), np.int64(A[i] > 0))
            dW[i] = 1./m * np.dot(dZ[i], A[i - 1].T) + lambd / m * W[i]
            db[i] = 1./m * np.sum(dZ[i], axis=1, keepdims = True)
        
        self.grads.update(dW = dW, db = db)

        return

    def compute_cost(self, fw, Y):
        self.cache.update(Y = Y)
        lambd=0.7
        
        size = self.size
        W = self.parameters['W']
        
        sum_W = 0.
        for i in W:
            sum_W += np.sum(np.square(i))
            
        L2_cost = lambd / (2 * Y.shape[1]) * sum_W
        logprobs = (np.multiply(np.log(fw), Y) + np.multiply(np.log(1 - fw), 1 - Y) )
        cost = -np.sum(logprobs) * (1 / Y.shape[1]) + L2_cost
        
        cost = float(np.squeeze(cost))  

        assert(isinstance(cost, float))
        
        return cost
    
    def update_params(self, learning_rate=1.2):
        W, b = self.parameters.values()
        dW, db = self.grads.values()
        
        for i in range(1, self.size):
            W[i] = W[i] - (learning_rate) * dW[i]
            b[i] = b[i] - (learning_rate) * db[i]
            
        self.parameters.update(W = W, b = b)
        
        return 

    def predict(self, X):
        return self.forward(X) > 0.5
    
def main():
    np.random.seed(1)
    num_iterations=12000
    learning_rate =0.3

    ### dataset loading 하기.
    X_train,Y_train, X_test, Y_test = generate_dataset()
    # plt.title("Data distribution")
    # plt.scatter(X_train[0, :], X_train[1, :], c=Y_train[0,:], s=20, cmap=plt.cm.RdBu)
    # plt.show()

    
    ## 코딩 시작
    nSample = X_train.shape[1]
    layerDims = [X_train.shape[0], 20, 7, 1]
    ## 코딩 끝

    simpleNN = NeuralNetwork(layerDims,nSample)
    training(X_train, Y_train, simpleNN, num_iterations, learning_rate)
    
def training(X_train, Y_train, simpleNN, num_iterations, learning_rate):

    ## 코딩시작
    # num_iterations 동안 training regularization이 없는 simple neural network를 학습하는 code를 작성하세요.
    # return 값으론, 학습된 모델을 출력하세요.

    print('Starting training')

    _, _, X_test, Y_test = generate_dataset()
    
    for i in range(0, num_iterations):
        A3 = simpleNN.forward(X_train)
        cost = simpleNN.compute_cost(A3, Y_train)
        simpleNN.backward()
        simpleNN.update_params()
        if i % 1000 ==0:
            print("Cost after iteration %i: %f" %(i,cost))

    predictions = simpleNN.predict(X_train)
    print ('Train Accuracy: %d' % float((np.dot(Y_train,predictions.T) + np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')
    decision_boundary(lambda x: simpleNN.predict(x.T), X_train, Y_train, name="Train Dataset")
    predictions = simpleNN.predict(X_test)
    print ('Test Accuracy: %d' % float((np.dot(Y_test,predictions.T) + np.dot(1-Y_test,1-predictions.T))/float(Y_test.size)*100) + '%')

    print('--------------------------------------------')
    
    ##코딩 끝 

    # predictions = simpleNN.predict(X_train)
    # print ('Accuracy: %d' % float((np.dot(Y_train,predictions.T) + np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')
    # decision_boundary(lambda x: simpleNN.predict(x.T), X_train, Y_train, name="Train Dataset")

    return simpleNN

if __name__=='__main__':
    main()
