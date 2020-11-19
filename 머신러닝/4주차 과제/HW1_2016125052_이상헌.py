'''
Copyright (C)2020 KAURML  <ingeechart@kau.kr>

'''

import numpy as np
import matplotlib.pyplot as plt
from data_utils import decision_boundary, generate_dataset
import sklearn
import sklearn.datasets
import sklearn.linear_model
np.random.seed(1)


def sigmoid(x):
    """
    sigmoid 함수

    Arguments:
        x:  scalar 또는 numpy array

    Return:
        s:  sigmoid(x)
    """

    ## 코딩시작
    s = 1 / (1 + np.exp(-x))
    ## 코딩 끝

    return s

class NeuralNetwork:
    def __init__(self, nInput, nHidden, nOutput, nSample):
        '''
        학습할 네트워크. 

        Arguments:
            nInput: input layer의 크기
            nHidden: hidden unit의 개수
            nOutput: output layer의 크기
            nSample: 데이터셋의 샘플 수
        '''

        self.nSample = nSample
        self.parameters = {"W1": 1,
                        "b1": 1,
                        "W2": 1,
                        "b2": 1}

        self.cache = {"X": 1,
                    "Y": 1,
                    "Z1": 1,
                    "A1": 1,
                    "Z2": 1,
                    "A2": 1,}

        self.grads = {"dW1": 1,
                    "db1": 1,
                    "dW2": 1,
                    "db2": 1}
        
        self.weightInit(nInput, nHidden, nOutput)

    def weightInit(self, nInput, nHidden, nOutput):
        '''
        network parameter 초기화
        
        Arguments:
            nInput: input layer의 크기
            nHidden: hidden unit의 개수
            nOutput: output layer의 크기

        Return:
        '''

        ## 코딩시작
        W1 = np.random.randn(nHidden, nInput) * 0.1
        b1 = np.zeros((nHidden, 1))
        W2 = np.random.randn(nOutput, nHidden) * 0.1
        b2 = np.zeros((nOutput, 1))
        ## 코딩끝

        self.parameters.update(W1=W1, b1=b1, W2=W2, b2=b2)
        
        return 

    def forward(self, X):
        '''
        forward propagation

        Arguments:
            X: input data

        Return:
            A2: network output
        '''
        ## 코딩시작 
        W1, b1, W2, b2 = self.parameters.values()
        
        Z1 = np.dot(W1,X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = sigmoid(Z2)

        ## 코딩끝
        
        self.cache.update(X=X, Z1=Z1, A1= A1, Z2=Z2, A2=A2)

        return A2

    def backward(self):
        '''
        backward propagation. gradients를 구한다.

        Arguments:

        Return:
        '''
        ## 코딩시작
        W1, b1, W2, b2 = self.parameters.values()
        X, Y, _, A1, _, A2 = self.cache.values()

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / X.shape[1]
        db2 = np.sum(dZ2, axis = 1,keepdims = True) / X.shape[1]
        
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) / X.shape[1]
        db1 = np.sum(dZ1, axis = 1,keepdims = True) / X.shape[1]
        ## 코딩끝

        self.grads.update(dW1=dW1, db1= db1, dW2=dW2, db2=db2)

        return

    def update_params(self, learning_rate=1.2):
        '''
        backpropagation을 통해 얻은 gradients를 update한다.

        Arguments:
            learning_rate:  학습할 learning rate

        Return:
        '''

        ## 코딩시작
        W1,b1,W2,b2 = self.parameters.values()
        dW1, db1, dW2, db2 = self.grads.values()

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        ## 코딩 끝
        
        self.parameters.update(W1=W1, b1= b1, W2=W2, b2=b2)

        return 

    def compute_cost(self, A2, Y):
        '''
        cross-entropy loss를 이용하여 cost를 구한다.

        Arguments:
            A2 : network 결과값
            Y  : 정답 label(groud truth)
        Return:
            cost
        '''
        self.cache.update(Y=Y)
        
        ## 코딩 시작
        logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
        cost = -np.sum(logprobs) * (1 / Y.shape[1]) 
        ## 코딩끝
        
        cost = float(np.squeeze(cost))  

        return cost
    
    def predict(self,X):
        '''
        학습한 network가 잘 학습했는지, test set을 통해 확인한다.

        Arguments:
            X: input data
        Return:
        '''
        ## 코딩 시작
        A2 = self.forward(X)
        predictions = (A2 > 0.5)
        ## 코딩 끝
        return predictions

def main():
    np.random.seed(1)
    num_iterations=10000

    ### dataset loading 하기.
    X_train,Y_train, X_test, Y_test = generate_dataset()
    # plt.title("Data distribution")
    # plt.scatter(X_train[0, :], X_train[1, :], c=Y_train[0,:], s=20, cmap=plt.cm.RdBu)
    # plt.show()
    
    ## 코딩시작
    nInput = 2
    nHidden = 4
    nOutput = 1
    nSample = 1000
    ## 코딩시작

    simpleNN = NeuralNetwork(nInput, nHidden, nOutput, nSample)

    ### training
    ## 코딩 시작
    for i in range(0, num_iterations):
        ##forward propagation 하기.
        A2 = simpleNN.forward(X_train)
        ## cost function을 이용해 cost 구하기.
        cost = simpleNN.compute_cost(A2, Y_train)
        ## 위에 만든 함수를 이용하여 backpropagation 진행
        simpleNN.backward()
        ## 위에 만든 함수를 이용하여 network의 weight update하기.
        simpleNN.update_params()
        if i % 1000 ==0:
            print("Cost after iteration %i: %f" %(i,cost))
    ## 코딩끝

    ### prediction
    predictions = simpleNN.predict(X_train)
    print ('Accuracy: %d' % float((np.dot(Y_train,predictions.T) + np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')
    # decision_boundary(lambda x: simpleNN.predict(x.T), X_train, Y_train)

    predictions = simpleNN.predict(X_test)
    print ('Accuracy: %d' % float((np.dot(Y_test,predictions.T) + np.dot(1-Y_test,1-predictions.T))/float(Y_test.size)*100) + '%')
    # decision_boundary(lambda x: simpleNN.predict(x.T), X_test, Y_test)

if __name__=='__main__':
    main()
