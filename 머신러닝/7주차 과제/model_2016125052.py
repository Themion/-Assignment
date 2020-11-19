import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(x, 0)

class NeuralNetwork:
    def __init__(self, layerDims, 
                 learning_rate = 1, 
                 decay_rate = 0.2,
                 nSample = 0):
        
        ############

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.parameters = self.weightInit(layerDims)
        self.grads = {}
        self.cache = {}
        self.initialize_optimizer()
        
    def initialize_optimizer(self):
        size = self.parameters['size']
        VdW = [0 for _ in range(size)]
        Vdb = [0 for _ in range(size)]
        
        self.grads.update(VdW = VdW, Vdb = Vdb)
            
    def weightInit(self, layerDims):
        np.random.seed(1)
        
        W = [0 for _ in range(len(layerDims))]
        b = [0 for _ in range(len(layerDims))]
        
        for l in range(1, len(layerDims)):
            W[l] = np.random.randn(layerDims[l], layerDims[l - 1]) * np.sqrt(2 / layerDims[l - 1])
            b[l] = np.zeros((layerDims[l], 1))
            
        return {"W": W, "b": b, "size": len(layerDims)}
    
    def forward(self, X, training = True):
        W, b, size = self.parameters.values()
        
        Z = [0 for _ in range(size)]
        A = [0 for _ in range(size)]
        
        A[0] = X
        
        for i in range(1, size):
            Z[i] = np.dot(W[i], A[i - 1]) + b[i]
            A[i] = relu(Z[i])
        
        A[size - 1] = sigmoid(Z[size - 1])
        
        if training is True:
            self.cache.update(X = X, Z = Z, A = A)

        return A[size - 1]

    def backward(self, lambd = 0.7):
        W, b, size = self.parameters.values()
        VdW, Vdb = self.grads.values()
        A = self.cache['A']
        
        dZ = [0 for _ in range(size)]
        dW = [0 for _ in range(size)]
        db = [0 for _ in range(size)]
        
        m = float(A[0].shape[1])
        
        size = size - 1
        
        dZ[size] = A[size] - self.cache['Y']
        dW[size] = (np.dot(dZ[size], A[size - 1].T) + lambd * W[size]) / m
        db[size] = 1./m * np.sum(dZ[size], axis=1, keepdims = True)
        
        for i in range(size - 1, 1, -1):
            dZ[i] = np.multiply(np.dot(W[i + 1].T, dZ[i + 1]), np.int64(A[i] > 0))
            dW[i] = (np.dot(dZ[i], A[i - 1].T) + lambd * W[i]) / m
            db[i] = 1./m * np.sum(dZ[i], axis=1, keepdims = True)
        
        beta = 0.9
        div = 1
        
        for i in range(1, size):
            div *= beta
            VdW[i] = (beta * VdW[i - 1] + (1 - beta) * dW[i]) / (1 - div)
            Vdb[i] = (beta * Vdb[i - 1] + (1 - beta) * db[i]) / (1 - div)
        
        self.grads.update(VdW = VdW, Vdb = Vdb)

        return

    def compute_cost(self, X, Y):
        self.cache.update(Y = Y)
        lambd = 0.7
        
        W = self.parameters['W']
         
        sum_W = 0.
        for Wi in W:
            sum_W += np.sum(np.square(Wi))
            
        L2_cost = lambd / (2 * len(Y)) * sum_W
        logprobs = np.multiply(np.log(X), Y) + np.multiply(np.log(1 - X), 1 - Y)
        cost = -np.sum(logprobs) * (1 / len(Y)) + L2_cost
        
        cost = float(np.squeeze(cost))  

        assert(isinstance(cost, float))
        
        return cost
        
    def update_params(self, epoch_num):
        W, b, size = self.parameters.values()
        VdW, Vdb = self.grads.values()
        learning_rate = self.learning_rate / (1 + self.decay_rate * epoch_num)
            
        '''
        backward의 결과로 dW[1], db[1]과 VdW[1], Vdb[1]은 모두 0이 나오지만
        첫째, dW[1]과 dW[2]의 사이즈 차이로 VdW[2]를 계산하려 할 시에 에러가 발생함
        둘째, db[1]을 계산한 결과 원인 모를 이유로 정확도개 개선되지 않음
        따라서 VdW[2:]와 Vdb[2:]만을 사용한 결과 정확도가 개선됨을 확인하였으므로
        VdW[2:]만을 사용하려 W를 계산하기로 하였다
        '''
        
        for i in range(1, size):
            W[i] = W[i] - (learning_rate) * VdW[i]
            b[i] = b[i] - (learning_rate) * Vdb[i]
            
        self.parameters.update(W = W, b = b)
        
        return 

    def predict(self, X):
        return self.forward(X, training = False) > 0.5
        
    def hit_ratio(self, Y_, Y):
        return (np.dot(Y, Y_.T) + np.dot(1 - Y, 1 - Y_.T)) / float(Y.size)