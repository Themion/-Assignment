'''
Copyright (C)2020 KAURML  <ingeechart@kau.kr>

'''
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


def decision_boundary(model, X, Y, grid=0.1,name=None):

    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid), np.arange(y_min, y_max, grid))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.scatter(X[0, :], X[1, :], s=20, c=Y.ravel(), cmap=plt.cm.RdBu)
    plt.title(name)
    plt.show()
    

def generate_dataset():
    np.random.seed(1)
    N = 300 # number of examples
    nTrain = 200
    # X,Y = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    X,Y = sklearn.datasets.make_blobs(n_samples=N, centers=2, n_features=2, cluster_std=2.0, center_box=(-5,5),random_state=3)
    # X,Y = sklearn.datasets.make_classification(n_samples=N, n_features=2, n_clusters_per_class=2, n_classes=2, n_informative=2,n_redundant=0,random_state=4)

    
    X = X.T
    Y = Y.reshape(1,-1)

    X_train = X[:,:nTrain]
    X_test = X[:,nTrain:]
    Y_train = Y[:,:nTrain]
    Y_test = Y[:,nTrain:]


    return X_train, Y_train, X_test, Y_test

if __name__=='__main__':
    X_train, Y_train, X_test, Y_test = generate_dataset()    
    plt.scatter(X_train[0, :], X_train[1, :], c=Y_train.ravel(), cmap=plt.cm.RdBu)
    plt.show()