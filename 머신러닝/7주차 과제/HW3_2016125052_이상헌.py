import numpy as np
import dataUtils
import model_2016125052 as model

import json
from json import JSONEncoder
import numpy

np.random.seed(1)

'''
TrainSet Accuracy with best NN: 83.691406%
TestSet Accuracy with best NN: 78.191489%
'''

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
# save weights of model from json file
def saveParams(params, path):
    with open(path, "w") as make_file:
        json.dump(params, make_file, cls=NumpyArrayEncoder)
    print("Done writing serialized NumPy array into file")

# load weights of model from json file
def loadParams(path):
    with open(path, "r") as read_file:
        print("Converting JSON encoded data into Numpy array")
        decodedArray = json.load(read_file)
    return decodedArray

def main():
    epochs = 10000
    test_rate = 100
    batch_size = 2 ** 7
    resume = False # path of model weights
    model_weights_path = 'weights.json'

    ### dataset loading 하기.
    dataPath = 'dataset/train'
    valPath = 'dataset/val'
    dataloader = dataUtils.Dataloader(dataPath, minibatch = batch_size)
    val_dataloader = dataUtils.Dataloader(valPath)
    
    # NeuralNetwork 생성
    layerDims = [len(dataloader.getImage(0)), 10, 10, 1]
    #simpleNN = model.NeuralNetwork(layerDims, learning_rate = 1.5)
    simpleNN = model.NeuralNetwork(layerDims, 
                                   learning_rate = 1.5,
                                   decay_rate = 0.2)
    if resume:
        simpleNN.parameters = loadParams(resume)

    for epoch in range (0, epochs, 1):
        training(dataloader, simpleNN, epoch)
        
        if epoch % test_rate == test_rate - 1: 
            rate = validation(val_dataloader, simpleNN)
            print('\nTest Accuracy: %.6f%%\n' %(rate * float(100)))

    print('------------------------------\n')
    rate = validation(dataloader, simpleNN)
    print('TrainSet Accuracy with best NN: %.6f%%' %(rate * float(100)))
    rate = validation(val_dataloader, simpleNN)
    print('TestSet Accuracy with best NN: %.6f%%' %(rate * float(100)))
    print('\n------------------------------\n')
    
    saveParams(simpleNN.parameters, model_weights_path)

def validation(dataloader, simpleNN):
    rate = 0
    cnt = 0
    
    for i, (images, targets) in enumerate(dataloader):
        p = simpleNN.predict(images)
        rate += simpleNN.hit_ratio(p, targets)
        cnt += 1
        
    return rate / cnt
        
def training(dataloader, simpleNN, epoch):
    min_cost = 10000000
    max_cost = -1
    min_rate = 101
    max_rate = -1
    
    for i, (images, targets) in enumerate(dataloader):
        train = simpleNN.forward(images)
        cost = simpleNN.compute_cost(train, targets)
        rate = simpleNN.hit_ratio(train, targets)[0]       
        simpleNN.backward()
        simpleNN.update_params(epoch)
        
        if min_cost > cost: min_cost = cost
        if max_cost < cost: max_cost = cost
        if min_rate > rate: min_rate = rate
        if max_rate < rate: max_rate = rate
        
    print("Cost     %7i| %.9f\t%.9f" %(epoch + 1, min_cost, max_cost))
    print("Accuracy\t| " + format(float(min_rate) * 100., "5.3f") + 
          '%\t' + format(float(max_rate) * 100., "5.3f") + '%')
    
    return simpleNN

if __name__=='__main__':
    main()