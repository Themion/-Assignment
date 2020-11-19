'''
이번 과제는 사과와 바나나를 구별하는 네트워크를 만드는 것입니다.
데이터셋은 training 1024장, validation 188장으로 구성되어 있습니다.
training set은 사과 512장, 바나나 512장 으로 이루어져 있습니다.
'''
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os.path as osp

class Dataloader():
    '''
    Dataloader는 학습할 이미지의 경로를 받아 학습할 이미지와 정답을 출력합니다.
    minibatch: 학습할 데이터를 mini batch로 묶어 출력합니다.
    data augmentation을 추가하고 싶은 학생은 자유롭게 추가하여 학습해도 됩니다.
    '''
    def __init__(self, imagePath, minibatch=None, shuffle=False):
        
        self.imgList = glob.glob(imagePath+'/*')
        self.targets = [1 if osp.basename(path)[0]=='a' else 0 for path in self.imgList]

        if minibatch is None:
            self.step = len(self.imgList)
        else:
            self.step = minibatch
        self.size = self.len()

    def len(self):
        return len(self.imgList)

    def getImage(self, idx):
        try:
            img = Image.open(self.imgList[idx])
        except:
            print("Undable to find correspoding input image.")
        
        ## DATA transform
        # add more data augmentation if you need
        img = img.resize((50,50))
        # img.show()
        # data augmentation end

        img = np.array(img)
        img = img.reshape(1,-1).T # flatten image
        img = img/255  # normalize

        return img

    def getTarget(self,idx):
        return self.targets[idx]

    def __iter__(self):
        # self.index = 0
        self.index = np.array(range(self.step))
        return self

    def __next__(self):
        if self.index[-1] >= self.size:
            raise StopIteration
        
        imgs = np.concatenate([self.getImage(x) for x in self.index],axis=1)
        targets = np.array(list(map(self.getTarget,self.index)))
        self.index += self.step

        return imgs, targets
    

if __name__=='__main__':
    dataPath = 'dataset/train'
    valPath = 'dataset/val'
    
    dataloader = Dataloader(dataPath, minibatch=64)
    val_dataloader = Dataloader(valPath)

