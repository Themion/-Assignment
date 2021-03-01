import cv2
import numpy as np
import math

def main():
    size = 1
    mode = 3
    kernel = cv2.getGaussianKernel(size, (0.3*((size-1)*0.5 - 1) + 0.8))
    
    def callback_size(x):
        nonlocal size
        nonlocal kernel
        size = x*2+1
        kernel = cv2.getGaussianKernel(size, (0.3*((size-1)*0.5 - 1) + 0.8))
        img_remake()

    def callback_mode(x):
        nonlocal mode
        mode = x
        img_remake()
            
    original = cv2.imread('L1.png')
    img = cv2.imread('L2.png')
    
    def img_remake():
        nonlocal size
        nonlocal mode
        nonlocal kernel

        print (mode, size)

        filt = cv2.filter2D(img, -1, kernel)
        blur = cv2.blur(img, (size, size))
        gau = cv2.GaussianBlur(img, (size, size), 0)
        med = cv2.medianBlur(img, size)
        err = [0, 0, 0, 0]
        err[0] = np.sum(abs(original - filt)) / (img.shape[0]*img.shape[1])
        err[1] = np.sum(abs(original - blur)) / (img.shape[0]*img.shape[1])
        err[2] = np.sum(abs(original - gau)) / (img.shape[0]*img.shape[1])
        err[3] = np.sum(abs(original - med)) / (img.shape[0]*img.shape[1])
        print(err)

        #Convolution
        if mode==0:
            cv2.imshow("image", np.hstack([original, img, filt])) #kernel: kernel array
        #Average
        elif mode==1:
            cv2.imshow("image", np.hstack([original, img, blur])) #(k, k): kernel shape
        #Gaussian
        elif mode==2:
            cv2.imshow("image", np.hstack([original, img, gau])) # 0: how to determine variance
        #Median
        else :
            cv2.imshow("image", np.hstack([original, img, med]))
            
        #print ((np.sum(original) - np.sum(blur))/(original.shape[0]*original.shape[1]*3))
        #cv2.imshow('image', blur)

    cv2.namedWindow('image')
    cv2.createTrackbar('Size', 'image', 0, 5, callback_size)
    cv2.createTrackbar('Filter', 'image', mode, 3, callback_mode)
    cv2.imshow("image", np.hstack([original, img, img]))

if __name__ == "__main__":
    main()
