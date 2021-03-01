# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:19:09 2019

@author: Kei
"""

import cv2
import numpy as np
import winsound as ws
from matplotlib import pyplot as plt


def main():
    imgpath = "C:\\Users\\Kei\\Desktop\\cg\\team\\1\\messi5.jpg"
    img = cv2.imread(imgpath)
    
    img_buf = img.copy()
    print(img)
    
    
    # event loop
    while(True):
        # display the image in the same window
        cv2.imshow('messi', img_buf)
        # 64bit os returns 32bits, mask out to get the ascii 8bits
        key = cv2.waitKey(0) & 0xFF     
        print(key)

        if key == ord('q'):   # esc: quit
            break
        
        # show the original image
        elif key == ord('c'):
            img_buf = img[:, :, :]
            
        elif key == ord('g'):
            img_buf = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        elif key == ord('m'):
            print("행을 나눌 칸 갯수를 입력하시오.")
            r = input()
            print("열을 나눌 칸 갯수를 입력하시오.")
            c = input()
            
            new = img.copy()
            new2 = img.copy()
            new3 = img.copy()
            print(img.shape[0], img.shape[1])
            r = int(r)
            c = int(c)
            
            T = img.shape[0]/r
            S = img.shape[1]/c
            T = int (T)
            S  = int (S)
            arsize = T*S
            print(T, S, "T, S")
            print("a.average b.median c.center")
            ikey = input()
            if ikey == ('a'):
                for k in range(r):
                    for f in range(c):
                        T2 = T*k
                        S2 = S*f
                        #print(k, f, T*(k), S*(f), "k f")
                        temp1 = 0
                        temp2 = 0
                        temp3 = 0
                        for i in range(T):
                            for j in range(S):
                                #print(i, j)
                                temp1 = temp1 + img[i+T2][j+S2][0]
                                temp2 = temp2 + img[i+T2][j+S2][1]
                                temp3 = temp3 + img[i+T2][j+S2][2]
                        temp1 = int(temp1 / arsize)
                        temp2 = int(temp2 / arsize)
                        temp3 = int(temp3 / arsize)
                        for i in range(T):
                            for j in range(S):
                                new[i+T2][j+S2][0] = temp1
                                new[i+T2][j+S2][1] = temp2
                                new[i+T2][j+S2][2] = temp3
                                if i == T-1 or j == S-1:
                                    new[i+T2][j+S2][0] = 0 #line Color
                                    new[i+T2][j+S2][1] = 0
                                    new[i+T2][j+S2][2] = 0
                                    
                img_buf = new
                plt.imshow(img_buf)
                for k in range(r):
                    for f in range(c):
                        T2 = T*k
                        S2 = S*f
                        plt.text(S2+int(S/2),T2+int(T/2),"({}, {})".format(f, k), color='white')
                plt.show()
            elif ikey == ('b'):
                for k in range(r):
                    for f in range(c):
                        T2 = T*k
                        S2 = S*f
                        list1 = []
                        list2 = []
                        list3 = []
                        for i in range(T):
                            for j in range(S):
                                list1.append(img[i+T2][j+S2][0])
                                list2.append(img[i+T2][j+S2][1])
                                list3.append(img[i+T2][j+S2][2])
                        list1.sort()
                        list2.sort()
                        list3.sort()
                        temp1 = list1[int(arsize/2)]
                        temp2 = list2[int(arsize/2)]
                        temp3 = list3[int(arsize/2)]
                        for i in range(T):
                            for j in range(S):
                                new2[i+T2][j+S2][0] = temp1
                                new2[i+T2][j+S2][1] = temp2
                                new2[i+T2][j+S2][2] = temp3
                                if i == T-1 or j == S-1:
                                    new2[i+T2][j+S2][0] = 0
                                    new2[i+T2][j+S2][1] = 0
                                    new2[i+T2][j+S2][2] = 0
                img_buf = new2
                plt.imshow(img_buf)
                for k in range(r):
                    for f in range(c):
                        T2 = T*k
                        S2 = S*f
                        plt.text(S2+int(S/2),T2+int(T/2),"({}, {})".format(f, k), color='white')
                plt.show()
            elif ikey == ('c'):    
                for k in range(r):
                    for f in range(c):
                        T2 = T*k
                        S2 = S*f
                        #print(k, f, T*(k), S*(f), "k f")
                        temp1 = img[T2+int(T/2)][S2+int(S/2)][0]
                        temp2 = img[T2+int(T/2)][S2+int(S/2)][1]
                        temp3 = img[T2+int(T/2)][S2+int(S/2)][2]
                        for i in range(T):
                            for j in range(S):
                                new3[i+T2][j+S2][0] = temp1
                                new3[i+T2][j+S2][1] = temp2
                                new3[i+T2][j+S2][2] = temp3
                                if i == T-1 or j == S-1:
                                    new3[i+T2][j+S2][0] = 0
                                    new3[i+T2][j+S2][1] = 0
                                    new3[i+T2][j+S2][2] = 0
                img_buf = new3
                plt.imshow(img_buf)
                for k in range(r):
                    for f in range(c):
                        T2 = T*k
                        S2 = S*f
                        plt.text(S2+int(S/2),T2+int(T/2),"({}, {})".format(f, k), color='white')
                plt.show()
            else:
                print("Wrong input")
                break
                
            
        else:   # defalut: warning
            ws.PlaySound("*", ws.SND_ALIAS)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
